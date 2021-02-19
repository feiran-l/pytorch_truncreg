import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from time import time
import pandas as pd
from scipy.stats import truncnorm
from py4etrics.truncreg import Truncreg
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
import math
from numbers import Number


CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {'a': constraints.real, 'b': constraints.real}
    support = constraints.real
    has_rsample = True

    def __init__(self, a, b, eps=1e-8, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError('Truncation bounds types are different')
        if any((self.a >= self.b).view(-1,).tolist()):
            raise ValueError('Incorrect truncation range')
        self._dtype_min_gt_0 = torch.tensor(torch.finfo(self.a.dtype).eps, dtype=self.a.dtype)
        self._dtype_max_lt_1 = torch.tensor(1 - torch.finfo(self.a.dtype).eps, dtype=self.a.dtype)
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        self._lpbb_m_lpaa_d_Z = (self._little_phi_b * self.b - self._little_phi_a * self.a) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value ** 2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)

    def expand(self, batch_shape, _instance=None):
        # TODO: it is likely that keeping temporary variables in private attributes violates the logic of this method
        raise NotImplementedError


class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        'loc': constraints.real,
        'scale': constraints.positive,
        'a': constraints.real,
        'b': constraints.real}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, a, b, eps=1e-8, validate_args=None):
        self.loc, self.scale, self.a, self.b = broadcast_all(loc, scale, a, b)
        a_standard = (a - self.loc) / self.scale
        b_standard = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a_standard, b_standard, eps=eps, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale ** 2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale


##-------------------------------------------------------------------------------


class MyTLR(nn.Module):
    def __init__(self, y, X, device, l_thred=None, r_thred=None):
        super(MyTLR, self).__init__()
        ## STEP 1: create the indicator
        z = torch.zeros(y.shape)
        if l_thred is not None:
            z[y <= l_thred] = -1
        if r_thred is not None:
            z[y >= r_thred] = 1
        l_z, mid_z, r_z = torch.where(z == -1, 1, 0), torch.where(z == 0, 1, 0), torch.where(z == 1, 1, 0)
        l_thred, r_thred = 0 if l_thred is None else l_thred, 0 if r_thred is None else r_thred
        # STEP 2: calculate init beta & sigma with TLS
        self.init_beta, self.init_sigma = [], []
        for yy, XX, zz in zip(y, X, mid_z):
            yy, XX = yy[zz == 1].numpy(), XX[zz == 1].numpy()
            bb = np.linalg.lstsq(XX, yy, rcond=None)[0]  # expectation
            ss = (yy - XX @ bb) @ (yy - XX @ bb) / yy.shape[0] if yy.shape[0] != 0 else 1  # variance
            self.init_beta.append(bb)
            self.init_sigma.append(ss)
        ## STEP 3: transfer to device
        self.l_z, self.mid_z, self.r_z = l_z.to(device), mid_z.to(device), r_z.to(device)
        self.y, self.X, self.z, self.l_thred, self.r_thred = y.to(device), X.to(device), z.to(device), l_thred, r_thred
        self.beta = nn.Parameter(torch.tensor(self.init_beta).to(device))
        self.sigma = nn.Parameter(torch.tensor(self.init_sigma).to(device))
        self.snd = dist.Normal(torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))

    def forward(self):
        Xb = torch.bmm(self.X, self.beta.unsqueeze(2)).squeeze()
        exp_sigma = torch.exp(self.sigma).view(-1, 1).repeat(1, self.X.shape[1])
        _l, _r = (self.l_thred - Xb) / exp_sigma, (self.r_thred - Xb) / exp_sigma

        log_prob = truncnorm.logpdf(self.y, a=_l, b=_r, loc=Xb, scale=exp_sigma)

        return torch.sum(log_prob, dim=-1)


def torch_TLR(y, X, device='cuda:0', lr=1e-1, max_iter=1000, tol=1e-5, verbose=-1):
    y, X = torch.tensor(y), torch.tensor(X)
    solver = MyTLR(y=y, X=X, device=device, l_thred=0)
    optimizer = Adam(solver.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, max_iter, eta_min=1e-3)
    best_beta, best_loss, prev_loss = None, float('inf'), float('inf')
    for _ in tqdm(range(max_iter)):
        optimizer.zero_grad()
        loss = -1 * solver().sum() / y.shape[0]
        loss.backward()
        optimizer.step()
        scheduler.step()
        if loss.item() < best_loss:
            best_beta, best_loss = solver.beta.data.detach().cpu().numpy(), loss.item()
        if abs(loss.item() - prev_loss) <= tol:
            break
        else:
            prev_loss = loss.item()
        if verbose > 0:
            print('curr loss val is:', loss.item())
    return best_beta


def TLR(data):
    formula = 'm ~ L0 + L1 + L2 - 1'  # -1 is for removing the intercept
    trunc_res = Truncreg.from_formula(formula, left=0, data=data.query('m > 0')).fit(method='cg')
    trunc_res = trunc_res.params[:-1]
    return trunc_res


##--------------------------------------------------------------------------------


if __name__ == '__main__':

    bs, n_lights = 10, 100
    y, X = [], []
    for _ in range(bs):
        normal_gt = np.random.rand(3) - 0.5
        normal_gt = normal_gt / np.linalg.norm(normal_gt)
        L = np.random.rand(n_lights, 3)
        L[:, 0], L[:, 1] = L[:, 0] - 0.5, L[:, 1] - 0.5
        m = L @ normal_gt + np.random.normal(0, 0.1, n_lights)
        m[m < 0] = 0
        y.append(m)
        X.append(L)
    y, X = np.asarray(y), np.asarray(X)

    """ statsmodel """
    tic1 = time()
    stats_res = []
    for i in range(y.shape[0]):
        data = pd.DataFrame({'m': y[i], 'L0': X[i, :, 0], 'L1': X[i, :, 1], 'L2': X[i, :, 2]})
        stats_res.append(TLR(data))
    stats_res = np.asarray(stats_res)
    toc1 = time()

    """ pytorch: change the lr according to your data! """
    tic2 = time()
    torch_res = torch_TLR(y, X, device='cuda:0', lr=1e-1, max_iter=1000, verbose=0)
    toc2 = time()

    print('\n---------------------------------\n')
    print('torch err is', np.linalg.norm(normal_gt - torch_res))
    print('stats err is', np.linalg.norm(normal_gt - stats_res))
    print('stats-torch difference:', np.linalg.norm(stats_res - torch_res))
    print('stats time is {},  torch time is {}'.format(toc1 - tic1, toc2 - tic2))


