import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from time import time
import pandas as pd
from py4etrics.truncreg import Truncreg
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import truncnorm
import torch.distributions as dist
import gpytorch as gp


class TruncatedNormal:
    def __init__(self, device='cpu'):
        self.snd = dist.Normal(torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))
        self.device = device

    def get_logdelta(self, a, b):
        res = torch.full_like(a, np.nan, device=self.device)
        ids1 = torch.logical_and(a <= 30, b >= -30)
        ids11, ids12 = torch.logical_and(ids1, torch.where(a > 0, 1, 0)), \
                       torch.logical_and(ids1, torch.where(a <= 0, 1, 0))
        res[ids11] = self.snd.cdf(-a[ids11]) - self.snd.cdf(-b[ids11])
        res[ids12] = self.snd.cdf(b[ids12]) - self.snd.cdf(a[ids12])
        res[~torch.isnan(res)] = torch.max(res[~torch.isnan(res)], torch.zeros_like(res[~torch.isnan(res)]))
        res[res > 0] = torch.log(res[res > 0])

        ids2 = torch.logical_and(torch.where(res <= 0, 1, 0),
                                 torch.logical_or(b < 0, torch.where(torch.abs(a) >= torch.abs(b), 1, 0)))
        res[ids2] = gp.log_normal_cdf(-b[ids2]) + torch.log1p(
            -torch.exp(gp.log_normal_cdf(-a[ids2]) - gp.log_normal_cdf(-b[ids2])))

        ids3 = ~ids2
        res[ids3] = gp.log_normal_cdf(-a[ids3]) + \
                    torch.log1p(-torch.exp(gp.log_normal_cdf(-b[ids3]) - gp.log_normal_cdf(-a[ids3])))
        return res

    def log_prob(self, val, a, b, loc, scale):
        val_standard = (val - loc) / scale
        res = self.snd.log_prob(val_standard) - self.get_logdelta(a, b) - scale.log()
        res[torch.logical_or(val_standard < a, val_standard > b)] = -np.inf
        return res


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
        l_thred, r_thred = -np.inf if l_thred is None else l_thred, 1e2 if r_thred is None else r_thred
        # STEP 2: calculate init beta & sigma with TLS
        self.init_beta, self.init_sigma = [], []
        for yy, XX, zz in zip(y, X, mid_z):
            yy, XX = yy[zz == 1].numpy(), XX[zz == 1].numpy()
            bb = np.linalg.lstsq(XX, yy, rcond=None)[0]  # expectation
            ss = (yy - XX @ bb) @ (yy - XX @ bb) / yy.shape[0] if yy.shape[0] != 0 else 1  # variance
            self.init_beta.append(bb)
            self.init_sigma.append(ss)
        ## STEP 3: transfer to device
        self.y, self.X, self.z, self.l_thred, self.r_thred = y.to(device), X.to(device), z.to(device), l_thred, r_thred
        self.beta = nn.Parameter(torch.tensor(self.init_beta).to(device))
        self.sigma = nn.Parameter(torch.tensor(self.init_sigma).to(device))
        self.truncnorm = TruncatedNormal(device=device)

    def forward(self):
        Xb = torch.bmm(self.X, self.beta.unsqueeze(2)).squeeze()
        exp_sigma = torch.exp(self.sigma).view(-1, 1).repeat(1, self.X.shape[1])
        _l, _r = (self.l_thred - Xb) / exp_sigma, (self.r_thred - Xb) / exp_sigma

        log_prob = self.truncnorm.log_prob(val=self.y, a=_l, b=_r, loc=Xb, scale=exp_sigma)
        # tmp = truncnorm.logpdf(self.y.detach().numpy(), a=_l.detach().numpy(), b=_r.detach().numpy(),
        #                             loc=Xb.detach().numpy(), scale=exp_sigma.detach().numpy())
        #
        # if abs(np.linalg.norm(tmp - log_prob.detach().numpy())) / self.y.shape[0] > 1e-5:
        #     print(abs(np.linalg.norm(tmp - log_prob.detach().numpy())))
        #     tmp -= log_prob.detach().numpy()
        #     tmp[np.abs(tmp) <= 1e-5] = 0

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

    # np.random.seed(0)
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)

    bs, n_lights = 2, 100
    y, X = [], []
    for _ in range(bs):
        normal_gt = np.random.rand(3) - 0.5
        normal_gt = normal_gt / np.linalg.norm(normal_gt)
        L = np.random.rand(n_lights, 3)
        L[:, 0], L[:, 1] = L[:, 0] - 0.5, L[:, 1] - 0.5
        m = L @ normal_gt + np.random.normal(0, 0.01, n_lights)
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
    torch_res = torch_TLR(y, X, device='cpu', lr=1e-1, max_iter=1000, verbose=1)
    toc2 = time()

    print('\n---------------------------------\n')
    print('torch err is', np.linalg.norm(normal_gt - torch_res))
    print('stats err is', np.linalg.norm(normal_gt - stats_res))
    print('stats-torch difference:', np.linalg.norm(stats_res - torch_res))
    print('stats time is {},  torch time is {}'.format(toc1 - tic1, toc2 - tic2))


