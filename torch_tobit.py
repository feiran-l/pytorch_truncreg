import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import gpytorch as gp
from time import time
import pandas as pd
from py4etrics.tobit import Tobit
from torch.optim.lr_scheduler import CosineAnnealingLR



def tobit(data):
    formula = 'm ~ L0 + L1 + L2 - 1'  # -1 is for removing the intercept
    censor = data['m'].apply(lambda x: -1 if x == 0 else 0)
    tobit_res = Tobit.from_formula(formula, cens=censor, left=0, data=data).fit(method='cg')
    tobit_res = tobit_res.params[:-1]
    return tobit_res


class MyTobit(nn.Module):
    def __init__(self, y, X, device, l_thred=None, r_thred=None):
        super(MyTobit, self).__init__()
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
        l_mle = self.l_z * gp.log_normal_cdf((self.l_thred - Xb) / exp_sigma)
        mid_mle = self.mid_z * (self.snd.log_prob((self.y - Xb) / exp_sigma) - self.sigma.view(-1, 1))
        r_mle = self.r_z * gp.log_normal_cdf((Xb - self.r_thred) / exp_sigma)
        return torch.sum(l_mle + mid_mle + r_mle, dim=-1)


def torch_tobit(y, X, device='cuda:0', lr=1e-1, max_iter=1000, tol=1e-5, verbose=-1):
    y, X = torch.tensor(y), torch.tensor(X)
    solver = MyTobit(y=y, X=X, device=device, l_thred=0)
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
        stats_res.append(tobit(data))
    stats_res = np.asarray(stats_res)
    toc1 = time()

    """ pytorch: change the lr according to your data! """
    tic2 = time()
    torch_res = torch_tobit(y, X, device='cuda:0', lr=1e-1, max_iter=1000, verbose=0)
    toc2 = time()

    print('\n---------------------------------\n')
    print('torch err is', np.linalg.norm(normal_gt - torch_res))
    print('stats err is', np.linalg.norm(normal_gt - stats_res))
    print('stats-torch difference:', np.linalg.norm(stats_res - torch_res))
    print('stats time is {},  torch time is {}'.format(toc1 - tic1, toc2 - tic2))


