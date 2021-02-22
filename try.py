import numpy as np
import torch
from scipy.stats import truncnorm
import math
from scipy.stats import norm
import torch.distributions as dist
import gpytorch as gp



def get_logdelta(a, b):
    if (a <= 30) and (b >= -30):
        if a > 0:
            delta = norm.cdf(-a) - norm.cdf(-b)
        else:
            delta = norm.cdf(b) - norm.cdf(a)
        delta = max(delta, 0)
        if delta > 0:
            return np.log(delta)
    if b < 0 or (np.abs(a) >= np.abs(b)):
        nla, nlb = norm.logcdf(-a), norm.logcdf(-b)
        logdelta = nlb + np.log1p(-np.exp(nla - nlb))
    else:
        sla, slb = norm.logcdf(-a), norm.logcdf(-b)
        logdelta = sla + np.log1p(-np.exp(slb - sla))
    return logdelta


def logpdf_scalar(x, a, b, scale):
    if x < a or x > b:
        return -np.inf
    shp = np.shape(x)
    x = np.atleast_1d(x)
    out = np.full_like(x, np.nan, dtype=np.double)
    condlta, condgtb = (x < a), (x > b)
    if np.any(condlta):
        np.place(out, condlta, -np.inf)
    if np.any(condgtb):
        np.place(out, condgtb, -np.inf)
    cond_inner = ~condlta & ~condgtb
    if np.any(cond_inner):
        _logdelta = get_logdelta(a, b)
        np.place(out, cond_inner, norm.logpdf(x[cond_inner]) - _logdelta)
    res = out[0] if (shp == ()) else out
    return res - np.log(scale)


def my_logpdf(val, loc, scale, a=None, b=None):
    _val = (val - loc) / scale
    res = []
    for _x, _a, _b, _s in zip(_val.flatten(), a.flatten(), b.flatten(), scale.flatten()):
        res.append(logpdf_scalar(_x, _a, _b, _s))
    return np.asarray(res).reshape(scale.shape)


class TruncatedNormal:
    def __init__(self, device='cpu', eps=1e-20):
        self.eps = eps
        self.INV_SQRT_2PI, self.INV_SQRT_2 = 1 / math.sqrt(2 * math.pi), 1 / math.sqrt(2)
        self.snd = dist.Normal(torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))

    def get_logdelta(self, a, b):
        res = torch.full_like(a, np.nan)
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
        # Phi_a, Phi_b = self.snd.cdf(a), self.snd.cdf(b)
        # log_delta = (Phi_b - Phi_a).clamp_min(self.eps).log()
        # print(self.get_logdelta(a, b))
        # print(log_delta)
        # print(torch.norm(log_delta - self.get_logdelta(a, b)))
        val_standard = (val - loc) / scale
        res = self.snd.log_prob(val_standard) - self.get_logdelta(a, b) - scale.log()
        res[torch.logical_or(val_standard < a, val_standard > b)] = -np.inf
        return res





##------------------------------------------------------------------------------------


if __name__ == '__main__':



    a = 0.1 * np.ones((2, 3))
    b = np.random.rand(2, 3) + 1.23434
    loc = np.random.rand(2, 3) + 1
    val = np.random.rand(2, 3) + 2
    scale = np.random.rand(2, 3) + 1
    # scale = np.ones((2, 3))

    res1 = TruncatedNormal().log_prob(val=torch.tensor(val), loc=torch.tensor(loc), scale=torch.tensor(scale),
                                      a=torch.tensor(a), b=torch.tensor(b)).numpy()

    res2 = my_logpdf(val=val, loc=loc, scale=scale, a=a, b=b)
    res3 = truncnorm.logpdf(x=val, loc=loc, scale=scale, a=a, b=b)

    print('\n-----------------------------\n')
    print(res1)
    print(res2)
    print(np.linalg.norm(res1 - res2), np.linalg.norm(res1 - res3), np.linalg.norm(res3 - res2))




    # a = np.random.rand(10)
    # b = np.random.rand(10)
    #
    # print('a is', a)
    # print('b is', b)
    #
    # res1 = np.asarray([get_logdelta(x ,y) for (x, y) in zip(a, b)])
    # print('should be', res1)
    #
    # print('-----------------------------')
    #
    # res2 = my_get_logdelta(torch.as_tensor(a), torch.as_tensor(b)).numpy()
    #
    # print(res2)
    #
    # print(np.linalg.norm(res1 - res2))


