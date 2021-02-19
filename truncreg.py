import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from py4etrics.truncreg import Truncreg  # pip install git+https://github.com/spring-haru/py4etrics.git
from py4etrics.tobit import Tobit



def plot_left_trunc_3d(A, b, thred):
    mesh_x = np.meshgrid(np.linspace(0, 1, 25), np.linspace(0, 1, 25))
    trunc_plane = np.transpose([[thred for x in slice] for slice in np.transpose(mesh_x)])
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    A_trunc, b_trunc = A[b > thred], b[b > thred]
    A_ignore, b_ignore = A[b <= thred], b[b <= thred]
    ax.scatter(A_trunc[:, 0], A_trunc[:, 1], b_trunc, c='r', alpha=0.3)
    ax.scatter(A_ignore[:, 0], A_ignore[:, 1], b_ignore, c='b', alpha=0.3)
    ax.plot_surface(*mesh_x, trunc_plane, color='b', alpha=0.3)
    plt.show()


def ang_err(gt, est):
    def rad2deg(a):
        return a / np.pi * 180.
    gt, est = gt / np.linalg.norm(gt), est / np.linalg.norm(est)
    x = gt.dot(est)
    if x > 1.0:
        x = 1.0
    return rad2deg(np.arccos(x))


def OLS(data, normal_gt):
    formula = 'm ~ L0 + L1 + L2 - 1'  # -1 is for removing the intercept
    ols_res = smf.ols(formula, data=data.query('m > 0')).fit()
    ols_res = np.asarray([x for x in ols_res.params])
    return ang_err(normal_gt, ols_res)

def tobit(data, normal_gt):
    formula = 'm ~ L0 + L1 + L2 - 1'  # -1 is for removing the intercept
    censor = data['m'].apply(lambda x: -1 if x == 0 else 0)
    tobit_res = Tobit.from_formula(formula, cens=censor, left=0,  data=data).fit(method='cg')
    tobit_res = tobit_res.params[:-1]
    return ang_err(normal_gt, tobit_res)

def truncate(data, normal_gt):
    formula = 'm ~ L0 + L1 + L2 - 1'  # -1 is for removing the intercept
    trunc_res = Truncreg.from_formula(formula, left=0, data=data.query('m > 0')).fit(method='cg')
    trunc_res = trunc_res.params[:-1]
    return ang_err(normal_gt, trunc_res)


def test_n_lights():
    num_lights = np.arange(50, 200, 20)
    normal_gt = np.random.rand(3) - 0.5
    normal_gt = normal_gt / np.linalg.norm(normal_gt)
    rho_gt = 1.0
    res = []
    for n_lights in num_lights:
        tmp_res = []
        for count in range(10):
            L = np.random.rand(n_lights, 3)
            L[:, 0], L[:, 1] = L[:, 0] - 0.5, L[:, 1] - 0.5
            m = rho_gt * L @ normal_gt + np.random.normal(0, 0.1, n_lights)
            m[m < 0] = 0
            data = pd.DataFrame({'m': m, 'L0': L[:, 0], 'L1': L[:, 1], 'L2': L[:, 2]})
            err_ols, err_cen, err_trun = OLS(data, normal_gt), tobit(data, normal_gt), truncate(data, normal_gt)
            tmp_res.append([err_ols, err_cen, err_trun])
        tmp_res = np.mean(tmp_res, axis=0)
        res.append(tmp_res)
    return num_lights, np.asarray(res)  # len(num_lights) * 3



def test_variance():
    variances = np.arange(0.001, 0.01, 0.002)
    n_lights = 100
    normal_gt = np.random.rand(3) - 0.5
    normal_gt = normal_gt / np.linalg.norm(normal_gt)
    rho_gt = 1.0
    L = np.random.rand(n_lights, 3)
    L[:, 0], L[:, 1] = L[:, 0] - 0.5, L[:, 1] - 0.5
    res = []
    for v in variances:
        tmp_res = []
        for count in range(10):
            m = rho_gt * L @ normal_gt + np.random.normal(0, v, n_lights)
            m[m < 0] = 0
            data = pd.DataFrame({'m': m, 'L0': L[:, 0], 'L1': L[:, 1], 'L2': L[:, 2]})
            err_ols, err_cen, err_trun = OLS(data, normal_gt), tobit(data, normal_gt), truncate(data, normal_gt)
            tmp_res.append([err_ols, err_cen, err_trun])
        tmp_res = np.mean(tmp_res, axis=0)
        res.append(tmp_res)
    return variances, np.asarray(res)  # len(num_lights) * 3



if __name__ == '__main__':
    np.random.seed(0)

    # """ test variance """
    # inds, res = test_variance()
    # inds = ['{:.1f}'.format(x) for x in inds]
    #
    # fontsize, bar_width = 20, 0.2
    # x = np.arange(len(inds))
    # fig, ax = plt.subplots(figsize=(8, 6))
    # rects1 = ax.bar(x - bar_width, res[:, 0], bar_width, label='Trunc-OLS')
    # rects2 = ax.bar(x, res[:, 1], bar_width, label='Censored-MLE')
    # rects4 = ax.bar(x + bar_width, res[:, 2], bar_width, label='Trunc-MLE')
    # plt.legend(loc='upper right', fontsize=fontsize - 4)
    # ax.set_ylabel('Ang. err', fontsize=fontsize)
    # ax.set_xlabel('Variance', fontsize=fontsize)
    # plt.xticks(np.arange(len(inds)), inds, fontsize=fontsize - 4)
    # plt.yticks(fontsize=fontsize - 4)
    # plt.tight_layout()
    # plt.show()

    """ test num_lights """
    # inds, res = test_n_lights()
    # inds = ['{:.0f}'.format(x) for x in inds]
    #
    # fontsize, bar_width = 20, 0.2
    # x = np.arange(len(inds))
    # fig, ax = plt.subplots(figsize=(8, 6))
    # rects1 = ax.bar(x - bar_width, res[:, 0], bar_width, label='Trunc-OLS')
    # rects2 = ax.bar(x, res[:, 1], bar_width, label='Censored-MLE')
    # rects4 = ax.bar(x + bar_width, res[:, 2], bar_width, label='Trunc-MLE')
    # plt.legend(loc='upper right', fontsize=fontsize - 4)
    # ax.set_ylabel('Ang. err', fontsize=fontsize)
    # ax.set_xlabel('num_lights', fontsize=fontsize)
    # plt.xticks(np.arange(len(inds)), inds, fontsize=fontsize - 4)
    # plt.yticks(fontsize=fontsize - 4)
    # plt.tight_layout()
    # plt.show()


    # np.random.seed(0)
    # x = np.random.rand(100) - 0.5
    # y = 2 * x + np.random.normal(0, 0.1, 100)
    # x_trunc, y_trunc = x[y > 0], y[y > 0]
    # x_hidden, y_hidden = x[y <= 0], np.zeros(x.shape[0] - x_trunc.shape[0])
    # fontsize, bar_width = 20, 0.2
    # fig, ax = plt.subplots(figsize=(8, 6))
    # rects1 = ax.scatter(x, y, alpha=0.3)
    # rects2 = ax.scatter(x_trunc, y_trunc, alpha=1, c='r')
    # rects3 = ax.scatter(x_hidden, y_hidden, alpha=1, c='r')
    # plt.show()

    n_lights = 100
    normal_gt = np.random.rand(3) - 0.5
    normal_gt = normal_gt / np.linalg.norm(normal_gt)
    rho_gt = 1.0
    L = np.random.rand(n_lights, 3)
    L[:, 0], L[:, 1] = L[:, 0] - 0.5, L[:, 1] - 0.5

    m = rho_gt * L @ normal_gt + np.random.normal(0, 0.1, n_lights)
    m[m < 0] = 0
    data = pd.DataFrame({'m': m, 'L0': L[:, 0], 'L1': L[:, 1], 'L2': L[:, 2]})
    err_cen = tobit(data, normal_gt)


