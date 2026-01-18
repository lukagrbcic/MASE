# -*- coding: utf-8 -*-
import numpy as np
from sampler import Sampler
from lambda_mu import LambdaMU
import matplotlib.pyplot as plt
import sys

def sphere(x):
    x = np.asarray(x)
    # axis=-1 sums across the last dimension (the coordinates)
    return np.sum(x**2, axis=-1)

def rastrigin(x):
    
    x = np.asarray(x)
    
    d = x.shape[-1]
        
    return 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=-1)

dim=10
lb = np.ones(dim)*-3
ub = np.ones(dim)*5.12

f = rastrigin
n_evals = 1000
n_init = 10
n_size = 10
seed=111
runs=20



plt.figure(figsize=(6,5))
optimizer_lmu = LambdaMU(f, lb, ub, n_evals, mu=n_size, seed=seed)
x_, fx_ = optimizer_lmu.n_runs(n=runs)
plt.plot(np.arange(0, len(fx_[-1]), 1), np.mean(fx_, axis=0), label='(Lambda, mu), dyn tau')

optimizer_lmu = LambdaMU(f, lb, ub, n_evals, mu=n_size, tau=0.01, seed=seed)
x_, fx_ = optimizer_lmu.n_runs(n=runs)
plt.plot(np.arange(0, len(fx_[-1]), 1), np.mean(fx_, axis=0), label='(Lambda, mu), 0.01 tau')

# optimizer_random = Sampler(f, lb, ub, n_evals, n_size, n_init, seed=seed)
# x_, fx_ = optimizer_random.n_runs(n=runs)
# plt.plot(np.arange(0, len(fx_[-1]), 1), np.mean(fx_, axis=0), label='Random')

# optimizer_halton = Sampler(f, lb, ub, n_evals, n_size, n_init, method='halton', seed=seed)
# x_, fx_ = optimizer_halton.n_runs(n=50)
# plt.plot(np.arange(0, len(fx_[-1]), 1), np.mean(fx_, axis=0), label='Halton')

# optimizer_sobol = Sampler(f, lb, ub, n_evals, n_size, n_init, method='sobol', seed=seed)
# x_, fx_ = optimizer_sobol.n_runs(n=50)
# plt.plot(np.arange(0, len(fx_[-1]), 1), np.mean(fx_, axis=0), label='Sobol')

# optimizer_lhs = Sampler(f, lb, ub, n_evals, n_size, n_init, method='lhs', seed=seed)
# x_, fx_ = optimizer_lhs.n_runs(n=50)
# plt.plot(np.arange(0, len(fx_[-1]), 1), np.mean(fx_, axis=0), label='LHS')
plt.legend()
