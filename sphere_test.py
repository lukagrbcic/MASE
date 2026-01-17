# -*- coding: utf-8 -*-
import numpy as np
from sampler import Sampler
import matplotlib.pyplot as plt

def sphere(x):
    x = np.asarray(x)
    # axis=-1 sums across the last dimension (the coordinates)
    return np.sum(x**2, axis=-1)


dim=2
lb = np.ones(dim)*-5.12
ub = np.ones(dim)*5.12

f = sphere 
n_evals = 1024
n_init = 100
n_size = 10
seed=111
runs=50

plt.figure(figsize=(6,5))
optimizer_random = Sampler(f, lb, ub, n_evals, n_size, n_init, seed=seed)
x_, fx_ = optimizer_random.n_runs(n=50)
plt.plot(np.arange(0, len(fx_[-1]), 1), np.mean(fx_, axis=0), label='Random')


optimizer_halton = Sampler(f, lb, ub, n_evals, n_size, n_init, method='halton', seed=seed)
x_, fx_ = optimizer_halton.n_runs(n=50)
plt.plot(np.arange(0, len(fx_[-1]), 1), np.mean(fx_, axis=0), label='Halton')


optimizer_sobol = Sampler(f, lb, ub, n_evals, n_size, n_init, method='sobol', seed=seed)
x_, fx_ = optimizer_sobol.n_runs(n=50)
plt.plot(np.arange(0, len(fx_[-1]), 1), np.mean(fx_, axis=0), label='Sobol')

optimizer_lhs = Sampler(f, lb, ub, n_evals, n_size, n_init, method='lhs', seed=seed)
x_, fx_ = optimizer_lhs.n_runs(n=50)
plt.plot(np.arange(0, len(fx_[-1]), 1), np.mean(fx_, axis=0), label='LHS')
plt.legend()
