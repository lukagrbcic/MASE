# -*- coding: utf-8 -*-
import numpy as np
from sampler import Sampler
from lambda_mu import LambdaMU
import matplotlib.pyplot as plt
import sys
from indago import *

def sphere(x):
    x = np.asarray(x)
    # axis=-1 sums across the last dimension (the coordinates)
    return np.sum(x**2, axis=-1)

def rastrigin(x):
    
    x = np.asarray(x)
    
    d = x.shape[-1]
        
    return 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=-1)


def rastrigin_shifted(x):
    x = np.asarray(x)
    shift = 2.2#np.array([2.7, -1.3]) 
    d = x.shape[-1]
    z = x - shift
    return 10 * d + np.sum(z**2 - 10 * np.cos(2 * np.pi * z), axis=-1)

def ackley(x):
    
    x = np.asarray(x)
    
    d = x.shape[-1]
    
    a = 20
    b = 0.2
    c = 2 * np.pi
    
    sum_sq = np.sum(x**2, axis=-1)
    sum_cos = np.sum(np.cos(c * x), axis=-1)
    
    return (
        -a * np.exp(-b * np.sqrt(sum_sq / d))
        - np.exp(sum_cos / d)
        + a
        + np.e
    )


dim=200
lb = np.ones(dim)*-5#5.12
ub = np.ones(dim)*10#5.12

f = ackley
n_evals = 10000
n_init = 5
n_size = 5
seed=111
runs=15



plt.figure(figsize=(6,5))
optimizer_lmu = LambdaMU(f, lb, ub, n_evals, mu=n_size)
x_dyn, fx_dyn = optimizer_lmu.n_runs(n=runs)
plt.plot(np.arange(0, len(fx_dyn[-1]), 1), np.mean(fx_dyn, axis=0), label='(mu,lambda, dyn tau,')


optimizer_lmu = LambdaMU(f, lb, ub, n_evals, mu=n_size, strategy='(mu+lambda)')
x_plus, fx_plus = optimizer_lmu.n_runs(n=runs)
plt.plot(np.arange(0, len(fx_plus[-1]), 1), np.mean(fx_plus, axis=0), label='(mu+lambda), dyn tau,')


# optimizer_lmu = LambdaMU(f, lb, ub, n_evals, mu=n_size, tau=0.01)
# x_fixed, fx_fixed = optimizer_lmu.n_runs(n=runs)
# plt.plot(np.arange(0, len(fx_fixed[-1]), 1), np.mean(fx_fixed, axis=0), label='(mu,lambda), 0.01 tau')

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
# plt.ylim(0, 16)
plt.legend()

# opt = FWA()
# opt.lb = lb
# opt.ub = ub
# opt.evaluation_function = ackley
# opt.max_evaluations = n_evals
# run = opt.optimize()
# print (run.f)

