# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats.qmc as qmc
import sys


class Sampler:
    
    def __init__(self, f, lb, ub, n_evals, n_size, n_init, method='random', seed=None):
        
        self.f = f
        self.lb = lb
        self.ub = ub
        self.n_evals = n_evals
        self.n_size = n_size
        self.n_init = n_init
        self.method = method
        self.seed = seed
        
        if self.seed is None:
            self.seed = np.random.randint(1, 1e5)
    
        self.dim = len(self.lb)
        
        self.X_elite = []
        self.fx_elite = []
        
            
    def _archive(self, X, FX):
        
        best_idx = np.argmin(FX)
        current_best_f = FX[best_idx]
        current_best_x = X[best_idx]
    
        if not self.fx_elite:
            self.fx_elite.append(current_best_f)
            self.X_elite.append(current_best_x)
        else:
            prev_best_f = self.fx_elite[-1]
            
            if current_best_f < prev_best_f:
                self.fx_elite.append(current_best_f)
                self.X_elite.append(current_best_x)
            else:
                self.fx_elite.append(prev_best_f)
                self.X_elite.append(self.X_elite[-1])
            
    def _evaluate(self, x):
        
        return self.f(x)

    def _initialize(self):
            
        x = qmc.scale(qmc.LatinHypercube(d=self.dim, seed=self.seed).random(n=self.n_init), self.lb, self.ub)
        
        return x
    
    def _generate(self):
        
        if self.method == 'random':
            x = np.random.uniform(self.lb, self.ub, size=(self.n_size, self.dim))
        
        elif self.method == 'lhs':
            x = qmc.scale(qmc.LatinHypercube(d=self.dim).random(n=self.n_size), self.lb, self.ub)
        
        elif self.method == 'sobol':
            x = qmc.scale(qmc.Sobol(d=self.dim).random(n=self.n_size), self.lb, self.ub)
            
        elif self.method == 'halton':
            x = qmc.scale(qmc.Halton(d=self.dim).random(n=self.n_size), self.lb, self.ub)
        
        return x
    
    def search(self):
        
        x = self._initialize()
        fx = self._evaluate(x)
        self._archive(x, fx)
        
        while len(self.X_elite) < self.n_evals:
            
            x = self._generate()
            fx = self._evaluate(x)
            self._archive(x, fx)
            
        return self.X_elite, self.fx_elite
    
    def n_runs(self, n=1):
        
        x_n = []
        fx_n = []
        
        for i in range(n):
            x_elite, fx_elite = self.search()
            x_n.append(x_elite)
            fx_n.append(fx_elite)
            
            self.X_elite = []
            self.fx_elite = []
            
        x_n = np.array(x_n)
        fx_n = np.array(fx_n)
                
        return x_n, fx_n
        

            




        
        
        
        
        
        
        