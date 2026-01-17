# -*- coding: utf-8 -*-
import torch
import numpy as np
import scipy.stats.qmc as qmc



class Sampler:
    
    def __init__(self, f, lb, ub, n_evals, n_size, n_init, algorithm='random', seed=None):
        
        self.f = f
        self.lb = lb
        self.ub = ub
        self.n_evals = n_evals
        self.n_size = n_size
        self.n_init = n_init
        self.algorithm = algorithm
        self.seed = seed
        
        if self.seed is None:
            self.seed = np.random.randint(1, 1e5)
    
        self.dim = np.shape(self.lb)[1]
        
        self.X_elite = []
        self.fx_elite = []
        
        
        
    def archive_elite(self, X, FX):
        
        fx_min_idx = np.argsort(FX)[0]
        x_min = X[fx_min_idx]
        fx_min = FX[fx_min_idx]
        
        self.X_elite.append(x_min)
        self.fx_elite.append(fx_min)
        
        
        
        
        
        
        

    def evaluate(self, x):
        
        return self.f(x)

    def initialize(self):
            
        x = qmc.scale(qmc.LatinHypercube(d=self.dim, seed=self.seed).random(n=self.n_init), self.lb, self.ub)
        
        return x
    
    def search(self):
        
        x = self.initialize()
        fx = self.evaluate(x)
        
        X = np.copy(x)
        FX = np.copy(fx)
        
        while len(X) < self.n_evals:
            
            if self.method == 'random':
                x = np.random.unifrom(self.lb, self.ub, size=(self.n_size, np.shape(self.lb)[1]))
            
            elif self.method == 'lhs':
                x = qmc.scale(qmc.LatinHypercube(d=self.dim).random(n=self.n_size), self.lb, self.ub)
            
            elif self.method == 'sobol':
                x = qmc.scale(qmc.Sobol(d=self.dim).random(n=self.n_size), self.lb, self.ub)
                
            elif self.method == 'halton':
                x = qmc.scale(qmc.Sobol(d=self.dim).random(n=self.n_size), self.lb, self.ub)
        
            fx = self.evaluate(x)
            
            X = np.vstack((X,x))
            FX = np.append(fx)
        

        
        
        return
        
        
        
        
        
        
        
        