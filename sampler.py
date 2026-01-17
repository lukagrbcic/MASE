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
                
        self.archive_elite(X, FX)
        
        while len(X) < self.n_evals:
            
            if self.method == 'random':
                x = np.random.uniform(self.lb, self.ub, size=(self.n_size, self.dim))
            
            elif self.method == 'lhs':
                x = qmc.scale(qmc.LatinHypercube(d=self.dim).random(n=self.n_size), self.lb, self.ub)
            
            elif self.method == 'sobol':
                x = qmc.scale(qmc.Sobol(d=self.dim).random(n=self.n_size), self.lb, self.ub)
                
            elif self.method == 'halton':
                x = qmc.scale(qmc.Halton(d=self.dim).random(n=self.n_size), self.lb, self.ub)
        
            fx = self.evaluate(x)
                        
            X = np.vstack((X,x))
            FX = np.vstack((FX, fx))
            
            # self.archive_elite(X, FX)
            
            #print (self.X_elite)
            #print (self.fx_elite)
            print (len(X))
            # print (FX)
            
            # sys.exit()
                
        
        # return np.array(self.X_elite[-1]), self.fx_elite[-1], np.array(self.X_elite), np.array(self.fx_elite)
        
        
        
        
        
        
        
        