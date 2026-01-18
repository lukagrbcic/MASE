# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats.qmc as qmc
import sys


class LambdaMU:
    
    def __init__(self, f, lb, ub, n_evals, mu, tau=None, strategy='', seed=None):
        
        self.f = f
        self.lb = lb
        self.ub = ub
        self.n_evals = n_evals
        self.mu = mu
        self.strategy = strategy
        self.seed = seed
        self.tau = tau
        
        if self.seed is None:
            self.seed = np.random.randint(1, 1e5)
    
        self.dim = len(self.lb)
        self.lambda_ = int(7*mu)
        
        if self.tau is None:
            self.tau = 1/np.sqrt(2*self.dim)
        else:
            self.tau = tau

        
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
    
    def _denormalize(self, x_n):
        
        x = x_n * (self.ub - self.lb) + self.lb
        x = np.clip(x, self.lb, self.ub)
        
        return x
        
    
    def _evaluate(self, x):
        
        x = self._denormalize(x)
              
        
        return self.f(x)

    def _initialize(self):
            
        x = qmc.LatinHypercube(d=self.dim, seed=self.seed).random(n=self.mu)
        
        return x
    
    def search(self):
        
        x = self._initialize()
        x_sigma = np.random.uniform(0.05, 0.1, self.mu)
               
        fx = self._evaluate(x)
        self._archive(x, fx)
        
        func_call = len(fx)
        
        while func_call < self.n_evals - self.mu:
            
            x_lambda = []
            x_lambda_sigma = []
            #fx_lambda = []
        
            for i in range(self.lambda_):
                
                """select two from x"""
                idx1, idx2 = np.random.choice(self.mu, 2, replace=False)
                
                x_new = np.where(np.random.rand(self.dim) < 0.5, x[idx1], x[idx2])
                x_new_sigma = (x_sigma[idx1] + x_sigma[idx2]) / 2.0
                
                # -- Mutation --
                # Mutate sigma first (log-normal mutation)
                x_new_sigma *= np.exp(self.tau * np.random.randn())
                x_new_sigma = max(x_new_sigma, 1e-5)
                x_new += x_new_sigma * np.random.randn(self.dim)
                x_new = np.clip(x_new, 0, 1)

                x_lambda.append(x_new)
                x_lambda_sigma.append(x_new_sigma)
            
            
            fx_lambda = self._evaluate(np.array(x_lambda))
            # print (len(fx_lambda))
            func_call += len(fx_lambda)
                    

            indices = np.argsort(fx_lambda)
            top_indices = indices[:self.mu]
        
            x = np.array(x_lambda)[top_indices]
            x_sigma = np.array(x_lambda_sigma)[top_indices]                
            fx = np.array(fx_lambda)[top_indices]
            
            self._archive(x, fx)
        
        X_elite_denorm = self._denormalize(self.X_elite)

            
        return X_elite_denorm, self.fx_elite
    
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
        

            




        
        
        
        
        
        
        