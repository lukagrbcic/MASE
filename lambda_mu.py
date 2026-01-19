# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats.qmc as qmc
import sys


class LambdaMU:
    
    def __init__(self, f, lb, ub, n_evals, mu, tau=None, strategy='(mu,lambda)', seed=None):
        
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
        # 1. Initialize population and step sizes
        x = self._initialize()
        x_sigma = np.random.uniform(0.05, 0.1, self.mu)
        
        # 2. FIX: Evaluate initial population so 'fx' exists for the first (+) comparison
        fx = self._evaluate(x)
        self._archive(x, fx)
        
        # 3. Start counter with initial evaluations
        func_call = len(fx)
        
        while func_call < self.n_evals:
            x_lambda = []
            x_lambda_sigma = []
        
            # Generate offspring
            for i in range(self.lambda_):
                # Recombination (Discrete for X, Intermediate for Sigma)
                idx1, idx2 = np.random.choice(self.mu, 2, replace=False)
                x_new = np.where(np.random.rand(self.dim) < 0.5, x[idx1], x[idx2])
                x_new_sigma = (x_sigma[idx1] + x_sigma[idx2]) / 2.0
                
                # Mutation (Log-normal for Sigma, Gaussian for X)
                x_new_sigma *= np.exp(self.tau * np.random.randn())
                x_new_sigma = max(x_new_sigma, 1e-5) # Prevent sigma from hitting 0
                
                x_new += x_new_sigma * np.random.randn(self.dim)
                
                # DNA Drift Fix: Clip before saving to population
                x_new = np.clip(x_new, 0, 1)

                x_lambda.append(x_new)
                x_lambda_sigma.append(x_new_sigma)
            
            # Vectorized evaluation of the offspring
            x_lambda_arr = np.array(x_lambda)
            fx_lambda = self._evaluate(x_lambda_arr)
            func_call += len(fx_lambda)
            
            # Selection Strategy
            if self.strategy == '(mu,lambda)':
                # Comma strategy: parents are ignored, select only from offspring
                indices = np.argsort(fx_lambda)
                top_indices = indices[:self.mu]
                
                x = x_lambda_arr[top_indices]
                x_sigma = np.array(x_lambda_sigma)[top_indices]                
                fx = fx_lambda[top_indices]
            
            elif self.strategy == '(mu+lambda)':
                # Plus strategy: select from combined pool of parents + offspring
                combined_x = np.vstack((x, x_lambda_arr))
                combined_x_sigma = np.concatenate((x_sigma, np.array(x_lambda_sigma)))
                combined_fx = np.concatenate((fx, fx_lambda))
                
                indices = np.argsort(combined_fx)
                top_indices = indices[:self.mu]
                
                x = combined_x[top_indices]
                x_sigma = combined_x_sigma[top_indices]
                fx = combined_fx[top_indices]
            
            # Archive the best of the current generation
            self._archive(x, fx)
            
        # Denormalize the history of elites for the final return
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
        

            




        
        
        
        
        
        
        