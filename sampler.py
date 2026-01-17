# -*- coding: utf-8 -*-
import torch
import numpy as np
import scipy.stats.qmc as qmc



class Sampler:
    
    def __init__(self, f, lb, ub, n_evals, n_size, n_init, algorithm='lhs', seed=None):
        
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
    
    def initialize(self):
        
        dim = np.shape(self.lb)[1]
               
        x = qmc.LatinHypercube(d=dim, seed=self.seed).random(n=self.n_init)
        
    def evaluate(self):
        
        
        
        
        
        