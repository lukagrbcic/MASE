# -*- coding: utf-8 -*-
import numpy as np
import openai





class Agent:
    
    def __init__(self, prompt, model):
        
        self.prompt = prompt
        self.model = model
        
    def query(self):
        
        response = 'This is a reply to the prompt'
        
        return response
    
    
        