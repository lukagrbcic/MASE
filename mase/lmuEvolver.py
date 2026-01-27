# -*- coding: utf-8 -*-
import numpy as np
import random
import time
from concurrent.futures import ThreadPoolExecutor
import openai
import os
import re as re
import threading

# ===================================================================
# START: Your provided client code and LLM class
# ===================================================================

client = openai.OpenAI(
    api_key = os.environ.get('CBORG_API_KEY'),
    base_url = "https://api.cborg.lbl.gov"
)

class LLM:
    def __init__(self, query, model, temperature=0.75):
        self.query = query
        self.model = model
        self.temperature = temperature
        
    def get_response(self):
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages = [{"role": "user", "content": self.query}],
                temperature=self.temperature
            )
            model_response = response.choices[-1].message.content
            return model_response
        except Exception as e:
            print(f"Error calling model {self.model}: {e}")
            return None
    
    def get_code(self, response):
        match = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        sections = response.split("```")
        if len(sections) > 1:
            return sections[1].replace('python', '').strip()
        raise IndexError("Could not extract code from response.")

# ===================================================================
# END: Your provided client code and LLM class
# ===================================================================

# ===================================================================
# START: Real Evaluation Server for Symbolic Regression
# ===================================================================

TARGET_DATA = {
    -5: 53, -4: 39, -3: 27, -2: 17, -1: 9, 0: 3, 
    1: -1, 2: -3, 3: -3, 4: -1, 5: 3
}
TARGET_FUNCTION_NAME = "model_function"

def evaluation_server(code_string: str) -> float:
    if not code_string or not code_string.strip():
        return float('inf')
    total_squared_error = 0
    try:
        local_scope = {}
        exec(code_string, {}, local_scope)
        if TARGET_FUNCTION_NAME not in local_scope:
            return float('inf')
        model_func = local_scope[TARGET_FUNCTION_NAME]
        for x, y_true in TARGET_DATA.items():
            y_pred = model_func(x)
            total_squared_error += (y_pred - y_true)**2
        mse = total_squared_error / len(TARGET_DATA)
        print(f"Evaluated successfully. Fitness (MSE): {mse:.4f}")
        return mse
    except Exception:
        return float('inf')

# ===================================================================
# END: Real Evaluation Server
# ===================================================================


class LLMAgentEvolver:
    
    def __init__(self, problem_description, model_name, n_queries, mu, strategy='(mu,lambda)'):

        self.problem_description = problem_description
        self.model_name = model_name
        self.n_queries = n_queries
        self.mu = mu
        self.strategy = strategy


        self.lambda_ = int(2*mu)
        self.best_agents_history = []
        
        # --- Counters and Locks for Parallel Execution ---
        self.query_calls = 0
        self.query_lock = threading.Lock()

    def _archive(self, population):
        current_best_agent = population[0]
        if not self.best_agents_history or current_best_agent['fitness'] < self.best_agents_history[-1]['fitness']:
            self.best_agents_history.append(current_best_agent)
        else:
            self.best_agents_history.append(self.best_agents_history[-1])

    def _evaluate_population(self, population):
        codes_to_evaluate = [agent['code'] for agent in population]
        with ThreadPoolExecutor(max_workers=10) as executor:
            fitness_scores = list(executor.map(evaluation_server, codes_to_evaluate))
        for agent, fitness in zip(population, fitness_scores):
            agent['fitness'] = fitness
        return population

    # --- THIS METHOD IS NOW CORRECTLY INSIDE THE CLASS ---
    def _llm_query(self, prompt):
        """Internal, thread-safe method to query the LLM and count the call."""
        with self.query_lock:
            if self.query_calls >= self.n_queries:
                return "" # Stop if budget is exceeded
            self.query_calls += 1
            current_query_num = self.query_calls
        
        print(f"--- LLM Query #{current_query_num}/{self.n_queries} ---")
        
        try:
            llm_instance = LLM(query=prompt, model=self.model_name, temperature=0.7)
            response_text = llm_instance.get_response()
            if response_text is None: return ""
            return llm_instance.get_code(response_text)
        except IndexError:
            # print("Warning: Could not find code block '```' in LLM response.")
            return response_text.strip() if response_text else ""
        except Exception as e:
            print(f"An unexpected error occurred in _llm_query: {e}")
            return ""

    def _initialize_population(self):
        print(f"Initializing population of size {self.mu}...")
        # Use a ThreadPool to initialize in parallel
        prompts = [self.problem_description] * self.mu
        with ThreadPoolExecutor(max_workers=10) as executor:
            initial_codes = list(executor.map(self._llm_query, prompts))
        
        population = [{'code': code, 'fitness': None} for code in initial_codes if code]
        return self._evaluate_population(population)

    def _recombination_worker(self, parent_pair):
        """Worker function to perform recombination for a pair of parents."""
        p1, p2 = parent_pair
        recombine_prompt = f"Here are two Python solutions for a problem. Combine their best ideas to create a superior one. Output only the raw code.\n\nSolution A (fitness: {p1['fitness']:.4f}):\n```python\n{p1['code']}\n```\n\nSolution B (fitness: {p2['fitness']:.4f}):\n```python\n{p2['code']}\n```"
        return self._llm_query(recombine_prompt)

    def _mutation_worker(self, recombined_code):
        """Worker function to perform mutation on a single code string."""
        if not recombined_code: return ""
        mutate_prompt = f"Critically analyze and improve this Python code. Fix bugs, improve logic, or make it more accurate. Output only the raw, improved code.\n\nCode:\n```python\n{recombined_code}\n```"
        return self._llm_query(mutate_prompt)
    
    def search(self):
        MAX_LLM_WORKERS = 10
        parents = self._initialize_population()
        if not parents:
            print("Initialization failed to produce any valid code. Stopping.")
            return []
            
        parents.sort(key=lambda x: x.get('fitness', float('inf')))
        self._archive(parents)
        
        generation = 0
        
        while self.query_calls < self.n_queries:
            generation += 1
            print(f"\n--- Generation {generation} | LLM Queries: {self.query_calls}/{self.n_queries} ---")
            
            queries_left = self.n_queries - self.query_calls
            offspring_to_generate = min(self.lambda_, queries_left // 2)
            
            if offspring_to_generate <= 0:
                print("Query budget exhausted. Ending evolution.")
                break

            parent_pairs = [random.sample(parents, 2) for _ in range(offspring_to_generate)]
            with ThreadPoolExecutor(max_workers=MAX_LLM_WORKERS) as executor:
                recombined_codes = list(executor.map(self._recombination_worker, parent_pairs))

            with ThreadPoolExecutor(max_workers=MAX_LLM_WORKERS) as executor:
                mutated_codes = list(executor.map(self._mutation_worker, recombined_codes))

            offspring = [{'code': code, 'fitness': None} for code in mutated_codes if code]

            if not offspring:
                print("No valid offspring were generated in this generation.")
                continue

            offspring = self._evaluate_population(offspring)
            combined_population = parents + offspring
            combined_population.sort(key=lambda x: x.get('fitness', float('inf')))
            
            if self.strategy == '(mu,lambda)':
                offspring.sort(key=lambda x: x.get('fitness', float('inf')))
                parents = offspring[:self.mu]
            elif self.strategy == '(mu+lambda)':
                parents = combined_population[:self.mu]
            
            self._archive(parents)
            best_fitness_so_far = self.best_agents_history[-1]['fitness']
            print(f"End of Generation {generation}. Best fitness so far: {best_fitness_so_far:.4f}")
            
        return self.best_agents_history

# Example Usage
if __name__ == '__main__':
    MODEL_TO_USE = "lbl/cborg-coder:latest"

    
    PROBLEM_PROMPT = f"""
Write a Python function named `{TARGET_FUNCTION_NAME}(x)` that accurately models the relationship in the following data points: {list(TARGET_DATA.items())}. The function should take a number `x` as input and return the corresponding `y` value.

Output only the raw code in a single ```python block.
"""
    
    evolver = LLMAgentEvolver(
        problem_description=PROBLEM_PROMPT,
        model_name=MODEL_TO_USE,
        n_queries=10,  # This is your LLM call budget
        mu=3,
        strategy='(mu,lambda)'
    )
    
    history = evolver.search()
    
    if history:
        print("\n\n--- Evolution Complete ---")
        best_solution = history[-1]
        print(f"Best fitness (MSE) found: {best_solution['fitness']:.4f}")
        print("Best code found:")
        print("```python")
        print(best_solution['code'])
        print("```")
    else:
        print("\n\n--- Evolution Complete ---")
        print("No valid solutions were found.")
