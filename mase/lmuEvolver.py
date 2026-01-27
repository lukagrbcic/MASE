# -*- coding: utf-8 -*-
import numpy as np
import random
import time
from concurrent.futures import ThreadPoolExecutor
import openai
import os
import re as re
import sys
import threading
from evaluatorServer import algorithm_analysis

client = openai.OpenAI(
    api_key = os.environ.get('CBORG_API_KEY'),
    base_url = "https://api.cborg.lbl.gov"
)

class LLM:
    def __init__(self, query, model, temperature=1.0):
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


class LLMAgentEvolver:
    
    def __init__(self, problem_description, model_name, n_queries, mu, evaluator, strategy='(mu,lambda)', max_repair_attempts=1):

        self.problem_description = problem_description
        self.model_name = model_name
        self.n_queries = n_queries
        self.mu = mu
        self.evaluator = evaluator
        self.strategy = strategy
        self.max_repair_attempts = max_repair_attempts 

        self.lambda_ = int(2*mu)
        self.best_agents_history = []
        
        self.query_calls = 0
        self.query_lock = threading.Lock()

    def _archive(self, population):
        current_best_agent = population[0]
    
        if current_best_agent['fitness'] == float('inf'):
            if not self.best_agents_history:
                self.best_agents_history.append(current_best_agent)
            return # Exit the function
    
        # If the history is empty, this is the first valid agent we've seen.
        if not self.best_agents_history or self.best_agents_history[-1]['fitness'] == float('inf'):
            self.best_agents_history.append(current_best_agent)
        # If the new agent is better than the best we've seen before, add it.
        elif current_best_agent['fitness'] < self.best_agents_history[-1]['fitness']:
            self.best_agents_history.append(current_best_agent)
        # Otherwise, carry the previous best forward.
        else:
            self.best_agents_history.append(self.best_agents_history[-1])


    def _evaluate_population(self, population):
        codes_to_evaluate = [agent['code'] for agent in population]
        with ThreadPoolExecutor(max_workers=1) as executor:
            # executor.map now returns tuples of (fitness, error)
            #results = list(executor.map(evaluation_server, codes_to_evaluate))
            results = list(executor.map(self.evaluator, codes_to_evaluate))

        for agent, (fitness, error) in zip(population, results):
            agent['fitness'] = fitness
            agent['error'] = error
        return population
    

    def _repair_agent_worker(self, agent):
        """Attempts to repair a single agent that failed evaluation."""
        if agent.get('fitness', float('inf')) != float('inf'):
            return agent # Agent is already valid, no repair needed
    
        print(f"--- Attempting to repair code with error: {agent.get('error')} ---")
        current_code, current_error = agent['code'], agent['error']

        for i in range(self.max_repair_attempts):
            if self.query_calls >= self.n_queries:
                print("Query budget exhausted, skipping repair.")
                break

            repair_prompt = f"""
    You are an expert Python programmer and debugger. The following code has a bug.
    Your task is to fix the code to resolve the error.
    
    The faulty code is:
    ```python
    {current_code}
    When executed, it produced this error:
    "{current_error}"
    
    Fix the bug that caused this error. Output only the raw, complete, corrected Python code.
    Please DON'T FORGET TO ADD imports to the start of the code!'
    """
            fixed_code = self._llm_query(repair_prompt)
            if not fixed_code:
                continue
            fitness, error = self.evaluator(fixed_code)
            
            
            if fitness != float('inf'):
                print(f"--- Repair successful after {i+1} attempt(s)! ---")
                agent.update({'code': fixed_code, 'fitness': fitness, 'error': None})
                return agent # Success: return the updated agent
            else:
                print(f"--- Repair attempt {i+1} failed. New error: {error} ---")
                # Update code and error for the *next* iteration of the loop
                current_code, current_error = fixed_code, error

        return agent
    
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
        prompts = [self.problem_description] * self.mu
        with ThreadPoolExecutor(max_workers=1) as executor:
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
        MAX_LLM_WORKERS = 1
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
            if offspring_to_generate <= 0: break
    
            # --- Parallel Generation ---
            parent_pairs = [random.sample(parents, 2) for _ in range(offspring_to_generate)]
            with ThreadPoolExecutor(max_workers=MAX_LLM_WORKERS) as executor:
                recombined_codes = list(executor.map(self._recombination_worker, parent_pairs))
            with ThreadPoolExecutor(max_workers=MAX_LLM_WORKERS) as executor:
                mutated_codes = list(executor.map(self._mutation_worker, recombined_codes))
    
            # --- Evaluation Step ---
            offspring = [{'code': code, 'fitness': None, 'error': None} for code in mutated_codes if code]
            if not offspring: continue
            offspring = self._evaluate_population(offspring)
    
            # --- NEW: Repair Step ---
            failed_offspring = [o for o in offspring if o['fitness'] == float('inf')]
            print(f"--- Starting repair phase for {len(failed_offspring)} failed agents ---")
            with ThreadPoolExecutor(max_workers=MAX_LLM_WORKERS) as executor:
                # The map will run the repair worker on all offspring.
                # It will instantly return the already-valid ones and work on the failed ones.
                repaired_offspring = list(executor.map(self._repair_agent_worker, offspring))
            
            # --- Selection Step ---
            combined_population = [p for p in (parents + repaired_offspring) if p['fitness'] != float('inf')]
            if not combined_population: continue
            
            combined_population.sort(key=lambda x: x.get('fitness', float('inf')))
            
            if self.strategy == '(mu,lambda)':
                valid_offspring = [o for o in repaired_offspring if o['fitness'] != float('inf')]
                if valid_offspring:
                    valid_offspring.sort(key=lambda x: x.get('fitness', float('inf')))
                    parents = valid_offspring[:self.mu]
            elif self.strategy == '(mu+lambda)':
                parents = combined_population[:self.mu]
            
            self._archive(parents)
            best_fitness_so_far = self.best_agents_history[-1]['fitness']
            print(f"End of Generation {generation}. Best fitness so far: {best_fitness_so_far:.4f}")
            
        return self.best_agents_history

# Example Usage
#if __name__ == '__main__':
MODEL_TO_USE = "lbl/cborg-coder:latest"
#MODEL_TO_USE = 'google/gemini-flash'
#MODEL_TO_USE = 'lbl/cborg-coder'

#MODEL_TO_USE = 'xai/grok-mini'



#MODEL_TO_USE='google/gemini-flash'
#MODEL_TO_USE ='anthropic/claude-haiku'
#MODEL_TO_USE ='openai/o4-mini'
#MODEL_TO_USE ='openai/o3-mini'
#MODEL_TO_USE ='openai/gpt-5-mini'
#MODEL_TO_USE ='lbl/cborg-coder:latest'
#MODEL_TO_USE ='lbl/cborg-deepthought:latest'
#MODEL_TO_USE = 'xai/grok-mini'



PROBLEM_PROMPT = """This a code I use in my sampling loop with Random Forests. It is a random sampling approach at the moment.

            import numpy as np
            import random
            from sklearn.ensemble import RandomForestRegressor

            np.random.seed(random.randint(0, 10223))


            class modelSampler:

                def __init__(self,  X, y, sample_size, lb, ub):

                    self.X = X
                    self.y = y
                    self.sample_size = sample_size
                    self.lb = lb
                    self.ub = ub

                    self.model = RandomForestRegressor().fit(self.X, self.y)

                def gen_samples(self):

                    n = self.sample_size
                    d = len(self.lb)
                    sample_set = np.array([np.random.uniform(self.lb, self.ub, d) for i in range(n)])

                    return sample_set, self.model


            # =============================================================================
            # Active Learning Sampler Implementation
            # =============================================================================
            #
            # WORKFLOW CONTEXT:
            # -----------------
            # This `modelSampler` class is part of a larger ACTIVE LEARNING LOOP.
            # The loop operates iteratively as follows:
            #   1. Call `gen_samples()` to select a new set of samples within bounds `lb` and `ub`.
            #   2. Add these samples to the datasets `self.X` (features) and `self.y` (labels).
            #   3. Retrain the model using the updated `self.X` and `self.y`.
            #
            # This process repeats many times — meaning the sampling algorithm must aim to:
            #   - Improve the model not just in the current iteration, but OVER MULTIPLE FUTURE ITERATIONS.
            #   - Select batches that maximize cumulative accuracy across the loop.
            #
            # The acquisition strategy must be effective in HIGH-DIMENSIONAL problems
            # (e.g., expanding 3D input to 822D features).
            #
            # -----------------------------------------------------------------------------
            #
            # CURRENT BEHAVIOR:
            # -----------------
            # 1. `gen_samples()` returns a matrix `sample_set` with:
            #       - Exactly `sample_size` rows
            #       - Number of columns == length of `lb` and `ub`
            # 2. `self.X` and `self.y` are datasets gathered so far in the active learning loop.
            # 3. A model is initially trained on `self.X` and `self.y`.
            # 4. The model is returned later for accuracy assessment.
            #
            # -----------------------------------------------------------------------------
            #
            # CONSTRAINTS:
            # ------------
            # - Keep the class name `modelSampler` and all method names unchanged (external code depends on them).
            # - Preserve the exact return format and interface of `gen_samples()`.
            # - `sample_size` is fixed and should not be changed. The returned `sample_set` must have `sample_size` number of rows.
            # - Maintain the meaning of `lb` and `ub` as feature bounds.
            # - Do NOT implement multiple strategies — only ONE algorithm should exist and be active at runtime.
            # - Input/output behavior must remain exactly the same (shape, types).
            # - Preserve full compatibility with the existing active learning workflow.
            # - Do not use standalone random or any kind of sampling.
            #
            # -----------------------------------------------------------------------------
            #
            # GOALS:
            # ------
            # - Implement a single, innovative BATCHED ACTIVE LEARNING acquisition strategy.
            #     ("Batched" = selecting multiple samples at once per iteration.)
            # - Can adapt known acquisition functions (e.g., uncertainty sampling, diversity sampling)
            #   or combine them creatively in a hybrid approach.
            # - Minimize the number of samples required to reach high accuracy.
            # - Model must have `.fit` and `.predict` methods (scikit-learn style) AND provide robust uncertainty estimates.
            # - Code should be clear, maintainable, and computationally efficient for high-dimensional datasets.
            # - The algorithm must consistently contribute to iterative improvements in the active learning loop.
            #
            # -----------------------------------------------------------------------------
            #
            # DELIVERABLES:
            # -------------
            # 1. Full Python code for the improved `modelSampler` (output as one complete code block).
            # 2. Inline comments describing WHY each change improves performance or accuracy.
            # 3. The algorithm implemented in `gen_samples()` must be directly used — no placeholders or optional code paths.
            # 4. Code must be ready to run in the described active learning loop without interface changes.
            # 5. Always respond ONLY with Python code inside a fenced code block like this:
            #        ```python
            #        # code here
            #        ```
            #
            # =============================================================================

            """

evolver = LLMAgentEvolver(
    problem_description=PROBLEM_PROMPT,
    model_name=MODEL_TO_USE,
    n_queries=200,
    mu=2,
    evaluator=algorithm_analysis,
    strategy='(mu,lambda)'
)

history = evolver.search()

if history:
    print("\n\n--- Evolution Complete ---")
    best_solution = history[-1]
    print(f"Best fitness found: {best_solution['fitness']:.4f}")
    print("Best code found:")
    print("```python")
    print(best_solution['code'])
    print("```")
else:
    print("\n\n--- Evolution Complete ---")
    print("No valid solutions were found.")

