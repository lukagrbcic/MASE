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
import json 
from evaluator import evaluate_code
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    
    def __init__(self, problem_description, model_name,
                 n_queries, mu, evaluator,
                 mutate_recombine_context,
                 strategy='(mu,lambda)',
                 max_repair_attempts=1,
                 n_jobs_eval=1,
                 n_jobs_query=1,
                 memetic_period=2,
                 inspiration_prob=0.5,
                 tournament_selection_k=0,
                 temperature=0.75):

        self.problem_description = problem_description
        self.model_name = model_name
        self.n_queries = n_queries
        self.mu = mu
        self.evaluator = evaluator
        self.mutate_recombine_context = mutate_recombine_context
        self.strategy = strategy
        self.max_repair_attempts = max_repair_attempts 
        self.n_jobs_eval = n_jobs_eval
        self.n_jobs_query = n_jobs_query
        self.memetic_period = memetic_period
        self.inspiration_prob = inspiration_prob
        self.tournament_selection_k = tournament_selection_k
        self.temperature = temperature

        if self.tournament_selection_k > 0:
            print ('Using tournament selection!')

        self.lambda_ = int(2*mu)
        self.best_agents_history = []
        self.convergence_history = [] 

        self.query_calls = 0
        self.query_lock = threading.Lock()
        

    def _archive(self, population):
        """
        Updates the global best agent and records the convergence history.
        """
        valid_agents = [p for p in population if p.get('fitness', float('inf')) != float('inf')]
        if not valid_agents:
            # Everyone failed, carry forward old best
            if self.best_agents_history:
                self.best_agents_history.append(self.best_agents_history[-1])
            return

        # Best in current batch
        best_in_batch = min(valid_agents, key=lambda x: x['fitness'])
        
        # Current Global Best
        current_global_best = float('inf')
        if self.best_agents_history and self.best_agents_history[-1]['fitness'] != float('inf'):
            current_global_best = self.best_agents_history[-1]['fitness']

        # LOGIC: Did we beat the record?
        if best_in_batch['fitness'] < current_global_best:
            # Yes -> Record breakthrough
            self.best_agents_history.append(best_in_batch)
            
            entry = {
                'query': self.query_calls,
                'fitness': best_in_batch['fitness'],
                'code': best_in_batch['code']
            }
            self.convergence_history.append(entry)
            print(f"*** NEW RECORD: Fitness {best_in_batch['fitness']:.4f} (Query {self.query_calls}) ***")
            
        else:
            self.best_agents_history.append(self.best_agents_history[-1])

    # --- REPLACE THIS METHOD ---
    def get_dense_convergence(self):
        """
        Returns a numpy array of size (n_queries + 1).
        It maps the best fitness found in a batch to all queries involved in that batch.
        """
        # 1. Create the array filled with Infs
        dense_fitness = np.full(self.n_queries + 1, float('inf'))
        
        # 2. Sort the sparse history (e.g., Query 10 -> -2.6, Query 20 -> -2.8)
        sorted_history = sorted(self.convergence_history, key=lambda x: x['query'])
        
        # 3. Fill the array forward
        current_best = float('inf')
        history_idx = 0
        
        for q in range(self.n_queries + 1):
            # If we passed a checkpoint (e.g. q=10), update the current best
            while history_idx < len(sorted_history) and sorted_history[history_idx]['query'] <= q:
                current_best = sorted_history[history_idx]['fitness']
                history_idx += 1
            dense_fitness[q] = current_best
            
        # 4. CRITICAL FIX: Backfill the "waiting period"
        # If the first batch finished at Query 10 with score -2.6, 
        # we assume queries 0-9 also represent this -2.6 capability.
        first_valid_val = float('inf')
        first_valid_idx = -1
        
        # Find first number that isn't inf
        for i, val in enumerate(dense_fitness):
            if val != float('inf'):
                first_valid_val = val
                first_valid_idx = i
                break
        
        # Fill the start of the array with that number
        if first_valid_idx != -1:
            dense_fitness[:first_valid_idx] = first_valid_val
            
        return dense_fitness

    # --- REPLACE THIS METHOD (Headless Plotting) ---
    def save_convergence_plot(self, filename="convergence.png"):
        """
        Generates the dense array, saves the plot, and prints the array.
        Uses 'Agg' backend to prevent crashes on servers.
        """
        dense_array = self.get_dense_convergence()
        
        print("\n=== DENSE CONVERGENCE ARRAY ===")
        print(dense_array)
        
        # Create plot without a window (Headless)
        queries = np.arange(len(dense_array))
        
        plt.figure(figsize=(10, 6))
        
        # Plot the data
        plt.plot(queries, dense_array, drawstyle='steps-post', color='blue', linewidth=2)
        
        # Labeling
        final_val = dense_array[-1]
        plt.title(f"Convergence Profile (Best: {final_val:.4f})")
        plt.xlabel("Query Count")
        plt.ylabel("Best Fitness")
        plt.grid(True, alpha=0.3)
        
        # Save and Close
        try:
            plt.savefig(filename)
            print(f"\nPlot successfully saved to {filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        finally:
            plt.close() # Clean up memory

    def _evaluate_population(self, population):
        codes_to_evaluate = [agent['code'] for agent in population]
        with ThreadPoolExecutor(max_workers=self.n_jobs_eval) as executor:
            results = list(executor.map(self.evaluator, codes_to_evaluate))
        for agent, (fitness, error) in zip(population, results):
            agent['fitness'] = fitness
            agent['error'] = error
        return population

    # --- STRATEGY 1: Memetic (Local Search) Worker ---
    def _memetic_worker(self, agent_code):
        memetic_prompt = f"""
        You are an optimization expert. You have the current Global Best solution.
        Your goal is to perform a local search: refactor, optimize, or handle edge cases better WITHOUT breaking existing functionality.
        {self.mutate_recombine_context}
        Make it perfect.

        Current Best Code:
        ```python
        {agent_code}
        ```
        Output only the raw, optimized code.
        """
        return self._llm_query(memetic_prompt)

    def _llm_diversity_agent(self, population):
            """
            Uses the LLM to filter out semantically identical code.
            Input: Population sorted by fitness.
            Output: A filtered list of agents.
            """
            # 1. Prepare the prompt with indexed codes
            # We limit the code length slightly to avoid massive context usage if they are huge
            code_blocks = ""
            for i, agent in enumerate(population):
                code_snippet = agent['code']
                code_blocks += f"--- SOLUTION {i} ---\n{code_snippet}\n\n"
    
            prompt = f"""
            You are a Senior Code Reviewer. Your task is to filter out semantically duplicate solutions to maintain diversity.
            
            I will provide a list of Python solutions, sorted by performance (Solution 0 is best, Solution 1 is second best, etc.).
            
            Task:
            1. Analyze the logic of the solutions sequentially.
            2. Identify solutions that are fundamentally DIFFERENT in approach/logic from the ones accepted so far.
            3. If a solution is just a variable rename or minor reformatting of a previous (better) solution, IGNORE it.
            4. Solution 0 is always kept.
            
            Output format:
            Return ONLY a raw JSON list of integers representing the indices of the solutions to KEEP.
            Example: [0, 2, 5]
            
            Solutions to evaluate:
            {code_blocks}
            """
    
            response = self._llm_query(prompt)
            
            # 2. Parse the indices
            try:
                # Try to find a JSON list in the response
                match = re.search(r"\[.*?\]", response, re.DOTALL)
                if match:
                    indices_to_keep = json.loads(match.group(0))
                else:
                    # Fallback: if model just typed numbers "0, 2, 5"
                    indices_to_keep = [int(s) for s in re.findall(r"\d+", response)]
                
                # Sanity check: Ensure indices are valid integers
                indices_to_keep = [i for i in indices_to_keep if isinstance(i, int) and 0 <= i < len(population)]
                
                # If empty (model hallucinated), always keep the best one (index 0)
                if not indices_to_keep:
                    indices_to_keep = [0]
                    
                print(f"Diversity Agent kept indices: {indices_to_keep}")
                
                # 3. Construct the new population
                filtered_pop = [population[i] for i in indices_to_keep]
                return filtered_pop
    
            except Exception as e:
                print(f"Diversity Agent failed to parse: {e}. Keeping top {self.mu}.")
                # Fallback: Just return the top k original ones to be safe
                return population[:self.mu]

    
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
    Please DON'T FORGET TO ADD imports to the start of the code! Do not rename anything, keep the format as is, just focus on the error!
    """
            fixed_code = self._llm_query(repair_prompt)
            if not fixed_code:
                continue

            temp_population = [{'code': fixed_code, 'fitness': None}]
            evaluated_pop = self._evaluate_population(temp_population)
            fitness = evaluated_pop[0]['fitness']
            error = evaluated_pop[0]['error']
            
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
            llm_instance = LLM(query=prompt, model=self.model_name, temperature=self.temperature)
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
            print(f"Initializing population (Target: {self.mu} valid agents)...")
            population = []
            
            while len(population) < self.mu and self.query_calls < self.n_queries:
                
                needed = self.mu - len(population)
                batch_size = max(needed, self.n_jobs_query) 
                
                prompts = [self.problem_description] * batch_size
                
                print(f"--- Init Batch: Requesting {batch_size} solutions... ---")
                
                with ThreadPoolExecutor(max_workers=self.n_jobs_query) as executor:
                    codes = list(executor.map(self._llm_query, prompts))
    
                candidates = [{'code': c, 'fitness': None} for c in codes if c]
                candidates = self._evaluate_population(candidates)
                
                failed = [c for c in candidates if c['fitness'] == float('inf')]
                passed = [c for c in candidates if c['fitness'] != float('inf')]
                
                if failed and self.max_repair_attempts > 0:
                     print(f"Init: Attempting to repair {len(failed)} failed candidates...")
                     with ThreadPoolExecutor(max_workers=self.n_jobs_eval) as executor:
                        repaired = list(executor.map(self._repair_agent_worker, failed))
                     passed.extend([r for r in repaired if r['fitness'] != float('inf')])
    
                existing_codes = set(p['code'] for p in population)
                for p in passed:
                    if p['code'] not in existing_codes:
                        population.append(p)
                        existing_codes.add(p['code'])
                
                print(f"Valid agents so far: {len(population)}/{self.mu}")
    
            if not population:
                return []
                
            return population[:self.mu]
        
    def _tournament_selection(self, population, k=3):
        """
        Selects self.mu parents using tournament selection.
        k: Tournament size (larger k = higher selection pressure, less diversity)
        """
        selected_parents = []
        # We need to fill the parent slots
        while len(selected_parents) < self.mu:
            # 1. Pick k random individuals from the population
            # (Use min in case population is smaller than k)
            tournament_candidates = random.sample(population, min(k, len(population)))
            
            # 2. The winner is the one with the lowest fitness
            winner = min(tournament_candidates, key=lambda x: x.get('fitness', float('inf')))
            
            # 3. Add winner to new parents (Note: standard tournament allows duplicates)
            selected_parents.append(winner)
            
        return selected_parents
        

    def _crossover_and_mutate_worker(self, parent_pair):
        """
        Combines Recombination and Mutation into a single LLM call.
        """
        p1, p2 = parent_pair

        # Logic to check if we should use Global Best as inspiration (from your original mutation worker)
        use_inspiration = False
        global_best_code = ""
        if self.best_agents_history and self.best_agents_history[-1]['fitness'] != float('inf'):
            if random.random() < self.inspiration_prob:
                use_inspiration = True
                global_best_code = self.best_agents_history[-1]['code']

        # Construct the Prompt
        # Base instruction: Combine parents
        prompt = f"""
        You are an expert Python developer.
        Task 1 (Recombination): Combine the best ideas from the two Parent solutions below to create a superior merged solution.
        Task 2 (Mutation): {self.mutate_recombine_context} - Critically analyze the merged result and improve it.

        Parent A (fitness: {p1['fitness']:.4f}):
        ```python
        {p1['code']}
        ```

        Parent B (fitness: {p2['fitness']:.4f}):
        ```python
        {p2['code']}
        ```
        """

        # Add Global Best context if applicable
        if use_inspiration and global_best_code:
            prompt += f"""
        REFERENCE: Here is the current Best Known Solution (Global Best).
        You may borrow logic or style from it to improve the offspring.
        Global Best:
        ```python
        {global_best_code}
        ```
        """

        prompt += "\nOutput only the raw, complete, combined, and improved Python code."

        return self._llm_query(prompt)


    def search(self):
        MAX_LLM_WORKERS = self.n_jobs_query
        parents = self._initialize_population()
        if not parents:
            print("Initialization failed.")
            return []

        self._archive(parents)

        generation = 0
        while self.query_calls < self.n_queries:
            generation += 1
            print(f"\n--- Generation {generation} | Queries: {self.query_calls}/{self.n_queries} ---")

            # --- STRATEGY 1: Memetic Step (Run periodically) ---
            if (self.memetic_period > 0 and
                generation % self.memetic_period == 0 and
                self.best_agents_history and
                self.best_agents_history[-1]['fitness'] != float('inf')):

                print(f"--- Running Memetic Local Search on Global Best ---")
                best_code = self.best_agents_history[-1]['code']
                optimized_code = self._memetic_worker(best_code)

                if optimized_code:
                    temp_pop = [{'code': optimized_code, 'fitness': None}]
                    eval_pop = self._evaluate_population(temp_pop)
                    fit = eval_pop[0]['fitness']
                    if fit < self.best_agents_history[-1]['fitness']:
                        print(f"Memetic Search Success! Improved fitness to {fit:.4f}")
                        # Immediately update archive
                        self._archive([{'code': optimized_code, 'fitness': fit, 'error': None}])
                    else:
                        print("Memetic Search: No improvement found.")

            # Check budget after memetic step
            if self.query_calls >= self.n_queries: break

            # Standard Evolution Flow
            queries_left = self.n_queries - self.query_calls
            offspring_to_generate = min(self.lambda_, queries_left // 2)
            if offspring_to_generate <= 0: break

            ## Parallel Recombination
            parent_pairs = [random.sample(parents, 2) for _ in range(offspring_to_generate)]

            with ThreadPoolExecutor(max_workers=MAX_LLM_WORKERS) as executor:
                mutated_codes = list(executor.map(self._crossover_and_mutate_worker, parent_pairs))

            # Evaluation
            offspring = [{'code': code, 'fitness': None, 'error': None} for code in mutated_codes if code]
            if not offspring: continue
            offspring = self._evaluate_population(offspring)

            # Repair
            failed_offspring = [o for o in offspring if o['fitness'] == float('inf')]
            if failed_offspring:
                print(f"--- Repairing {len(failed_offspring)} failed agents ---")
                with ThreadPoolExecutor(max_workers=MAX_LLM_WORKERS) as executor:
                    repaired_offspring = list(executor.map(self._repair_agent_worker, offspring))
            else:
                repaired_offspring = offspring
                
            valid_repaired_offspring = [o for o in repaired_offspring if o['fitness'] != float('inf')]
            valid_parents = [p for p in parents if p['fitness'] != float('inf')]
            
            if self.tournament_selection_k == 0:
                if self.strategy == '(mu,lambda)':
                    
                    # EXTINCTION CHECK: If all offspring failed, keep parents
                    if not valid_repaired_offspring:
                        print("!!! WARNING: All offspring failed. Keeping previous generation. !!!")
                        # We do NOT update 'parents', so the loop continues with the old ones.
                    else:
                        valid_repaired_offspring.sort(key=lambda x: x.get('fitness', float('inf')))
                        
                        # --- DIVERSITY AGENT START ---
                        # Only run filter if we have a surplus of candidates to choose from
                        if len(valid_repaired_offspring) > self.mu:
                            print("Running Diversity Filter on offspring...")
                            valid_repaired_offspring = self._llm_diversity_agent(valid_repaired_offspring)
                        # --- DIVERSITY AGENT END ---

                        # If we have enough offspring to replace parents entirely
                        if len(valid_repaired_offspring) >= self.mu:
                            parents = valid_repaired_offspring[:self.mu]
                        else:
                            # Partial replacement: Fill the rest with best parents
                            # Note: We prefer filling with parents rather than the duplicates we just filtered out
                            print(f"Partial success: Keeping {len(valid_repaired_offspring)} diverse offspring and filling rest with parents.")
                            n_needed = self.mu - len(valid_repaired_offspring)
                            parents = valid_repaired_offspring + valid_parents[:n_needed]
            
                elif self.strategy == '(mu+lambda)':
                    # Combine everyone
                    combined = valid_parents + valid_repaired_offspring
                    combined.sort(key=lambda x: x.get('fitness', float('inf')))
                    
                    # --- DIVERSITY AGENT START ---
                    if len(combined) > self.mu:
                        print("Running Diversity Filter on combined pool...")
                        combined = self._llm_diversity_agent(combined)
                    # --- DIVERSITY AGENT END ---
                    
                    parents = combined[:self.mu]
            
                self._archive(parents)
                best_fitness_so_far = self.best_agents_history[-1]['fitness']
                print(f"End of Generation {generation}. Best fitness so far: {best_fitness_so_far:.4f}")
            
            else:
                selection_pool = []
            
                if self.strategy == '(mu,lambda)':
                    if len(valid_repaired_offspring) >= self.mu:
                        selection_pool = valid_repaired_offspring
                    elif valid_repaired_offspring:
                         selection_pool = valid_parents + valid_repaired_offspring
                    else:
                        # EXTINCTION CHECK
                        print("!!! WARNING: All offspring failed. Keeping previous generation. !!!")
                        selection_pool = valid_parents # Force fallback to parents
            
                elif self.strategy == '(mu+lambda)':
                    selection_pool = valid_parents + valid_repaired_offspring
            
                if not selection_pool:
                    print("CRITICAL: No valid agents found anywhere. Stopping search.")
                    break
                
                # --- DIVERSITY AGENT START ---
                # For tournament, we want to filter the pool BEFORE selection
                # This prevents the tournament from picking 3 copies of the same code
                selection_pool.sort(key=lambda x: x.get('fitness', float('inf'))) # Sort helps the agent
                if len(selection_pool) > self.mu:
                     print("Running Diversity Filter on selection pool...")
                     selection_pool = self._llm_diversity_agent(selection_pool)
                # --- DIVERSITY AGENT END ---
   
                parents = self._tournament_selection(selection_pool, k=self.tournament_selection_k)
            
                self._archive(parents)
                best_fitness_so_far = self.best_agents_history[-1]['fitness']
                print(f"End of Generation {generation}. Best fitness so far: {best_fitness_so_far:.4f}")
                

        return self.best_agents_history, self.convergence_history


