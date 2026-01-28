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
from evaluator import evaluate_code

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
        print(f"Initializing population of size {self.mu}...")
        prompts = [self.problem_description] * self.mu
        with ThreadPoolExecutor(max_workers=self.n_jobs_query) as executor:
            initial_codes = list(executor.map(self._llm_query, prompts))

        population = [{'code': code, 'fitness': None} for code in initial_codes if code]

        return self._evaluate_population(population)
        
    def _recombination_worker(self, parent_pair):
        """Worker function to perform recombination for a pair of parents."""
        p1, p2 = parent_pair
        recombine_prompt = f"Here are two Python solutions for a problem. Combine their best ideas to create a superior one! {self.mutate_recombine_context}. Do not rename anything, keep the format as is,  output only the raw code.\n\nSolution A (fitness: {p1['fitness']:.4f}):\n```python\n{p1['code']}\n```\n\nSolution B (fitness: {p2['fitness']:.4f}):\n```python\n{p2['code']}\n```"
        return self._llm_query(recombine_prompt)

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
        

    def _mutation_worker(self, recombined_code):
        if not recombined_code: return ""

        # Check if we should use Global Best as inspiration
        use_inspiration = False
        global_best_code = ""

        # We access best_agents_history safely (reading is generally thread-safe in Python lists)
        if self.best_agents_history and self.best_agents_history[-1]['fitness'] != float('inf'):
            if random.random() < self.inspiration_prob:
                use_inspiration = True
                global_best_code = self.best_agents_history[-1]['code']

        if use_inspiration and global_best_code:
            mutate_prompt = f"""
Critically analyze and improve this candidate Python code.
{self.mutate_recombine_context}.

REFERENCE: Here is the current Best Known Solution (Global Best).
You may borrow logic or style from it, but try to improve upon the candidate code specifically.
Global Best:
```python
{global_best_code}
```

Candidate Code to Improve:
```python
{recombined_code}
```
Output only the raw, improved code.
"""
        else:
            mutate_prompt = f"Critically analyze and improve this Python code. {self.mutate_recombine_context}. Output only the raw, improved code.\n\nCode:\n```python\n{recombined_code}\n```"

        return self._llm_query(mutate_prompt)

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
            print ('offsprings to generate', offspring_to_generate)
            if offspring_to_generate <= 0: break

            # Parallel Recombination
            parent_pairs = [random.sample(parents, 2) for _ in range(offspring_to_generate)]
            with ThreadPoolExecutor(max_workers=MAX_LLM_WORKERS) as executor:
                recombined_codes = list(executor.map(self._recombination_worker, parent_pairs))

            # Parallel Mutation (Includes Strategy 2: Inspiration)
            with ThreadPoolExecutor(max_workers=MAX_LLM_WORKERS) as executor:
                mutated_codes = list(executor.map(self._mutation_worker, recombined_codes))

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

            if self.tournament_selection_k == 0:

                ### Selection
                combined_population = [p for p in (parents + repaired_offspring) if p['fitness'] != float('inf')]
                if not combined_population: continue

                combined_population.sort(key=lambda x: x.get('fitness', float('inf')))

                if self.strategy == '(mu,lambda)':
                    valid_offspring = [o for o in repaired_offspring if o['fitness'] != float('inf')]
                    if valid_offspring:
                        valid_offspring.sort(key=lambda x: x.get('fitness', float('inf')))
                        parents = valid_offspring[:self.mu]
                    else:
                        parents = combined_population[:self.mu] # Fallback
                elif self.strategy == '(mu+lambda)':
                    parents = combined_population[:self.mu]

                self._archive(parents)
                best_fitness_so_far = self.best_agents_history[-1]['fitness']
                print(f"End of Generation {generation}. Best fitness so far: {best_fitness_so_far:.4f}")


            else:
                valid_repaired_offspring = [o for o in repaired_offspring if o['fitness'] != float('inf')]
                valid_parents = [p for p in parents if p['fitness'] != float('inf')]

                # Pool to select from
                selection_pool = []

                if self.strategy == '(mu,lambda)':
                    # In (mu, lambda), we prefer selecting only from offspring.
                    # If we have enough valid offspring, use them.
                    if len(valid_repaired_offspring) >= self.mu:
                        selection_pool = valid_repaired_offspring
                    else:
                        # Fallback: if not enough valid offspring, mix in parents to survive
                        selection_pool = valid_parents + valid_repaired_offspring
                elif self.strategy == '(mu+lambda)':
                    # In (mu+lambda), we always select from the mix
                    selection_pool = valid_parents + valid_repaired_offspring

                if not selection_pool:
                    print("No valid agents found this generation. Carrying over previous parents.")
                    continue

                # Perform Tournament Selection
                parents = self._tournament_selection(selection_pool, k=self.tournament_selection_k)

                # Update Archive (Global Best)
                self._archive(parents)

                best_fitness_so_far = self.best_agents_history[-1]['fitness']
                print(f"End of Generation {generation}. Best fitness so far: {best_fitness_so_far:.4f}")


        return self.best_agents_history

