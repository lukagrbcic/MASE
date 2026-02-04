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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure your API key is set
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

class MutateEvolver:

    def __init__(self, problem_description, model_name,
                 n_queries,
                 k, # This is your Beam Width (Top k to keep)
                 branching_factor, # How many mutations per parent
                 evaluator,
                 mutate_context,
                 max_repair_attempts=1,
                 n_jobs=1,
                 temperature=0.75):

        self.problem_description = problem_description
        self.model_name = model_name
        self.n_queries = n_queries
        self.k = k
        self.branching_factor = branching_factor
        self.evaluator = evaluator
        self.mutate_context = mutate_context
        self.max_repair_attempts = max_repair_attempts
        self.n_jobs = n_jobs
        self.temperature = temperature

        self.best_agents_history = []
        self.convergence_history = []

        self.query_calls = 0
        self._reset()
        self.query_lock = threading.Lock()

    def _archive(self, population):
        """
        Updates the global best agent and records the convergence history.
        """
        valid_agents = [p for p in population if p.get('fitness', float('inf')) != float('inf')]
        if not valid_agents:
            if self.best_agents_history:
                self.best_agents_history.append(self.best_agents_history[-1])
            return

        best_in_batch = min(valid_agents, key=lambda x: x['fitness'])

        current_global_best = float('inf')
        if self.best_agents_history and self.best_agents_history[-1]['fitness'] != float('inf'):
            current_global_best = self.best_agents_history[-1]['fitness']

        if best_in_batch['fitness'] < current_global_best:
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

    def _reset(self):
        """Resets the internal state for a new run."""
        self.best_agents_history = []
        self.convergence_history = []
        self.query_calls = 0

    def get_dense_convergence(self):
        dense_fitness = np.full(self.n_queries + 1, float('inf'))
        sorted_history = sorted(self.convergence_history, key=lambda x: x['query'])
        current_best = float('inf')
        history_idx = 0
        for q in range(self.n_queries + 1):
            while history_idx < len(sorted_history) and sorted_history[history_idx]['query'] <= q:
                current_best = sorted_history[history_idx]['fitness']
                history_idx += 1
            dense_fitness[q] = current_best

        # Backfill
        first_valid_val = float('inf')
        first_valid_idx = -1
        for i, val in enumerate(dense_fitness):
            if val != float('inf'):
                first_valid_val = val
                first_valid_idx = i
                break
        if first_valid_idx != -1:
            dense_fitness[:first_valid_idx] = first_valid_val
        return dense_fitness

    def save_convergence_plot(self, filename="convergence_mutate.png"):
        dense_array = self.get_dense_convergence()
        queries = np.arange(len(dense_array))
        plt.figure(figsize=(10, 6))
        plt.plot(queries, dense_array*-1, drawstyle='steps-post', color='red', linewidth=2)
        final_val = dense_array[-1]
        plt.title(f"Mutate Search Convergence (Best: {final_val:.4f})")
        plt.xlabel("Query Count")
        plt.ylabel("Best Fitness")
        plt.grid(True, alpha=0.3)
        try:
            plt.savefig(filename)
            print(f"\nPlot successfully saved to {filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        finally:
            plt.close()

    def _evaluate_population(self, population):
        codes_to_evaluate = [agent['code'] for agent in population]
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(executor.map(self.evaluator, codes_to_evaluate))
        for agent, (fitness, error) in zip(population, results):
            agent['fitness'] = fitness
            agent['error'] = error
        return population

    def _repair_agent_worker(self, agent):
        if agent.get('fitness', float('inf')) != float('inf'):
            return agent

        print(f"--- Attempting to repair code with error: {agent.get('error')} ---")
        current_code, current_error = agent['code'], agent['error']

        for i in range(self.max_repair_attempts):
            if self.query_calls >= self.n_queries:
                break

            repair_prompt = f"""
            You are an expert Python programmer. Fix the bug in the code below.
            Code:
            ```python
            {current_code}
            ```
            Error: "{current_error}"

            Output only the fixed, raw Python code. Do not rename classes or functions.
            """
            fixed_code = self._llm_query(repair_prompt)
            if not fixed_code: continue

            temp_population = [{'code': fixed_code, 'fitness': None}]
            evaluated_pop = self._evaluate_population(temp_population)
            fitness = evaluated_pop[0]['fitness']
            error = evaluated_pop[0]['error']

            if fitness != float('inf'):
                print(f"--- Repair successful! ---")
                agent.update({'code': fixed_code, 'fitness': fitness, 'error': None})
                return agent
            else:
                current_code, current_error = fixed_code, error

        return agent

    def _llm_query(self, prompt):
        with self.query_lock:
            if self.query_calls >= self.n_queries: return ""
            self.query_calls += 1
            current_query_num = self.query_calls

        print(f"--- LLM Query #{current_query_num}/{self.n_queries} ---")

        try:
            llm_instance = LLM(query=prompt, model=self.model_name, temperature=self.temperature)
            response_text = llm_instance.get_response()
            if response_text is None: return ""
            return llm_instance.get_code(response_text)
        except Exception as e:
            return ""

    def _initialize_population(self):
        print(f"Initializing population (Target: {self.k} valid agents)...")
        population = []

        # We try to fill at least K spots, but we can process in batches of 'n_jobs'
        while len(population) < self.k and self.query_calls < self.n_queries:
            needed = self.k - len(population)
            batch_size = needed
            prompts = [self.problem_description] * batch_size

            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                codes = list(executor.map(self._llm_query, prompts))

            candidates = [{'code': c, 'fitness': None} for c in codes if c]
            candidates = self._evaluate_population(candidates)

            # Auto-repair failures during init
            failed = [c for c in candidates if c['fitness'] == float('inf')]
            passed = [c for c in candidates if c['fitness'] != float('inf')]

            if failed and self.max_repair_attempts > 0:
                    with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                        repaired = list(executor.map(self._repair_agent_worker, failed))
                    passed.extend([r for r in repaired if r['fitness'] != float('inf')])

            for p in passed:
                population.append(p)

            print(f"Valid agents so far: {len(population)}/{self.k}")

        if not population: return []

        # Return best K
        population.sort(key=lambda x: x['fitness'])
        return population[:self.k]

    def _mutate_worker(self, parent_agent):
        """
        Takes a single parent and tries to improve it based on context.
        """
        parent_code = parent_agent['code']
        parent_fit = parent_agent['fitness']

        prompt = f"""
        You are an optimization expert.
        Objective: {self.mutate_context}

        The following code is a good solution (Fitness: {parent_fit:.4f}).
        Your task is to MUTATE it to create a slightly different, potentially better version.

        Strategies:
        1. Optimize hotspots (loops, expensive calls).
        2. Tweak heuristics or constants.
        3. Handle edge cases better.

        Current Code:
        ```python
        {parent_code}
        ```

        Output only the raw, complete, improved Python code.
        DO NOT rename functions or classes.
        """
        return self._llm_query(prompt)

    def search(self):
        parents = self._initialize_population()
        if not parents:
            print("Initialization failed.")
            return [], []

        self._archive(parents)
        generation = 0

        while self.query_calls < self.n_queries:
            generation += 1
            print(f"\n--- Generation {generation} | Queries: {self.query_calls}/{self.n_queries} ---")

            # 1. Expand: Each of the K parents produces 'branching_factor' children
            tasks = []
            for p in parents:
                for _ in range(self.branching_factor):
                    tasks.append(p)

            # Check if we have enough budget
            if self.query_calls + len(tasks) > self.n_queries:
                tasks = tasks[:self.n_queries - self.query_calls]

            if not tasks: break

            print(f"Expanding top {len(parents)} parents into {len(tasks)} mutations...")

            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                mutated_codes = list(executor.map(self._mutate_worker, tasks))

            # 2. Evaluate
            offspring = [{'code': code, 'fitness': None, 'error': None} for code in mutated_codes if code]
            offspring = self._evaluate_population(offspring)

            # 3. Repair
            failed_offspring = [o for o in offspring if o['fitness'] == float('inf')]
            if failed_offspring and self.max_repair_attempts > 0:
                print(f"--- Repairing {len(failed_offspring)} failed agents ---")
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    repaired_offspring = list(executor.map(self._repair_agent_worker, failed_offspring))
                # Re-integrate repaired agents
                valid_offspring = [o for o in offspring if o['fitness'] != float('inf')]
                valid_offspring.extend([r for r in repaired_offspring if r['fitness'] != float('inf')])
            else:
                valid_offspring = [o for o in offspring if o['fitness'] != float('inf')]

            # 4. Selection (Top K from Parents + Offspring)
            combined_pool = parents + valid_offspring


            combined_pool.sort(key=lambda x: x['fitness'])

            # Keep top K
            parents = combined_pool[:self.k]

            self._archive(parents)

            if self.best_agents_history:
                print(f"End of Generation {generation}. Global Best: {self.best_agents_history[-1]['fitness']:.4f}")

        return self.best_agents_history, self.convergence_history

    def run_batch(self, n_trials, plot_filename="batch_convergence.png"):
            """
            Runs the search 'n_trials' times.
            Aggregates the dense convergence arrays.
            Plots Mean +/- Std Dev.
            """
            all_dense_arrays = []

            print(f"Starting Batch Experiment: {n_trials} trials.")

            for i in range(n_trials):
                print(f"\n================ TRIAL {i+1}/{n_trials} ================")
                _, _ = self.search()

                # Get the dense array for this run
                dense = self.get_dense_convergence()

                # Handle cases where initialization failed completely (all infs)
                # We replace infs with NaN for calculation stats, or a penalty value
                if np.isinf(dense).all():
                    print(f"Trial {i+1} failed to find any valid solution.")
                else:
                    all_dense_arrays.append(dense)

                self._reset()
            if not all_dense_arrays:
                print("All trials failed.")
                return
            # Stack arrays: Shape (n_valid_trials, n_queries + 1)
            data_matrix = np.vstack(all_dense_arrays)

            # Calculate Mean and Std, ignoring NaNs/Infs if possible
            # We mask Infs for the plot
            masked_data = np.ma.masked_invalid(data_matrix)
            mean_curve = np.mean(masked_data, axis=0)
            std_curve = np.std(masked_data, axis=0)

            # --- Plotting ---
            queries = np.arange(len(mean_curve))

            plt.figure(figsize=(10, 6))

            # We assume minimization, so we might plot -1*fitness for visual "ascent"
            # OR just raw fitness if you prefer.
            # Following previous logic of multiplying by -1 for visualization:
            y_mean = mean_curve * -1
            y_std = std_curve # Scale doesn't change direction

            plt.plot(queries, y_mean, color='blue', label='Mean Best Fitness', linewidth=2)
            plt.fill_between(queries, y_mean - y_std, y_mean + y_std, color='blue', alpha=0.2, label='Std Dev')

            plt.title(f"Batch Convergence ({n_trials} Trials, K={self.k})")
            plt.xlabel("Query Count")
            plt.ylabel("Fitness (Inverted)")
            plt.legend()
            plt.grid(True, alpha=0.3)

            try:
                plt.savefig(plot_filename)
                print(f"\nBatch plot successfully saved to {plot_filename}")
            except Exception as e:
                print(f"Error saving plot: {e}")
            finally:
                plt.close()


