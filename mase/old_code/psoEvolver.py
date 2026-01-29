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
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from evaluator import evaluate_code

# ===================================================================
# START: Your provided client code and LLM class (Unchanged)
# ===================================================================

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

    def __init__(self, problem_description, model_name, n_queries, population_size, evaluator, max_repair_attempts=2):
        self.problem_description = problem_description
        self.model_name = model_name
        self.n_queries = n_queries
        self.mu = population_size
        self.evaluator = evaluator
        self.max_repair_attempts = max_repair_attempts
        self.population = []
        self.g_best = {'code': None, 'fitness': float('inf')}
        self.query_calls = 0
        self.query_lock = threading.Lock()

    def _evaluate_population(self, population):
        codes_to_evaluate = [agent['code'] for agent in population]
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(self.evaluator, codes_to_evaluate))
        for agent, (fitness, error) in zip(population, results):
            agent['fitness'] = fitness
            agent['error'] = error
        return population

    def _repair_agent_worker(self, agent):
        """Attempts to repair a single agent that failed evaluation."""
        if agent.get('fitness', float('inf')) != float('inf'):
            return agent # Agent is already valid, no repair needed

        #print(f"--- Attempting to repair code with error: {agent.get('error')} ---")
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
'
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
                #print(f"--- Repair attempt {i+1} failed. New error: {error} ---")
                # Update code and error for the *next* iteration of the loop
                current_code, current_error = fixed_code, error

        return agent

    # --- MODIFIED: Added expect_code flag ---
    def _llm_query(self, prompt, expect_code=True):
        with self.query_lock:
            if self.query_calls >= self.n_queries: return ""
            self.query_calls += 1
            current_query_num = self.query_calls
        print(f"--- LLM Query #{current_query_num}/{self.n_queries} ---")
        try:
            llm_instance = LLM(query=prompt, model=self.model_name, temperature=1)
            response_text = llm_instance.get_response()
            if response_text is None: return ""
            if not expect_code:
                return response_text.strip()
            return llm_instance.get_code(response_text)
        except IndexError:
            return response_text.strip() if response_text else ""
        except Exception as e:
            #print(f"An unexpected error occurred in _llm_query: {e}")
            return ""

    def _initialize_population(self):
            print(f"Initializing population of size {self.mu}...")
            prompts = [self.problem_description] * self.mu

            # 1. Generate Initial Code
            with ThreadPoolExecutor(max_workers=2) as executor:
                initial_codes = list(executor.map(self._llm_query, prompts))

            temp_agents = [{'code': code, 'fitness': None, 'error': None} for code in initial_codes if code]

            # 2. Evaluate Initial Code
            evaluated_agents = self._evaluate_population(temp_agents)

            # 3. NEW: Attempt to Repair Initial Failures
            # We reuse the existing repair worker logic here
            print("Attempting to repair invalid initial agents...")
            with ThreadPoolExecutor(max_workers=2) as executor:
                repaired_agents = list(executor.map(self._repair_agent_worker, evaluated_agents))

            # 4. Populate the PSO swarm with valid agents
            count_valid = 0
            for agent in repaired_agents:
                if agent['fitness'] != float('inf'):
                    count_valid += 1
                    pso_agent = {
                        'current_code': agent['code'],
                        'current_fitness': agent['fitness'],
                        'p_best_code': agent['code'],
                        'p_best_fitness': agent['fitness']
                    }
                    self.population.append(pso_agent)
                    if agent['fitness'] < self.g_best['fitness']:
                        self.g_best['fitness'] = agent['fitness']
                        self.g_best['code'] = agent['code']
                        print(f"New global best found during initialization! Fitness: {self.g_best['fitness']:.4f}")

            print(f"Initialization complete. {count_valid}/{self.mu} agents successfully initialized.")

    def _pso_update_worker(self, agent):
        # --- Using Safe String Format ---
        pso_prompt = (
            "You are an expert programmer participating in a Particle Swarm Optimization-like process to solve a problem.\n"
            "Your goal is to generate a new, improved solution by synthesizing ideas from three sources:\n"
            "1. Your **Current Code**: The last solution you generated.\n"
            "2. Your **Personal Best Code**: The best solution you have personally found so far.\n"
            "3. The **Global Best Code**: The best solution found by anyone in the entire population.\n\n"
            "Problem: Improve the sampling acquisition function so the model is trained more efficiently, model and sampling strategy can be changed.\n\n"
            f"**Global Best Code (Fitness: {self.g_best['fitness']:.4f}):**\n"
            f"```python\n{self.g_best['code']}\n```\n"
            f"**Your Personal Best Code (Fitness: {agent['p_best_fitness']:.4f}):**\n"
            f"```python\n{agent['p_best_code']}\n```\n"
            f"**Your Current Code (Fitness: {agent['current_fitness']:.4f}):**\n"
            f"```python\n{agent['current_code']}\n```\n"
            "Output only the raw, complete, corrected Python code in a single code block. Ensure all necessary imports like import numpy as np are included."
        )
        return self._llm_query(pso_prompt, expect_code=True)

    # --- NEW: The LLM-powered Diversity Agent ---
    def _filter_for_diversity_with_agent(self, candidate_agents):
        """
        Uses a single LLM call to act as a 'curator', selecting a diverse
        subset of candidate agents to prevent premature convergence.
        """
        valid_candidates = [agent for agent in candidate_agents if agent['fitness'] != float('inf')]
        if not valid_candidates:
            print("--- Diversity Agent: No valid candidates to filter. ---")
            return []

        archive_str = "".join(
            f"--- Archive Solution {i} (Fitness: {agent['current_fitness']:.4f}) ---\n"
            f"```python\n{agent['current_code']}\n```\n\n"
            for i, agent in enumerate(self.population)
        )
        candidates_str = "".join(
            f"--- Candidate {i} (Fitness: {agent['fitness']:.4f}) ---\n"
            f"```python\n{agent['code']}\n```\n\n"
            for i, agent in enumerate(valid_candidates)
        )

        diversity_prompt = (
            "You are a Diversity Agent in an evolutionary algorithm. Your job is to prevent the population from converging on a single idea too quickly.\n"
            "You will be given an 'Archive' of existing solutions and a list of new 'Candidates'.\n\n"
            "Analyze the Candidates. For each one, determine if its core algorithm or approach is conceptually distinct from all solutions in the Archive AND from the other Candidates you decide to keep.\n"
            "Your goal is to select a subset of candidates that introduces the most new ideas. Prioritize novelty over small fitness improvements.\n\n"
            f"## Archive (Existing Solutions)\n{archive_str}"
            f"## Candidates (New Solutions to evaluate for diversity)\n{candidates_str}"
            "Based on your analysis, provide a comma-separated list of the indices of the Candidates that should be kept (e.g., `0, 2, 4`).\n"
            "Only return the indices. Do not provide any explanation or other text."
        )

        response = self._llm_query(diversity_prompt, expect_code=False)

        diverse_new_agents = []
        try:
            indices_to_keep = [int(i.strip()) for i in response.split(',')]
            for index in indices_to_keep:
                if 0 <= index < len(valid_candidates):
                    diverse_new_agents.append(valid_candidates[index])
            print(f"--- Diversity Agent: Kept indices {indices_to_keep}. Total {len(diverse_new_agents)} of {len(valid_candidates)} valid candidates. ---")
        except (ValueError, IndexError):
            print(f"--- Diversity Agent Error: Could not parse response '{response}'. Keeping all valid candidates as a fallback. ---")
            return valid_candidates

        return diverse_new_agents

    def search(self):
        self._initialize_population()
        if not self.population:
            print("Initialization failed to produce any valid code. Stopping.")
            return None

        iteration = 0
        while self.query_calls < self.n_queries:
            iteration += 1
            print(f"\n--- PSO Iteration {iteration} | LLM Queries: {self.query_calls}/{self.n_queries} ---")

            # 1. Generate new code for each agent in parallel
            with ThreadPoolExecutor(max_workers=1) as executor:
                new_codes = list(executor.map(self._pso_update_worker, self.population))

            # 2. Evaluate and Repair the new solutions
            new_agents = [{'code': code, 'fitness': None, 'error': None} for code in new_codes if code]
            if not new_agents:
                print("No new agents were generated in this iteration (likely due to query budget).")
                continue

            evaluated_agents = self._evaluate_population(new_agents)
            repaired_agents = []
            with ThreadPoolExecutor(max_workers=2) as executor:
                repaired_agents = list(executor.map(self._repair_agent_worker, evaluated_agents))


          # --- 3. NEW STEP: Enforce Diversity ---
            diverse_repaired_agents = self._filter_for_diversity_with_agent(repaired_agents)
            if not diverse_repaired_agents:
                print("No diverse agents were found to update the population.")
                continue

            # --- 4. MODIFIED STEP: Update based on the DIVERSE set ---
            for agent, new_solution in zip(self.population, diverse_repaired_agents):
                if new_solution['fitness'] != float('inf'):
                    agent['current_code'] = new_solution['code']
                    agent['current_fitness'] = new_solution['fitness']
                    if new_solution['fitness'] < agent['p_best_fitness']:
                        agent['p_best_code'] = new_solution['code']
                        agent['p_best_fitness'] = new_solution['fitness']
                    if new_solution['fitness'] < self.g_best['fitness']:
                        self.g_best['code'] = new_solution['code']
                        self.g_best['fitness'] = new_solution['fitness']
                        print(f"!!! New Global Best Found !!! Fitness: {self.g_best['fitness']:.4f}")



            print(f"End of Iteration {iteration}. Global best fitness so far: {self.g_best['fitness']:.4f}")

        return self.g_best

if __name__ == '__main__':
    #MODEL_TO_USE = 'google/gemini-pro'
    MODEL_TO_USE = 'google/gemini-flash'
    #MODEL_TO_USE = "lbl/cborg-mini"
    #MODEL_TO_USE = "lbl/cborg-chat:latest"
    #MODEL_TO_USE = 'lbl/cborg-coder:latest'
    #MODEL_TO_USE = "lbl/cborg-chat:latest"
    #MODEL_TO_USE = 'openai/gpt-5-nano'





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
            # - Can change the model as long as it provides some kind of uncertainty estimation.
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
        evaluator=evaluate_code,
        n_queries=200,
        population_size=10,
    )
    best_solution = evolver.search()
    print (best_solution)
    if best_solution and best_solution.get('fitness') != float('inf'):
        print("\n\n--- Evolution Complete ---")
        print(f"Best fitness (neg_radii_sum) found: {best_solution['fitness']:.4f}")
        print("Best code found:")
        print("```python")
        print(best_solution['code'])
        print("```")
    else:
        print("\n\n--- Evolution Complete ---")
        print("No valid solutions were found.")

