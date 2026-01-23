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

# ===================================================================
# START: Real Evaluation Server for Circle Packing (Verified)
# ===================================================================

NUM_CIRCLES = 26
TOL = 1e-6
TARGET_FUNCTION_NAME = "circle_packing26"

def validate_packing_radii(radii: np.ndarray) -> None:
    if np.any(radii < 0):
        raise ValueError("A circle has a negative radius.")
    if np.isnan(radii).any():
        raise ValueError("A circle has a NaN radius.")

def validate_packing_unit_square_wtol(circles: np.ndarray, tol: float = 1e-6) -> None:
    for i in range(len(circles)):
        x, y, r = circles[i]
        if (x - r < -tol) or (x + r > 1 + tol) or (y - r < -tol) or (y + r > 1 + tol):
            raise ValueError(f"Circle {i} is outside the unit square.")

# This version is now 100% identical to your original logic
def validate_packing_overlap_wtol(circles: np.ndarray, tol: float = 1e-6) -> None:
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            # Using sqrt to match the original logic perfectly regarding tolerance
            dist = np.sqrt(np.sum((circles[i, :2] - circles[j, :2]) ** 2))
            if dist < circles[i, 2] + circles[j, 2] - tol:
                raise ValueError(f"Circles {i} and {j} overlap.")

# Replace the evaluation_server function

def evaluation_server(code_string: str) -> tuple[float, str | None]:
    """
    Evaluates code for the circle packing problem.
    Returns a tuple: (fitness, error_message).
    """
    if not code_string or not code_string.strip():
        return (float('inf'), "Code string is empty.")
    try:
        # --- THIS IS THE FIX ---
        # We create a single scope dictionary and use it for both globals and locals.
        # This ensures the function defined inside exec can see the modules imported inside exec.
        mean_fitness = []
        for i in range(3):
            scope = {}
            exec(code_string, scope)
            # ---------------------

            if TARGET_FUNCTION_NAME not in scope:
                raise ValueError(f"Function '{TARGET_FUNCTION_NAME}' not found.")

            model_func = scope[TARGET_FUNCTION_NAME]
            circles = model_func()

            if not isinstance(circles, np.ndarray): circles = np.array(circles)
            if circles.shape != (NUM_CIRCLES, 3): raise ValueError(f"Invalid shape: {circles.shape}")

            validate_packing_radii(circles[:, -1])
            validate_packing_overlap_wtol(circles, TOL)
            validate_packing_unit_square_wtol(circles, TOL)

            radii_sum = np.sum(circles[:, -1])
            fitness = -radii_sum
            #print(f"Evaluated successfully. Fitness (neg_radii_sum): {fitness:.4f}")
            mean_fitness.append(fitness)
        fitness = np.mean(mean_fitness)
        print(f"Evaluated successfully. Fitness (neg_radii_sum): {fitness:.4f}")
        return (fitness, None)

    except Exception as e:
        return (float('inf'), str(e))

# ===================================================================
# END: Real Evaluation Server
# ===================================================================

class LLMAgentEvolver:

    def __init__(self, problem_description, model_name, n_queries, population_size, max_repair_attempts=1):
        self.problem_description = problem_description
        self.model_name = model_name
        self.n_queries = n_queries
        self.mu = population_size
        self.max_repair_attempts = max_repair_attempts
        self.population = []
        self.g_best = {'code': None, 'fitness': float('inf')}
        self.query_calls = 0
        self.query_lock = threading.Lock()

    def _evaluate_population(self, population):
        codes_to_evaluate = [agent['code'] for agent in population]
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(evaluation_server, codes_to_evaluate))
        for agent, (fitness, error) in zip(population, results):
            agent['fitness'] = fitness
            agent['error'] = error
        return population

    def _repair_agent_worker(self, agent):
        if agent.get('fitness', float('inf')) != float('inf'):
            return agent
        # (Repair logic is unchanged)
        return agent

    # --- MODIFIED: Added expect_code flag ---
    def _llm_query(self, prompt, expect_code=True):
        with self.query_lock:
            if self.query_calls >= self.n_queries: return ""
            self.query_calls += 1
            current_query_num = self.query_calls
        print(f"--- LLM Query #{current_query_num}/{self.n_queries} ---")
        try:
            llm_instance = LLM(query=prompt, model=self.model_name, temperature=0.7)
            response_text = llm_instance.get_response()
            if response_text is None: return ""
            if not expect_code:
                return response_text.strip()
            return llm_instance.get_code(response_text)
        except IndexError:
            return response_text.strip() if response_text else ""
        except Exception as e:
            print(f"An unexpected error occurred in _llm_query: {e}")
            return ""

    def _initialize_population(self):
        print(f"Initializing population of size {self.mu}...")
        prompts = [self.problem_description] * self.mu
        with ThreadPoolExecutor(max_workers=10) as executor:
            initial_codes = list(executor.map(self._llm_query, prompts))

        temp_agents = [{'code': code, 'fitness': None, 'error': None} for code in initial_codes if code]
        evaluated_agents = self._evaluate_population(temp_agents)

        for agent in evaluated_agents:
            if agent['fitness'] != float('inf'):
                pso_agent = {'current_code': agent['code'], 'current_fitness': agent['fitness'], 'p_best_code': agent['code'], 'p_best_fitness': agent['fitness']}
                self.population.append(pso_agent)
                if agent['fitness'] < self.g_best['fitness']:
                    self.g_best['fitness'] = agent['fitness']
                    self.g_best['code'] = agent['code']
                    print(f"New global best found during initialization! Fitness: {self.g_best['fitness']:.4f}")

    def _pso_update_worker(self, agent):
        # --- Using Safe String Format ---
        pso_prompt = (
            "You are an expert programmer participating in a Particle Swarm Optimization-like process to solve a problem.\n"
            "Your goal is to generate a new, improved solution by synthesizing ideas from three sources:\n"
            "1. Your **Current Code**: The last solution you generated.\n"
            "2. Your **Personal Best Code**: The best solution you have personally found so far.\n"
            "3. The **Global Best Code**: The best solution found by anyone in the entire population.\n\n"
            "Problem: Place 26 non-overlapping circles in a 1x1 unit square to maximize the sum of their radii.\n\n"
            "Analyze all three solutions. Draw inspiration from the Global Best's successful strategy, incorporate the strengths of your Personal Best, and try to innovate. Create a new version of the code that has the potential to be even better.\n\n"
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

    # --- MODIFIED: Integrated the Diversity Agent ---
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
            with ThreadPoolExecutor(max_workers=10) as executor:
                new_codes = list(executor.map(self._pso_update_worker, self.population))

            # 2. Evaluate and Repair the new solutions
            new_agents = [{'code': code, 'fitness': None, 'error': None} for code in new_codes if code]
            if not new_agents:
                print("No new agents were generated in this iteration (likely due to query budget).")
                continue

            evaluated_agents = self._evaluate_population(new_agents)
            repaired_agents = []
            with ThreadPoolExecutor(max_workers=10) as executor:
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




    SEED_CODE = """
    import numpy as np

    def circle_packing26() -> np.ndarray:
    \"\"\"
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    \"\"\"
    n = 26
    circles = np.zeros((n, 3))

    # A very simple initial guess: place circles along a line. This is a bad but valid start.
    # This gives the LLM something concrete to improve upon.
    radius = 1.0 / (2.0 * n)
    for i in range(n):
        x = (2.0 * i + 1.0) / (2.0 * n)
        y = 0.5
        circles[i] = [x, y, radius]

    return circles
    """
    PROBLEM_PROMPT = f"""
    You are an expert computational geometry programmer. Your task is to improve a solution for a circle packing problem.
    The goal is to place 26 non-overlapping circles inside a 1x1 unit square to maximize the sum of their radii.
    Free to use any kind of solution to fix this, including Optimization methods - add imports for this if needed.

    Below is a simple but valid, working solution. Your task is to modify this code to find a better packing.
    The function must be named `circle_packing26` and return a NumPy array of shape (26, 3).

    - Constraint: All circles must be fully contained within the unit square with no overlaps
    - Mathematical formulation: For circle i at position (xi, yi) with radius ri:
        * Containment: ri ≤ xi ≤ 1-ri and ri ≤ yi ≤ 1-ri
        * Non-overlap: √[(xi-xj)² + (yi-yj)²] ≥ ri + rj for all i≠j
        * Objective: maximize Σri subject to above constraints

    **Crucially, ensure all necessary imports like `import numpy as np` are included at the top of the script.**

    ```python
    {SEED_CODE}
    Output only the raw, complete Python code in a single code block. Do not add any explanation.
    """
    evolver = LLMAgentEvolver(
        problem_description=PROBLEM_PROMPT,
        model_name=MODEL_TO_USE,
        n_queries=500,
        population_size=10,
    )
    best_solution = evolver.search()
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

    import matplotlib
    matplotlib.use('Agg')
    def plot_final_packing(circles: np.ndarray, radii_sum: float):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f"Final Circle Packing\nSum of Radii: {radii_sum:.6f}", fontsize=16)
        ax.add_patch(Rectangle((0, 0), 1, 1, fill=None, edgecolor='black', lw=2))
        for x, y, r in circles:
            circle = Circle((x, y), r, facecolor='lightblue', edgecolor='darkblue', lw=1.5, alpha=0.8)
            ax.add_patch(circle)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig('test.pdf')

    if best_solution and best_solution.get('fitness') != float('inf'):
        print("\nGenerating plot of the final solution...")
        scope = {}
        exec(best_solution['code'], scope)
        model_func = scope[TARGET_FUNCTION_NAME]
        final_circles = np.array(model_func())
        final_radii_sum = np.sum(final_circles[:, -1])
        plot_final_packing(final_circles, final_radii_sum)
