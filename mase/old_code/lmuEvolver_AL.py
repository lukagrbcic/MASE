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



import numpy as np
import warnings
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Suppress warnings and manage paths ---
warnings.filterwarnings("ignore")
import sys
# Make sure these paths are correct relative to where you run this script
sys.path.insert(0, 'ActiveLearningExperiment')
sys.path.insert(1, 'ActiveLearningExperiment/src')
sys.path.insert(2, 'ActiveLearningExperiment/src/samplers')


from active_learning import activeLearner
from InverseBench.benchmarks import load_model, load_test_data, benchmark_functions



def _single_run(config: dict, f, lb, ub, test_data: tuple) -> dict:
    """One run of an active learning experiment, using pre-loaded benchmark objects."""
    try:
        # Load AL parameters from the config dictionary
        sampler = config['sampler']
        init_size = config['init_size']
        batch_size = config['batch_size']
        max_samples = config['max_samples']

        # NOTE: The benchmark (f), its bounds (lb, ub), and test_data
        # are now passed directly into this function.

        # --- Setup and run the active learner ---
        al_setup = activeLearner(
            f, lb, ub,
            init_size, batch_size, max_samples,
            sampler, test_data,
            verbose=0
        )

        results = al_setup.run(n_runs=1)
        return {"results": results, "error": None}

    except Exception as e:
        tb_str = traceback.format_exc()
        return {"results": None, "error": (e, tb_str)}


def normalize_score(value, goal="max", eps=1e-8):
    if goal == "max":
        return np.clip(value, 0, 1)
    elif goal == "min":
        return 1.0 / (value + 1.0 + eps)

def aggregate_score(r2_int, rmse_int, maxae_int, w_r2=0.333, w_rmse=0.333, w_maxae=0.333):
    r2_norm = normalize_score(r2_int, goal="max")
    rmse_norm = normalize_score(rmse_int, goal="min")
    maxae_norm = normalize_score(maxae_int, goal="min")
    return (w_r2 * r2_norm) + (w_rmse * rmse_norm) + (w_maxae * maxae_norm)


def evaluation_server(code_string: str) -> tuple[float, str | None]:
    """
    Evaluates an active learning experiment for a hardcoded benchmark.
    Returns a tuple: (fitness_score, error_message).
    """
    FAILURE_SCORE = float('-inf')
    # NEW: The benchmark name is now a constant inside the server.
    BENCHMARK_NAME = 'inconel_benchmark'

    try:
        # --- NEW: Load benchmark data ONCE before parallel execution ---
        print(f"Loading benchmark '{BENCHMARK_NAME}'...")
        model = load_model(BENCHMARK_NAME).load()
        f = benchmark_functions(BENCHMARK_NAME, model)
        lb, ub = f.get_bounds()
        test_input, test_output = load_test_data(BENCHMARK_NAME).load()
        test_input = test_input[:3259, :]
        test_output = test_output[:3259, :]
        test_data = (test_input, test_output)
        print("Benchmark loaded successfully.")

        # --- Execute the user-provided code to get the AL configuration ---
        scope = {}
        exec(code_string, scope)

        if 'get_al_config' not in scope:
            raise ValueError("Function 'get_al_config' not found in the provided code.")

        config_func = scope['get_al_config']
        config = config_func()

        N = config.get('num_runs', 1)
        n_jobs = config.get('num_jobs', 1)

        # --- Run experiments in parallel ---
        all_results = []
        errors = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # CHANGED: Pass the pre-loaded objects (f, lb, ub, test_data) to each worker.
            futures = [executor.submit(_single_run, config, f, lb, ub, test_data) for _ in range(N)]
            for idx, fut in enumerate(as_completed(futures), start=1):
                res = fut.result()
                if res["error"] is not None:
                    e, tb_str = res["error"]
                    errors.append((idx, e, tb_str))
                else:
                    all_results.append(res["results"])

        if errors:
            idx, e, tb_str = errors[0]
            error_message = f"Run {idx} failed with exception: {repr(e)}\n{tb_str}"
            return FAILURE_SCORE, error_message

        if not all_results:
             return FAILURE_SCORE, "No successful runs were completed."

        # --- Aggregate scores for successful runs ---
        aggregated_means = {}
        for key in all_results[0].keys():
            values = [r[key] for r in all_results]
            stacked = np.vstack(values)
            aggregated_means[key] = np.mean(stacked, axis=0)

        # --- Compute final aggregated score ---
        mean_r2_int = np.trapz(np.clip(aggregated_means['r2_array'], 0, 1))
        mean_rmse_int = np.trapz(aggregated_means['rmse_array'])
        mean_maxae_int = np.trapz(aggregated_means['nmax_ae_array'])

        final_score = aggregate_score(mean_r2_int, mean_rmse_int, mean_maxae_int)

        print(f"Evaluated successfully. Final Score: {final_score:.4f}")
        return final_score, None

    except Exception as e:
        return FAILURE_SCORE, str(e)




class LLMAgentEvolver:
    
    def __init__(self, problem_description, model_name, n_queries, mu, strategy='(mu,lambda)', max_repair_attempts=1):

        self.problem_description = problem_description
        self.model_name = model_name
        self.n_queries = n_queries
        self.mu = mu
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
        with ThreadPoolExecutor(max_workers=10) as executor:
            # executor.map now returns tuples of (fitness, error)
            results = list(executor.map(evaluation_server, codes_to_evaluate))
        
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
            fitness, error = evaluation_server(fixed_code)
            
            
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
MODEL_TO_USE = 'google/gemini-flash'
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


# The initial "seed" code provided to the LLM
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
    n_queries=200,  # This is your LLM call budget
    mu=10,
    strategy='(mu+lambda)'
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
# Place this function near your evaluation_server function
import matplotlib.pyplot as plt # ADD THIS IMPORT
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
from matplotlib.patches import Circle, Rectangle # ADD THIS IMPORT
def plot_final_packing(circles: np.ndarray, radii_sum: float):
    """
    Uses matplotlib to draw the final circle packing solution.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set up the plot aesthetics
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(f"Final Circle Packing\nSum of Radii: {radii_sum:.6f}", fontsize=16)

    # Draw the unit square boundary
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=None, edgecolor='black', lw=2))

    # Draw each circle
    for x, y, r in circles:
        circle = Circle((x, y), r, facecolor='lightblue', edgecolor='darkblue', lw=1.5, alpha=0.8)
        ax.add_patch(circle)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

 # Now, run the final code and generate the plot
print("\nGenerating plot of the final solution...")
scope = {}
exec(best_solution['code'], scope)
model_func = scope[TARGET_FUNCTION_NAME]
final_circles = np.array(model_func())

# Verify the sum again just in case
final_radii_sum = np.sum(final_circles[:, -1])

plot_final_packing(final_circles, final_radii_sum)
