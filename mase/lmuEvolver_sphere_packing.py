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
