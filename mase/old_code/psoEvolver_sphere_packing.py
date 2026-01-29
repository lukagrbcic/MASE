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
# START: Real Evaluation Server for Circle Packing (Unchanged)
# ===================================================================

NUM_CIRCLES = 26
TOL = 1e-6
TARGET_FUNCTION_NAME = "circle_packing26"

def validate_packing_radii(radii: np.ndarray) -> None:
    if np.any(radii < 0): raise ValueError("A circle has a negative radius.")
    if np.isnan(radii).any(): raise ValueError("A circle has a NaN radius.")

def validate_packing_unit_square_wtol(circles: np.ndarray, tol: float = 1e-6) -> None:
    for i in range(len(circles)):
        x, y, r = circles[i]
        if (x - r < -tol) or (x + r > 1 + tol) or (y - r < -tol) or (y + r > 1 + tol):
            raise ValueError(f"Circle {i} is outside the unit square.")

def validate_packing_overlap_wtol(circles: np.ndarray, tol: float = 1e-6) -> None:
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((circles[i, :2] - circles[j, :2]) ** 2))
            if dist < circles[i, 2] + circles[j, 2] - tol:
                raise ValueError(f"Circles {i} and {j} overlap.")

def evaluation_server(code_string: str) -> tuple[float, str | None]:
    if not code_string or not code_string.strip():
        return (float('inf'), "Code string is empty.")
    try:
        scope = {}
        exec(code_string, scope)
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
        print(f"Evaluated successfully. Fitness (neg_radii_sum): {fitness:.4f}")
        return (fitness, None)
    except Exception as e:
        return (float('inf'), str(e))

# ===================================================================
# END: Real Evaluation Server
# ===================================================================

class LLMAgentEvolver:
    
    # --- MODIFIED: Simplified __init__ for PSO ---
    def __init__(self, problem_description, model_name, n_queries, population_size, max_repair_attempts=1):
        self.problem_description = problem_description
        self.model_name = model_name
        self.n_queries = n_queries
        self.mu = population_size  # mu is now just the population size
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
    
    # --- UNCHANGED: Repair worker is still valuable ---
    def _repair_agent_worker(self, agent):
        if agent.get('fitness', float('inf')) != float('inf'):
            return agent
        print(f"--- Attempting to repair code with error: {agent.get('error')} ---")
        current_code, current_error = agent['code'], agent['error']
        for i in range(self.max_repair_attempts):
            if self.query_calls >= self.n_queries:
                print("Query budget exhausted, skipping repair.")
                break
            repair_prompt = f"""You are an expert Python programmer and debugger... Fix the bug... Output only the raw, complete, corrected Python code..."""
            fixed_code = self._llm_query(repair_prompt)
            if not fixed_code: continue
            fitness, error = evaluation_server(fixed_code)
            if fitness != float('inf'):
                print(f"--- Repair successful after {i+1} attempt(s)! ---")
                agent.update({'code': fixed_code, 'fitness': fitness, 'error': None})
                return agent
            else:
                print(f"--- Repair attempt {i+1} failed. New error: {error} ---")
                current_code, current_error = fixed_code, error
        return agent
    
    # --- UNCHANGED: LLM query utility ---
    def _llm_query(self, prompt):
        with self.query_lock:
            if self.query_calls >= self.n_queries: return ""
            self.query_calls += 1
            current_query_num = self.query_calls
        print(f"--- LLM Query #{current_query_num}/{self.n_queries} ---")
        try:
            llm_instance = LLM(query=prompt, model=self.model_name, temperature=0.7)
            response_text = llm_instance.get_response()
            if response_text is None: return ""
            return llm_instance.get_code(response_text)
        except IndexError:
            return response_text.strip() if response_text else ""
        except Exception as e:
            print(f"An unexpected error occurred in _llm_query: {e}")
            return ""

    # --- MODIFIED: Initialization now creates particles with personal bests and sets global best ---
    def _initialize_population(self):
        print(f"Initializing population of size {self.mu}...")
        prompts = [self.problem_description] * self.mu
        with ThreadPoolExecutor(max_workers=10) as executor:
            initial_codes = list(executor.map(self._llm_query, prompts))
        
        # Create temporary agents for evaluation
        temp_agents = [{'code': code, 'fitness': None, 'error': None} for code in initial_codes if code]
        evaluated_agents = self._evaluate_population(temp_agents)
        
        for agent in evaluated_agents:
            if agent['fitness'] != float('inf'):
                # The PSO-style agent has a current state and a personal best state
                pso_agent = {
                    'current_code': agent['code'],
                    'current_fitness': agent['fitness'],
                    'p_best_code': agent['code'],
                    'p_best_fitness': agent['fitness']
                }
                self.population.append(pso_agent)
                
                # Update global best if this agent is the best so far
                if agent['fitness'] < self.g_best['fitness']:
                    self.g_best['fitness'] = agent['fitness']
                    self.g_best['code'] = agent['code']
                    print(f"New global best found during initialization! Fitness: {self.g_best['fitness']:.4f}")

    # --- NEW: The core PSO-inspired worker function ---
    def _pso_update_worker(self, agent):
        """Generates a new solution inspired by the agent's current, personal best, and global best states."""
        
        pso_prompt = f"""
        You are an expert programmer participating in a Particle Swarm Optimization-like process to solve a problem.
        Your goal is to generate a new, improved solution by synthesizing ideas from three sources:
        1. Your **Current Code**: The last solution you generated.
        2. Your **Personal Best Code**: The best solution you have personally found so far.
        3. The **Global Best Code**: The best solution found by anyone in the entire population.
        
        Problem: Place 26 non-overlapping circles in a 1x1 unit square to maximize the sum of their radii.
        
        Analyze all three solutions. Draw inspiration from the Global Best's successful strategy, incorporate the strengths of your Personal Best, and try to innovate. Create a new version of the code that has the potential to be even better.
        
        **Global Best Code (Fitness: {self.g_best['fitness']:.4f}):**
        ```python
        {self.g_best['code']}
        Your Personal Best Code (Fitness: {agent['p_best_fitness']:.4f}):
        {agent['p_best_code']}
        Your Current Code (Fitness: {agent['current_fitness']:.4f}):
        {agent['current_code']}
        Output only the raw, complete, corrected Python code in a single code block. Ensure all necessary imports like import numpy as np are included.
        """
        return self._llm_query(pso_prompt)
    
    def search(self):
        self._initialize_population()
        if not self.population:
            print("Initialization failed to produce any valid code. Stopping.")
            return None
    
        iteration = 0
        iteration = 0
        while self.query_calls < self.n_queries:
            iteration += 1
            print(f"\n--- PSO Iteration {iteration} | LLM Queries: {self.query_calls}/{self.n_queries} ---")
            
            # 1. Generate new code for each agent in parallel
            with ThreadPoolExecutor(max_workers=10) as executor:
                new_codes = list(executor.map(self._pso_update_worker, self.population))
            
            # 2. Evaluate and Repair the new solutions
            # This line correctly filters out agents that failed generation due to budget
            new_agents = [{'code': code, 'fitness': None, 'error': None} for code in new_codes if code]
            if not new_agents:
                print("No new agents were generated in this iteration (likely due to query budget).")
                continue
            
            evaluated_agents = self._evaluate_population(new_agents)
            repaired_agents = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                repaired_agents = list(executor.map(self._repair_agent_worker, evaluated_agents))
            
            # 3. Update agent states, personal bests, and global best
            for agent, new_solution in zip(self.population, repaired_agents):
                if new_solution['fitness'] != float('inf'):
                    # Update agent's current position
                    agent['current_code'] = new_solution['code']
                    agent['current_fitness'] = new_solution['fitness']
    
                    # Update personal best (p_best)
                    if new_solution['fitness'] < agent['p_best_fitness']:
                        agent['p_best_code'] = new_solution['code']
                        agent['p_best_fitness'] = new_solution['fitness']
    
                    # Update global best (g_best)
                    if new_solution['fitness'] < self.g_best['fitness']:
                        self.g_best['code'] = new_solution['code']
                        self.g_best['fitness'] = new_solution['fitness']
                        print(f"!!! New Global Best Found !!! Fitness: {self.g_best['fitness']:.4f}")
    
            print(f"End of Iteration {iteration}. Global best fitness so far: {self.g_best['fitness']:.4f}")
            
        return self.g_best
        
# Example Usage
#if __name__ == '__main__':
#MODEL_TO_USE = "lbl/cborg-deepthought:latest"
#MODEL_TO_USE = 'xai/grok-mini'
MODEL_TO_USE = 'google/gemini-flash'
#MODEL_TO_USE = "lbl/cborg-chat:latest"
#MODEL_TO_USE = 'google/gemini-pro'



#model='google/gemini-flash'
#model='anthropic/claude-haiku'
#model='openai/o4-mini'
#model='openai/o3-mini'
#model='openai/gpt-5-mini'
#MODEL_TO_USE='lbl/cborg-coder:latest'
#model='lbl/cborg-deepthought:latest'
#model = 'xai/grok-mini'


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

**Crucially, ensure all necessary imports like `import numpy as np` are included at the top of the script.**

```python
{SEED_CODE}
Output only the raw, complete Python code in a single code block. Do not add any explanation.
"""

evolver = LLMAgentEvolver(
    problem_description=PROBLEM_PROMPT,
    model_name=MODEL_TO_USE,
    n_queries=200,  # This is your LLM call budget
    population_size=10,  # This is the number of "particles",
)

best_solution = evolver.search()

# 2. We now check if the returned dictionary is valid.
if best_solution and best_solution.get('fitness') != float('inf'):
    print("\n\n--- Evolution Complete ---")

    # 3. We no longer need the line `best_solution = history[-1]` because we already have it.
    print(f"Best fitness (neg_radii_sum) found: {best_solution['fitness']:.4f}")
    print("Best code found:")
    print("```python")
    print(best_solution['code'])
    print("```")
else:
    print("\n\n--- Evolution Complete ---")
    print("No valid solutions were found.")

# Place this function near your evaluation_server function
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
from matplotlib.patches import Circle, Rectangle
def plot_final_packing(circles: np.ndarray, radii_sum: float):
    """
    Uses matplotlib to draw the final circle packing solution.
    """
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
    #plt.show()

# We also need to add a check here to ensure we only plot if a solution was found.
if best_solution and best_solution.get('fitness') != float('inf'):
    print("\nGenerating plot of the final solution...")
    scope = {}
    exec(best_solution['code'], scope)
    model_func = scope[TARGET_FUNCTION_NAME]
    final_circles = np.array(model_func())
    final_radii_sum = np.sum(final_circles[:, -1])
    plot_final_packing(final_circles, final_radii_sum)
