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

class SequentialEvolver:
    
    def __init__(self, problem_description, model_name, n_queries):
        self.problem_description = problem_description
        self.model_name = model_name
        self.n_queries = n_queries
        
        self.best_solution = {'code': None, 'fitness': float('inf')}
        self.query_calls = 0
        self.query_lock = threading.Lock() # Kept for compatibility with _llm_query

    def _llm_query(self, prompt):
        """A simple, single-threaded query wrapper."""
        with self.query_lock:
            if self.query_calls >= self.n_queries:
                return ""
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

    def search(self):
        """Performs a simple, sequential evolutionary search (hill-climbing)."""
        # 1. Initialization
        print("--- Initializing with the first solution ---")
        initial_code = self._llm_query(self.problem_description)

        print (initial_code)

        if not initial_code:
            print("Failed to get an initial solution from the LLM.")
            return self.best_solution

        fitness, error = evaluation_server(initial_code)
        if error is None:
            self.best_solution = {'code': initial_code, 'fitness': fitness}
            print(f"Initial solution is valid. Fitness: {fitness:.4f}")
        else:
            print(f"Initial solution failed evaluation: {error}")
            # Even if it fails, we keep it as a basis for mutation.
            self.best_solution['code'] = initial_code

        # 2. Sequential Mutation Loop
        while self.query_calls < self.n_queries:
            print(f"\n--- Iteration {self.query_calls + 1} | Best Fitness So Far: {self.best_solution['fitness']:.4f} ---")
            
            # Create a mutation prompt based on the current best code
            mutate_prompt = (
                "You are an expert programmer. Critically analyze and improve this Python code for the circle packing problem.\n"
                "Fix any bugs, improve the logic, or make the geometric packing more efficient to maximize the sum of radii.\n\n"
                "Current Code:\n"
                "```python\n"
                f"{self.best_solution['code']}\n"
                "```\n\n"
                "Output only the raw, improved Python code in a single code block. Ensure all necessary imports are included."
            )

            # Get a new, mutated solution
            new_code = self._llm_query(mutate_prompt)
            if not new_code:
                print("Skipping iteration due to empty response from LLM.")
                continue

            # Evaluate the new solution
            new_fitness, new_error = evaluation_server(new_code)

            # 3. Selection (Hill-climbing)
            if new_error is None:
                if new_fitness < self.best_solution['fitness']:
                    print(f"!!! Improvement Found !!! New best fitness: {new_fitness:.4f}")
                    self.best_solution = {'code': new_code, 'fitness': new_fitness}
                else:
                    print(f"Mutation was not an improvement. Fitness: {new_fitness:.4f}. Discarding.")
            else:
                print(f"Mutated code failed evaluation: {new_error}")
        
        return self.best_solution

if __name__ == '__main__':
    MODEL_TO_USE = 'google/gemini-flash'
    #MODEL_TO_USE = 'lbl/cborg-coder'

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
    # Use the new SequentialEvolver
    evolver = SequentialEvolver(
        problem_description=PROBLEM_PROMPT,
        model_name=MODEL_TO_USE,
        n_queries=200,  # This is your LLM call budget
    )

    best_solution = evolver.search()

    # The rest of the script is identical to the PSO versions
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
        plt.savefig('sequential_test.pdf') # Saved to a different file name

    if best_solution and best_solution.get('fitness') != float('inf'):
        print("\nGenerating plot of the final solution...")
        scope = {}
        exec(best_solution['code'], scope)
        model_func = scope[TARGET_FUNCTION_NAME]
        final_circles = np.array(model_func())
        final_radii_sum = np.sum(final_circles[:, -1])
        plot_final_packing(final_circles, final_radii_sum)
