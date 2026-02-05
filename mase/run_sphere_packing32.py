import numpy as np
import random
import time
from concurrent.futures import ThreadPoolExecutor
import openai
import os
import re as re
import sys
import threading
from lmuEvolver import LLMAgentEvolver


# Example Usage
#MODEL_TO_USE = "lbl/cborg-deepthought:latest"
MODEL_TO_USE = "lbl/cborg-coder:latest"
#MODEL_TO_USE ='openai/gpt-5-mini'
#MODEL_TO_USE = 'gcp/qwen-3'
MODEL_TO_USE = 'gcp/gpt-oss-20b'

#MODEL_TO_USE = 'google/gemini-flash'


SEED_CODE = """
import numpy as np

def circle_packing32() -> np.ndarray:
    \"\"\"
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    \"\"\"
    n = 32
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
The goal is to place 32 non-overlapping circles inside a 1x1 unit square to maximize the sum of their radii.
Free to use any kind of solution to fix this, including Optimization methods - add imports for this if needed.

Below is a simple but valid, working solution. Your task is to modify this code to find a better packing.
The function must be named `circle_packing32` and return a NumPy array of shape (32, 3).

**Crucially, ensure all necessary imports like `import numpy as np` are included at the top of the script.**

```python
{SEED_CODE}
Output only the raw, complete Python code in a single code block. Do not add any explanation. GIVE ME ONLY THE PYTHON CODE, NO OTHER COMMENTS.
"""

MUTATE_RECOMBINE_PROMPT = f"""You are an expert computational geometry programmer. Your task is to improve a solution for a circle packing problem.
The goal is to place 32 non-overlapping circles inside a 1x1 unit square to maximize the sum of their radii. OUTPUT ONLY THE PYTHON CODE!"""

from evaluator import CodeEvaluator

sphere_evaluator = CodeEvaluator(
    project_path="SpherePacking32",
    target_relative_path="sphere_packing.py",
    execution_script="get_result.py"
)

code_evaluator = sphere_evaluator.evaluate
evolver = LLMAgentEvolver(
    problem_description=PROBLEM_PROMPT,
    model_name=MODEL_TO_USE,
    n_queries=1000,
    mu=10,
    evaluator=code_evaluator,
    mutate_recombine_context=MUTATE_RECOMBINE_PROMPT,
    max_repair_attempts=0,
    n_jobs_eval=10,
    n_jobs_query=5,
    tournament_selection_k=0,
    memetic_period=1,
    strategy='(mu+lambda)'
)

evolver.run_batch(10)

#agents, history = evolver.search()
#evolver.save_convergence_plot(filename="convergence_graph.png")
