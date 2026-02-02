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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Example Usage
#MODEL_TO_USE = "lbl/cborg-deepthought:latest"
#MODEL_TO_USE = "lbl/cborg-chat:latest"
MODEL_TO_USE = 'google/gemini-flash'
#MODEL_TO_USE = 'xai/grok-mini'
#MODEL_TO_USE = "lbl/cborg-mini"



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

MUTATE_RECOMBINE_PROMPT = f"""You are an expert computational geometry programmer. Your task is to improve a solution for a circle packing problem.
The goal is to place 26 non-overlapping circles inside a 1x1 unit square to maximize the sum of their radii."""

from evaluator import CodeEvaluator

sphere_evaluator = CodeEvaluator(
    project_path="SpherePacking",
    target_relative_path="sphere_packing.py",
    execution_script="get_result.py"
)

code_evaluator = sphere_evaluator.evaluate



evolver = LLMAgentEvolver(
    problem_description=PROBLEM_PROMPT,
    model_name=MODEL_TO_USE,
    n_queries=200,
    mu=20,
    evaluator=code_evaluator,
    mutate_recombine_context=MUTATE_RECOMBINE_PROMPT,
    max_repair_attempts=2,
    n_jobs_eval=10,
    n_jobs_query=1,
    tournament_selection_k=0,
    memetic_period=5,
    strategy='(mu+lambda)'
)
agents, history = evolver.search()
evolver.save_convergence_plot(filename="convergence_graph.png")

