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
import itertools

# Example Usage
#MODEL_TO_USE = "lbl/cborg-deepthought:latest"
#MODEL_TO_USE = "lbl/cborg-chat:latest"
#MODEL_TO_USE = 'google/gemini-flash'
#MODEL_TO_USE = 'xai/grok-mini'
#MODEL_TO_USE = "lbl/cborg-mini"
#MODEL_TO_USE = 'google/gemini-flash-lite'
#MODEL_TO_USE = 'google/qwen3'
#MODEL_TO_USE = 'gpt-oss-20b-high'
MODEL_TO_USE = 'gemini-2.0-flash'
#MODEL_TO_USE = 'gcp/qwen-3'
#MODEL_TO_USE = 'gcp/gpt-oss-20b'



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

#results_full = []



#strategies = ['(mu,lambda)', '(mu+lambda)']
#n_lambda = [2, 3, 4]
#memetic_period = [1, 3, 5]
#inspiration_prob = [0.1, 0.5, 0.8]
#tournament_selection_k = [0, 2, 3]
#diversity_agent = [True, False]
#ideas_agent = [True, False]
#mu = [5, 10, 20]
#repairs = [0, 1, 2]

#evolver = LLMAgentEvolver(
    #problem_description=PROBLEM_PROMPT,
    #model_name=MODEL_TO_USE,
    #n_queries=5,
    #mu=2,
    #evaluator=code_evaluator,
    #mutate_recombine_context=MUTATE_RECOMBINE_PROMPT,
    #max_repair_attempts=1,
    #n_lambda=3,
    #n_jobs_eval=2,
    #n_jobs_query=2,
    #tournament_selection_k=0,
    #memetic_period=1,
    #inspiration_prob=0.2,
    #diversity_agent=False,
    #ideas_agent=False,
    #strategy='(mu+lambda)'
#)

#result_mean, result_std = evolver.run_batch(1)

#plt.plot(np.arange(0, len(result_mean), 1), result_mean*-1)

#plt.savefig('test.png')



results_full = []

strategies = ['(mu,lambda)', '(mu+lambda)']
n_lambda = [2, 3, 4]
memetic_period = [1, 3, 5]
inspiration_prob = [0.1, 0.5, 0.8]
tournament_selection_k = [0, 2, 3]
diversity_agent = [True, False]
ideas_agent = [True, False]
mu = [5, 10, 20]
repairs = [0, 1, 2]

strategies = ['(mu+lambda)']
n_lambda = [2]
memetic_period = [1]
inspiration_prob = [0.1, 0.2]
tournament_selection_k = [0]
diversity_agent = [False]
ideas_agent = [False]
mu = [5]
repairs = [0]


# Create a generator of all possible combinations
param_grid = itertools.product(
    strategies, n_lambda, memetic_period, inspiration_prob,
    tournament_selection_k, diversity_agent, ideas_agent, mu, repairs
)

for strategy, nl, mp, ip, tsk, div, idea, m, r in param_grid:

    print(f"Running: Strat={strategy}, Mu={m}, Lambda={nl}...")

    # Initialize and run evolver
    evolver = LLMAgentEvolver(
        problem_description=PROBLEM_PROMPT,
        model_name=MODEL_TO_USE,
        n_queries=5,
        mu=m,
        evaluator=code_evaluator,
        mutate_recombine_context=MUTATE_RECOMBINE_PROMPT,
        max_repair_attempts=r,
        n_lambda=nl,
        n_jobs_eval=2,
        n_jobs_query=2,
        tournament_selection_k=tsk,
        memetic_period=mp,
        inspiration_prob=ip,
        diversity_agent=div,
        ideas_agent=idea,
        strategy=strategy
    )

    result_mean, result_std = evolver.run_batch(3)

    safe_strat = strategy.replace('(', '').replace(')', '').replace('+', 'plus').replace(',', '')

    file_id = (f"strat-{safe_strat}_nl-{nl}_mp-{mp}_ip-{ip}_"
               f"k-{tsk}_div-{div}_idea-{idea}_mu-{m}_rep-{r}")

    # Readable string for the Legend
    legend_str = (f"S:{strategy} L:{nl} MP:{mp} IP:{ip} "
                  f"K:{tsk} Div:{div} Idea:{idea} M:{m} R:{r}")

    # 2. Store results (Make sure 'legend_label' is saved here)
    results_full.append({
        "legend_label": legend_str,
        "mean": result_mean,
        "std": result_std
    })

    # 3. Individual Plot
    x_axis = np.arange(0, len(result_mean), 1)
    y_mean = result_mean * -1

    plt.figure()
    plt.plot(x_axis, y_mean, label='Mean')
    plt.fill_between(x_axis, y_mean - result_std, y_mean + result_std, alpha=0.3, label='Std Dev')

    plt.title(legend_str, fontsize=8)
    plt.legend()
    plt.savefig(f"ablation_{file_id}.png")
    plt.close()

# 4. Summary Plot (All Results)
plt.figure(figsize=(14, 10))

for res in results_full:
    mean = res['mean']
    std = res['std']
    # Use .get() to avoid crashing if a key is missing, defaulting to "Unknown"
    label_text = res.get('legend_label', 'Unknown Parameters')

    x_axis = np.arange(0, len(mean), 1)
    y_mean = mean * -1

    # Plot line
    plt.plot(x_axis, y_mean, alpha=0.7, linewidth=1.5, label=label_text)

    # Fill uncertainty
    plt.fill_between(x_axis, y_mean - std, y_mean + std, alpha=0.05)

plt.title("Combined Ablation Results")
plt.xlabel("Query")
plt.ylabel("Score")

# Legend settings
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='x-small', ncol=1)

plt.tight_layout()
plt.savefig("ablation_summary_all.png")
plt.close()
