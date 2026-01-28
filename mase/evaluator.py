import subprocess
import shutil
import tempfile
import os
import sys

def evaluate_code(generated_code_string: str) -> tuple:
    """
    Evaluates generated code within a project sandbox and returns a simple
    tuple indicating success or failure.

    Args:
        generated_code_string: The Python code to be injected and tested.

    Returns:
        A tuple: (score, True) on success.
        A tuple: (-1, error_message) for any type of failure.
    """
    project_path = "ActiveLearningExperiment"

    # Check if the source project directory exists before starting.
    if not os.path.isdir(project_path):
        error_msg = f"CRITICAL ERROR: Project directory '{project_path}' not found."
        return (-1, error_msg)

    with tempfile.TemporaryDirectory() as sandbox_dir:
        try:
            # Setup the sandbox environment
            project_in_sandbox_path = os.path.join(sandbox_dir, os.path.basename(project_path))
            shutil.copytree(project_path, project_in_sandbox_path)

            # Inject the generated code into the target file
            target_file_path = os.path.join(project_in_sandbox_path, "src/samplers/model_sampler.py")
            with open(target_file_path, "w") as f:
                f.write(generated_code_string)

            # Execute the project's main script
            command = [sys.executable, "run_AL.py"]
            process = subprocess.run(
                command,
                cwd=project_in_sandbox_path,
                capture_output=True,
                text=True,
                timeout=300  # 5-minute timeout
            )

            # --- Process the results ---

            if process.returncode != 0:
                # FAILURE: The script crashed or exited with an error code.
                error_output = (f"Subprocess crashed with return code {process.returncode}.\n\n"
                                f"STDERR:\n{process.stderr.strip()}\n\n"
                                f"STDOUT:\n{process.stdout.strip()}")
                return (-1, error_output)
            else:
                # SUCCESS (so far): The script ran. Now find the score in its output.
                output = process.stdout
                for line in output.splitlines():
                    if line.strip().startswith("PERFORMANCE_SCORE:"):
                        # SUCCESS! We found the score.
                        score = float(line.strip().split(":")[1])
                        return (score, True)

                # FAILURE: The script ran but did not output the required score line.
                error_output = f"Script ran successfully but did not output a 'PERFORMANCE_SCORE:' line. Full output:\n{output.strip()}"
                return (-1, error_output)

        except subprocess.TimeoutExpired:
            # FAILURE: The process took too long.
            return (-1, "Evaluation failed: The process timed out.")
        except Exception as e:
            # FAILURE: An unexpected Python error occurred in this wrapper.
            return (-1, f"An unexpected exception occurred in the evaluation wrapper: {e}")

## --- Your Code String ---
#code_str = """
#import numpy as np
#from indago import PSO
#import random
#from sklearn.ensemble import RandomForestRegressor
#from _PSO import PSO

#class modelSampler:

    #def __init__(self, X, y, sample_size, lb, ub):
#ad
        #self.X = X
        #self.y = y
        #self.sample_size = sample_size
        #self.lb = lb
        #self.ub = ub

        #self.model = RandomForestRegressor().fit(self.X, self.y)

    #def gen_samples(self):
        #print ('running original code!')

        #X = []
        #f = []
        #for i in range(self.sample_size):

            #def get_values(X_population):

                #X_population = np.atleast_2d(X_population)


                #all_preds = np.array([
                    #tree.predict(X_population) for tree in self.model.estimators_
                #])

                #stds_across_trees = np.std(all_preds, axis=0)  # shape: (n_particles, horizon_length)

                #particle_fitness = -np.sum(stds_across_trees, axis=1)  # shape: (n_particles,)

                #assert particle_fitness.shape == (X_population.shape[0],), \
                    #f"Shape mismatch: got {particle_fitness.shape} expected {(X_population.shape[0],)}"

                #return particle_fitness

            #opt = PSO(function=get_values, lb=self.lb, ub=self.ub, swarm_size=10, max_evals=100, device='cpu')
            #min_x, _, min_f = opt.search()


            #min_x = np.ravel(min_x)

            #X.append(min_x)
            #f.append(min_f[0])

        #X = np.array(X)
        #f = np.array(f)

        #X = X[np.argsort(f)]

        #return X, self.model
#"""


#results = evaluate_code(code_str)
#print (results)

