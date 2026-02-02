import subprocess
import shutil
import tempfile
import os
import sys

class CodeEvaluator:
    def __init__(self, project_path: str, target_relative_path: str, execution_script: str = "get_result.py"):
        """
        Initializes the evaluator with specific project paths.

        Args:
            project_path: The path to the root folder of the project (e.g., "SpherePacking").
            target_relative_path: The path to the file inside the project where code should
                                  be injected (e.g., "sphere_packing.py" or "src/model.py").
            execution_script: The name of the script to run (default: "get_result.py").
        """
        # Validate the project path immediately upon initialization
        if not os.path.isdir(project_path):
            raise ValueError(f"CRITICAL ERROR: Project directory '{project_path}' not found.")

        self.project_path = project_path
        self.target_relative_path = target_relative_path
        self.execution_script = execution_script

    def evaluate(self, generated_code_string: str) -> tuple:
        """
        Evaluates generated code within a project sandbox.
        """
        with tempfile.TemporaryDirectory() as sandbox_dir:
            try:
                # Setup the sandbox environment
                # We use os.path.basename to get just the folder name (e.g., "SpherePacking")
                project_in_sandbox_path = os.path.join(sandbox_dir, os.path.basename(self.project_path))

                # Copy the project to the temp directory
                shutil.copytree(self.project_path, project_in_sandbox_path)

                # Determine where to write the generated code based on the instance variable
                target_file_path = os.path.join(project_in_sandbox_path, self.target_relative_path)

                with open(target_file_path, "w") as f:
                    f.write(generated_code_string)

                # Execute the project's main script using the instance variable
                command = [sys.executable, self.execution_script]
                process = subprocess.run(
                    command,
                    cwd=project_in_sandbox_path,
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                if process.returncode != 0:
                    # FAILURE: The script crashed or exited with an error code.
                    error_output = (f"Subprocess crashed with return code {process.returncode}.\n\n"
                                    f"STDERR:\n{process.stderr.strip()}\n\n"
                                    f"STDOUT:\n{process.stdout.strip()}")
                    return (float('inf'), error_output)
                else:
                    # SUCCESS (so far): The script ran. Now find the score in its output.
                    output = process.stdout
                    for line in output.splitlines():
                        if line.strip().startswith("PERFORMANCE_SCORE:"):
                            # SUCCESS! We found the score.
                            score = float(line.strip().split(":")[1])
                            return (-1*score, True)

                    # FAILURE: Script ran but no score found
                    error_output = f"Script ran successfully but did not output a 'PERFORMANCE_SCORE:' line. Full output:\n{output.strip()}"
                    return (float('inf'), error_output)

            except subprocess.TimeoutExpired:
                return (float('inf'), "Evaluation failed: The process timed out.")
            except Exception as e:
                return (float('inf'), f"An unexpected exception occurred in the evaluation wrapper: {e}")

# --- usage example ---

## 1. Initialize for the Sphere Packing project
#sphere_evaluator = CodeEvaluator(
    #project_path="SpherePacking",
    #target_relative_path="sphere_packing.py"
#)

## 2. Initialize for the Active Learning project (based on your commented out code)
#active_learning_evaluator = CodeEvaluator(
    #project_path="ActiveLearningExperiment",
    #target_relative_path="src/samplers/model_sampler.py"
#)

## 3. Use them
## result = sphere_evaluator.evaluate(some_code_string)
## result_2 = active_learning_evaluator.evaluate(other_code_string)
