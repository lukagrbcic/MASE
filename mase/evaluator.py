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
    project_path = "SpherePacking"

    # Check if the source project directory exists before starting.
    if not os.path.isdir(project_path):
        error_msg = f"CRITICAL ERROR: Project directory '{project_path}' not found."
        return (-1, error_msg)

    with tempfile.TemporaryDirectory() as sandbox_dir:
        try:
            # Setup the sandbox environment
            project_in_sandbox_path = os.path.join(sandbox_dir, os.path.basename(project_path))
            shutil.copytree(project_path, project_in_sandbox_path)

            target_file_path = os.path.join(project_in_sandbox_path, "sphere_packing.py")
            with open(target_file_path, "w") as f:
                f.write(generated_code_string)

            # Execute the project's main script
            command = [sys.executable, "get_result.py"]
            process = subprocess.run(
                command,
                cwd=project_in_sandbox_path,
                capture_output=True,
                text=True,
                timeout=300  # 5-minute timeout
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

                # FAILURE: The script ran but did not output the required score line.
                error_output = f"Script ran successfully but did not output a 'PERFORMANCE_SCORE:' line. Full output:\n{output.strip()}"
                return (float('inf'), error_output)

        except subprocess.TimeoutExpired:
            # FAILURE: The process took too long.
            return (float('inf'), "Evaluation failed: The process timed out.")
        except Exception as e:
            # FAILURE: An unexpected Python error occurred in this wrapper.
            return (float('inf'), f"An unexpected exception occurred in the evaluation wrapper: {e}")

