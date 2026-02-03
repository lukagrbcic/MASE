import sys
import numpy as np
from sphere_packing import circle_packing26


NUM_CIRCLES = 26
TOL = 1e-6

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

def run():
    """
    Runs the packing and validation, returning a tuple for success or failure.
    """
    try:
        # --- The "Happy Path" ---
        # All of this code is attempted first.
        circles = circle_packing26()
        validate_packing_radii(circles[:, -1])
        validate_packing_overlap_wtol(circles, TOL)
        validate_packing_unit_square_wtol(circles, TOL)
        radii_sum = np.sum(circles[:, -1])
        fitness = radii_sum

        # If we reach this line, it means no errors occurred.
        # Return the success tuple.
        return (fitness, True)

    except Exception as e:
        # --- The "Error Path" ---
        # If any line in the `try` block raises an Exception,
        # the code jumps directly to here.
        error_message = str(e)

        # Return the failure tuple.
        return (float('inf'), error_message)


result_tuple = run()

#if result_tuple[1] is True:
    ## SUCCESS CASE
    #score = result_tuple[0]
    #print(f"PERFORMANCE_SCORE:{score}")
    #sys.exit(0)  # Exit cleanly
#else:
    ## FAILURE CASE
    ## The detailed error message is the second element of the tuple.
    #error_message = result_tuple[1]
    #print(error_message, file=sys.stderr)
    #sys.exit(1)  # Exit with an error code

# Check success flag
if result_tuple[1] is True:
    score = result_tuple[0]

    # SAVE TO FILE INSTEAD OF PRINTING
    results = {
        "status": "success",
        "score": score
    }
    with open("results.json", "w") as f:
        json.dump(results, f)

    sys.exit(0)
else:
    error_message = result_tuple[1]

    # You can also write errors to the JSON if you want structured error handling
    results = {
        "status": "error",
        "message": error_message
    }
    with open("results.json", "w") as f:
        json.dump(results, f)

    #print(error_message, file=sys.stderr) # Keep this for logging if desired
    sys.exit(1)
