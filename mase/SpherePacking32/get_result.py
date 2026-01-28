import sys
import numpy as np
from sphere_packing import circle_packing32


NUM_CIRCLES = 32
TOL = 1e-6

def validate_packing_radii(radii: np.ndarray) -> None:
    n = len(radii)
    for i in range(n):
        if radii[i] < 0:
            raise ValueError(f"Circle {i} has negative radius {radii[i]}")
        elif np.isnan(radii[i]):
            raise ValueError(f"Circle {i} has nan radius")


def validate_packing_unit_square_wtol(circles: np.ndarray, tol: float = 1e-6) -> None:
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if (x - r < -tol) or (x + r > 1 + tol) or (y - r < -tol) or (y + r > 1 + tol):
            raise ValueError(
                f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square"
            )


def validate_packing_overlap_wtol(circles: np.ndarray, tol: float = 1e-6) -> None:
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((circles[i, :2] - circles[j, :2]) ** 2))
            if dist < circles[i, 2] + circles[j, 2] - tol:
                raise ValueError(
                    f"Circles {i} and {j} overlap: dist={dist}, r1+r2={circles[i,2]+circles[j,2]}"
                )

def run():
    """
    Runs the packing and validation, returning a tuple for success or failure.
    """
    try:
        # --- The "Happy Path" ---
        # All of this code is attempted first.
        circles = circle_packing32()
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

if result_tuple[1] is True:
    # SUCCESS CASE
    score = result_tuple[0]
    print(f"PERFORMANCE_SCORE:{score}")
    sys.exit(0)  # Exit cleanly
else:
    # FAILURE CASE
    # The detailed error message is the second element of the tuple.
    error_message = result_tuple[1]
    print(error_message, file=sys.stderr)
    sys.exit(1)  # Exit with an error code

