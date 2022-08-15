"""Import this before numpy."""
import sys
import os

if "numpy" in sys.modules:
    raise RuntimeError(
        "numpy already imported, setting OPENBLAS_NUM_THREADS "
        "won't have desired effect"
    )

# Prevent pathological multithreading with openblas, needs to be set before
# numpy import.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
