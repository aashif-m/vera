"""Compatibility wrapper for the old Step 2 script name."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    print("[DEPRECATED] Use scripts/2_distill_decomposition.py instead.")
    runpy.run_path(str(Path(__file__).with_name("2_distill_decomposition.py")), run_name="__main__")
