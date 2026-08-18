import csv
import runpy
from pathlib import Path

csv.field_size_limit(100_000_000)
runpy.run_path(
    str(Path(__file__).with_name("build_all_algorithm_runtime_distribution.py")),
    run_name="__main__",
)
