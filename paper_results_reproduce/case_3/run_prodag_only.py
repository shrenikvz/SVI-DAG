#!/usr/bin/env python
"""Case 3 parallel job: ProDAG."""
from _single_algo import run_subset

if __name__ == "__main__":
    run_subset(labels=["ProDAG"], output_suffix="prodag")
