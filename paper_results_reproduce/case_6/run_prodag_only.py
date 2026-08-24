#!/usr/bin/env python
"""Case 6 parallel job: ProDAG."""
from _single_algo import run_subset

if __name__ == "__main__":
    run_subset(labels=["ProDAG"], output_suffix="prodag")
