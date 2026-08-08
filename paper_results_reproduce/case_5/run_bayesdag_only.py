#!/usr/bin/env python
"""Case 5 parallel job: BayesDAG."""
from _single_algo import run_subset

if __name__ == "__main__":
    run_subset(labels=["BayesDAG"], output_suffix="bayesdag")
