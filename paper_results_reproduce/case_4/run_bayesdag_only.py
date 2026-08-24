#!/usr/bin/env python
"""
Case 4 parallel job: BayesDAG.
Writes case_4_results_bayesdag.json alongside run_case4.py.
"""
from _single_algo import run_subset

if __name__ == "__main__":
    run_subset(labels=["BayesDAG"], output_suffix="bayesdag")
