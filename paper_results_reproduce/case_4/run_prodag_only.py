#!/usr/bin/env python
"""
Case 4 parallel job: ProDAG.
Writes case_4_results_prodag.json alongside run_case4.py.
"""
from _single_algo import run_subset

if __name__ == "__main__":
    run_subset(labels=["ProDAG"], output_suffix="prodag")
