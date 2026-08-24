#!/usr/bin/env python
"""
Case 4 parallel job: BCD Nets.
Writes case_4_results_bcd.json alongside run_case4.py.
"""
from _single_algo import run_subset

if __name__ == "__main__":
    run_subset(labels=["BCD Nets"], output_suffix="bcd")
