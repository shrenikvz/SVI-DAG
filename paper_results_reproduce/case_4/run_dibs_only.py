#!/usr/bin/env python
"""
Case 4 parallel job: DiBS.
Writes case_4_results_dibs.json alongside run_case4.py.
"""
from _single_algo import run_subset

if __name__ == "__main__":
    run_subset(labels=["DiBS"], output_suffix="dibs")
