#!/usr/bin/env python
"""
Case 4 parallel job: SVIDAG (noninformative prior).

Writes case_4_results_svidag.json alongside run_case4.py.
"""
from _single_algo import run_subset

if __name__ == "__main__":
    run_subset(labels=["SVI-DAG (noninformative)"], output_suffix="svidag")
