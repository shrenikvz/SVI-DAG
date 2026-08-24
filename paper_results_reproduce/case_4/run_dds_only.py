#!/usr/bin/env python
"""
Case 4 parallel job: DDS (VI-DP-DAG autoencoder path).
Writes case_4_results_dds.json alongside run_case4.py.
"""
from _single_algo import run_subset

if __name__ == "__main__":
    run_subset(labels=["DDS"], output_suffix="dds")
