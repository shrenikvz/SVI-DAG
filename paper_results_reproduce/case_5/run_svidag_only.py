#!/usr/bin/env python
"""Case 5 parallel job: SVI-DAG."""
from _single_algo import run_subset

if __name__ == "__main__":
    run_subset(labels=["SVI-DAG (noninformative)"], output_suffix="svidag")
