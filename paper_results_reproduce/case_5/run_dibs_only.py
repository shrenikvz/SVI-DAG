#!/usr/bin/env python
"""Case 5 parallel job: DiBS."""
from _single_algo import run_subset

if __name__ == "__main__":
    run_subset(labels=["DiBS"], output_suffix="dibs")
