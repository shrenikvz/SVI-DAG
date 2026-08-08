#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.tree_util as tree_util


DEFAULT_MODULE = 'svidag.train'
DEFAULT_CALLABLE = 'train_step'
DEFAULT_INPUTS_FN = 'build_benchmark_inputs'
DEFAULT_MODULE_PATH = '.'
DEFAULT_WARMUP_RUNS = 3
DEFAULT_MEASURE_RUNS = 10


def _block_pytree(value):
    for leaf in tree_util.tree_leaves(value):
        blocker = getattr(leaf, "block_until_ready", None)
        if callable(blocker):
            blocker()


def _load_inputs(module, fn_name):
    builder = getattr(module, fn_name)
    payload = builder()
    metadata = {}
    if isinstance(payload, dict):
        args = list(payload.get("args", []))
        kwargs = dict(payload.get("kwargs", {}))
        metadata = dict(payload.get("metadata", {}))
        return args, kwargs, metadata
    if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[1], dict):
        args, kwargs = payload
        return list(args), dict(kwargs), metadata
    if isinstance(payload, tuple):
        return list(payload), {}, metadata
    return [payload], {}, metadata


def main():
    parser = argparse.ArgumentParser(description="Reusable JAX benchmark harness")
    parser.add_argument("--module", default=DEFAULT_MODULE)
    parser.add_argument("--callable", dest="callable_name", default=DEFAULT_CALLABLE)
    parser.add_argument("--inputs-fn", default=DEFAULT_INPUTS_FN)
    parser.add_argument("--module-path", default=DEFAULT_MODULE_PATH)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--measure-runs", type=int, default=DEFAULT_MEASURE_RUNS)
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if args.module_path:
        sys.path.insert(0, str(Path(args.module_path).resolve()))

    module = importlib.import_module(args.module)
    fn = getattr(module, args.callable_name)
    call_args, call_kwargs, metadata = _load_inputs(module, args.inputs_fn)

    compile_start = time.perf_counter()
    first_result = fn(*call_args, **call_kwargs)
    _block_pytree(first_result)
    compile_and_first_run_seconds = time.perf_counter() - compile_start

    extra_warmups = max(0, args.warmup_runs - 1)
    for _ in range(extra_warmups):
        warm = fn(*call_args, **call_kwargs)
        _block_pytree(warm)

    if args.trace_dir:
        jax.profiler.start_trace(args.trace_dir)
        traced = fn(*call_args, **call_kwargs)
        _block_pytree(traced)
        jax.profiler.stop_trace()

    steady_state_seconds = []
    for _ in range(max(1, args.measure_runs)):
        start = time.perf_counter()
        result = fn(*call_args, **call_kwargs)
        _block_pytree(result)
        steady_state_seconds.append(time.perf_counter() - start)

    payload = {
        "module": args.module,
        "callable": args.callable_name,
        "inputs_fn": args.inputs_fn,
        "devices": [{"platform": dev.platform, "device_kind": dev.device_kind} for dev in jax.devices()],
        "compile_and_first_run_seconds": compile_and_first_run_seconds,
        "steady_state_seconds": steady_state_seconds,
        "steady_state_median_seconds": statistics.median(steady_state_seconds),
        "steady_state_min_seconds": min(steady_state_seconds),
        "metadata": metadata,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
