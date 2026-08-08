import argparse
import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.bayesdag_jax import train_from_config_dict


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_array(path: str) -> np.ndarray:
    _, ext = os.path.splitext(path)
    if ext == ".npy":
        array = np.load(path)
    else:
        array = np.loadtxt(path, delimiter=",")
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    return array


def _resolve_config_path(path: str) -> str:
    if os.path.exists(path):
        return path

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(here, path)]
    if path.startswith("open_source/configs/"):
        candidates.append(os.path.join(here, "src", path.replace("open_source/configs/", "configs/", 1)))
    if path.startswith("configs/"):
        candidates.append(os.path.join(here, "src", path))
    if not path.startswith("src/"):
        candidates.append(os.path.join(here, "src", "configs", os.path.basename(path)))
        candidates.append(os.path.join(here, "src", "configs", "bayesdag", os.path.basename(path)))
        candidates.append(os.path.join(here, "src", "configs", "defaults", os.path.basename(path)))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not resolve config path: {path}")


def _default_dataset_config_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "src",
        "configs",
        "defaults",
        "dataset_config.json",
    )


def _find_first_existing(root: str, names) -> Optional[str]:
    for name in names:
        candidate = os.path.join(root, name)
        if os.path.exists(candidate):
            return candidate
    return None


def _resolve_dataset_root(dataset_name: Optional[str], data_dir: Optional[str]) -> str:
    candidates = []
    if dataset_name and os.path.exists(dataset_name):
        candidates.append(dataset_name)
    if data_dir and dataset_name:
        candidates.append(os.path.join(data_dir, dataset_name))
    if data_dir:
        candidates.append(data_dir)

    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue
        train_candidate = _find_first_existing(candidate, ("train.npy", "train.csv", "data.npy", "data.csv"))
        if train_candidate is not None:
            return candidate

    dataset_label = dataset_name if dataset_name is not None else "<unspecified>"
    raise FileNotFoundError(f"Could not resolve a dataset directory for dataset_name={dataset_label!r}, data_dir={data_dir!r}.")


def _load_dataset_from_directory(dataset_root: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    train_path = _find_first_existing(dataset_root, ("train.npy", "train.csv", "data.npy", "data.csv"))
    if train_path is None:
        raise FileNotFoundError(f"No train/data array found in dataset directory: {dataset_root}")

    test_path = _find_first_existing(dataset_root, ("test.npy", "test.csv"))
    adjacency_path = _find_first_existing(
        dataset_root,
        ("adj_matrix.csv", "adj_matrix.npy", "adjacency.npy", "adjacency.csv"),
    )

    train_data = _load_array(train_path)
    test_data = _load_array(test_path) if test_path is not None else None
    adjacency = _load_array(adjacency_path) if adjacency_path is not None else None
    return train_data, test_data, adjacency


def _apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = json.loads(json.dumps(config))
    model_hyperparams = config.setdefault("model_hyperparams", {})
    training_hyperparams = config.setdefault("training_hyperparams", {})

    if args.seed is not None:
        model_hyperparams["random_seed"] = [int(args.seed)]
    if args.scale_noise is not None:
        model_hyperparams["scale_noise"] = float(args.scale_noise)
    if args.scale_noise_p is not None:
        model_hyperparams["scale_noise_p"] = float(args.scale_noise_p)
    if args.lambda_sparse is not None:
        model_hyperparams["lambda_sparse"] = float(args.lambda_sparse)
    if args.learning_rate is not None:
        training_hyperparams["learning_rate"] = float(args.learning_rate)

    return config


def get_parser():
    parser = argparse.ArgumentParser(description="JAX BayesDAG runner with benchmark-compatible dataset loading.")
    parser.add_argument("dataset_name", nargs="?", help="Dataset name or dataset directory when using the original runner style.")
    parser.add_argument("--train_data", type=str, help="Path to a .npy or .csv training array.")
    parser.add_argument("--adjacency", type=str, help="Optional path to a .npy or .csv adjacency matrix.")
    parser.add_argument("--data_dir", type=str, help="Directory containing <dataset_name>/train.csv or equivalent.")
    parser.add_argument("--dataset_config", type=str, help="Optional dataset config JSON. Accepted for runner compatibility.")
    parser.add_argument("--model_type", type=str, choices=["bayesdag_linear", "bayesdag_nonlinear"], required=True)
    parser.add_argument("--model_config", type=str, required=True, help="Path to a BayesDAG JSON config.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--name", type=str, help="Optional experiment subdirectory name.")
    parser.add_argument("--seed", "--random_seed", dest="seed", type=int, default=0)
    parser.add_argument("--scale_noise", type=float)
    parser.add_argument("--scale_noise_p", type=float)
    parser.add_argument("--lambda_sparse", type=float)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--device", type=str, default="cpu", help="Accepted for compatibility; JAX device placement is environment-driven.")
    parser.add_argument("--causal_discovery", action="store_true")
    parser.add_argument("--run_inference", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _load_inputs(args: argparse.Namespace) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    if args.train_data:
        train_data = _load_array(args.train_data)
        adjacency = _load_array(args.adjacency) if args.adjacency else None
        return train_data, None, adjacency, None

    if args.dataset_name is None:
        raise ValueError("Either positional dataset_name or --train_data must be provided.")

    dataset_root = _resolve_dataset_root(args.dataset_name, args.data_dir)
    train_data, test_data, adjacency = _load_dataset_from_directory(dataset_root)
    return train_data, test_data, adjacency, dataset_root


def _save_outputs(output_dir: str, model, dataset_name: Optional[str], dataset_root: Optional[str], model_config_path: str):
    adj_samples, is_dag = model.get_adj_matrix(samples=100)
    adj_mean = adj_samples.mean(axis=0)
    adj_round = (adj_mean >= 0.5).astype(np.float32)

    np.save(os.path.join(output_dir, "adj_matrix_samples.npy"), adj_samples)
    np.save(os.path.join(output_dir, "is_dag.npy"), is_dag)
    np.savetxt(os.path.join(output_dir, "adj_matrix_predicted.csv"), adj_mean, delimiter=",", fmt="%.8f")
    np.savetxt(os.path.join(output_dir, "adj_matrix_predicted_round.csv"), adj_round, delimiter=",", fmt="%d")

    weighted_adj, _params, _buffers = model.get_weighted_adj_matrix(samples=min(100, adj_samples.shape[0]))
    np.save(os.path.join(output_dir, "weighted_adj_matrix_samples.npy"), weighted_adj)

    metadata = {
        "dataset_name": dataset_name,
        "dataset_root": dataset_root,
        "model_config_path": model_config_path,
        "num_adj_samples": int(adj_samples.shape[0]),
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main(argv=None):
    args = get_parser().parse_args(argv)
    del args.device
    del args.causal_discovery
    del args.run_inference
    del args.quiet

    output_dir = args.output_dir if args.name is None else os.path.join(args.output_dir, args.name)
    os.makedirs(output_dir, exist_ok=True)

    train_data, test_data, adjacency, dataset_root = _load_inputs(args)
    del test_data
    _dataset_config = _load_json(_resolve_config_path(args.dataset_config)) if args.dataset_config else _load_json(_default_dataset_config_path())
    del _dataset_config

    model_config_path = _resolve_config_path(args.model_config)
    config = _apply_overrides(_load_json(model_config_path), args)
    model = train_from_config_dict(
        train_data,
        model_type=args.model_type,
        model_config_dict=config,
        save_dir=output_dir,
        adjacency=adjacency,
        seed=args.seed,
    )
    _save_outputs(output_dir, model, args.dataset_name, dataset_root, model_config_path)


if __name__ == "__main__":
    main()
