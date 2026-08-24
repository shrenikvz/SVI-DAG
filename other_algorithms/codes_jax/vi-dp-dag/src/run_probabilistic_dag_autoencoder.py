import logging
import os
import shutil
import time

from src.datasets.DAGDataset import get_dag_dataset
from src.jax_utils import load_checkpoint
from src.probabilistic_dag_model.probabilistic_dag_autoencoder import ProbabilisticDAGAutoencoder
from src.probabilistic_dag_model.test_probabilistic_dag_autoencoder import test_autoencoder
from src.probabilistic_dag_model.train_probabilistic_dag_autoencoder import train_autoencoder

try:  # pragma: no cover - optional experiment harness
    from sacred import Experiment
    import seml
    from seml.utils import flatten as seml_flatten
except ImportError:  # pragma: no cover
    Experiment = None
    seml = None
    seml_flatten = None


def _flatten_dict(d):
    if seml_flatten is not None:
        return seml_flatten(d)

    flat = {}
    for key, value in d.items():
        if isinstance(value, dict):
            nested = _flatten_dict(value)
            for nested_key, nested_value in nested.items():
                flat[f"{key}.{nested_key}"] = nested_value
        else:
            flat[key] = value
    return flat


def run(
    seed_dataset,
    dataset_name,
    dataset_directory,
    i_dataset,
    split,
    seed_model,
    directory_model,
    ma_hidden_dims,
    ma_architecture,
    ma_fast,
    pd_initial_adj,
    pd_temperature,
    pd_hard,
    pd_order_type,
    pd_noise_factor,
    directory_results,
    max_epochs,
    patience,
    frequency,
    batch_size,
    ma_lr,
    pd_lr,
    loss,
    regr,
    prior_p,
):
    logging.info("Received the following configuration:")
    logging.info(
        f"DATASET | seed_dataset {seed_dataset} - dataset_name {dataset_name} - i_dataset {i_dataset} - split {split}"
    )
    logging.info(
        "ARCHITECTURE | "
        f"seed_model {seed_model} - ma_hidden_dims {ma_hidden_dims} - ma_architecture {ma_architecture} - "
        f"ma_fast {ma_fast} - pd_initial_adj {pd_initial_adj} - pd_temperature {pd_temperature} - "
        f"pd_hard {pd_hard} - pd_order_type {pd_order_type} - pd_noise_factor {pd_noise_factor}"
    )
    logging.info(
        "TRAINING | "
        f"max_epochs {max_epochs} - patience {patience} - frequency {frequency} - batch_size {batch_size} - "
        f"ma_lr {ma_lr} - pd_lr {pd_lr} - loss {loss} - regr {regr} - prior_p {prior_p}"
    )

    train_loader, val_loader, test_loader, true_dag_adj = get_dag_dataset(
        dataset_directory=dataset_directory,
        dataset_name=dataset_name,
        i_dataset=i_dataset,
        batch_size=batch_size,
        split=split,
        seed=seed_dataset,
    )
    input_dim = true_dag_adj.shape[0]

    if pd_initial_adj == "GT":
        pd_initial_adjacency = true_dag_adj
    elif pd_initial_adj == "RGT":
        pd_initial_adjacency = true_dag_adj.T
    elif pd_initial_adj == "Learned":
        pd_initial_adjacency = None
    else:
        raise NotImplementedError

    model = ProbabilisticDAGAutoencoder(
        input_dim=input_dim,
        output_dim=1,
        loss=loss,
        regr=regr,
        prior_p=prior_p,
        seed=seed_model,
        ma_hidden_dims=ma_hidden_dims,
        ma_architecture=ma_architecture,
        ma_lr=ma_lr,
        ma_fast=ma_fast,
        pd_temperature=pd_temperature,
        pd_hard=pd_hard,
        pd_order_type=pd_order_type,
        pd_noise_factor=pd_noise_factor,
        pd_initial_adj=pd_initial_adjacency,
        pd_lr=pd_lr,
    )

    full_config_dict = _flatten_dict(
        {
            "seed_dataset": seed_dataset,
            "dataset_name": dataset_name,
            "i_dataset": i_dataset,
            "split": list(split),
            "seed_model": seed_model,
            "ma_hidden_dims": list(ma_hidden_dims),
            "ma_architecture": ma_architecture,
            "ma_fast": ma_fast,
            "pd_initial_adj": pd_initial_adj,
            "pd_temperature": pd_temperature,
            "pd_hard": pd_hard,
            "pd_order_type": pd_order_type,
            "pd_noise_factor": pd_noise_factor,
            "max_epochs": max_epochs,
            "patience": patience,
            "frequency": frequency,
            "batch_size": batch_size,
            "ma_lr": ma_lr,
            "pd_lr": pd_lr,
            "loss": loss,
            "regr": regr,
            "prior_p": prior_p,
        }
    )
    full_config_name = "-".join(str(v) for v in full_config_dict.values())
    model_path = f"{directory_model}/model-autopdag-{full_config_name}"

    t_start = time.time()
    train_losses, val_losses, train_mse, val_mse = train_autoencoder(
        model,
        train_loader,
        val_loader,
        max_epochs=max_epochs,
        frequency=frequency,
        patience=patience,
        model_path=model_path,
        full_config_dict=full_config_dict,
    )
    t_end = time.time()

    result_path = f"{directory_results}/results-autopdag-{full_config_name}"
    os.makedirs(result_path, exist_ok=True)
    checkpoint = load_checkpoint(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = test_autoencoder(model, true_dag_adj, train_loader, test_loader, result_path, seed_dataset)
    shutil.rmtree(result_path)

    results = {
        "model_path": model_path,
        "result_path": result_path,
        "training_time": t_end - t_start,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "fail_trace": getattr(getattr(seml, "evaluation", None), "get_results", None),
    }
    return {**results, **metrics}


if Experiment is not None:  # pragma: no cover - optional experiment harness
    ex = Experiment()
    seml.setup_logger(ex)

    @ex.post_run_hook
    def collect_stats(_run):
        seml.collect_exp_stats(_run)

    @ex.config
    def config():
        overwrite = None
        db_collection = None
        if db_collection is not None:
            ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

    @ex.automain
    def _run(**kwargs):
        return run(**kwargs)

