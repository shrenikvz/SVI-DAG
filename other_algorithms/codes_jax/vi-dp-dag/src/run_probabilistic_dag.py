import logging
import os
import time

from src.datasets.DAGDataset import get_dag_dataset
from src.jax_utils import load_checkpoint
from src.probabilistic_dag_model.probabilistic_dag import ProbabilisticDAG
from src.probabilistic_dag_model.test_dag import test
from src.probabilistic_dag_model.train_dag import train

try:  # pragma: no cover - optional experiment harness
    from sacred import Experiment
    import seml
except ImportError:  # pragma: no cover
    Experiment = None
    seml = None


def _resolve_dataset_directory(dataset_name):
    if dataset_name == "sachs":
        return ""

    env_directory = os.environ.get("VI_DP_DAG_DATASET_DIRECTORY")
    if env_directory:
        return env_directory

    candidates = [
        os.path.join(os.getcwd(), "data"),
        os.path.join(os.getcwd(), "datasets"),
        os.path.join(os.path.dirname(__file__), "..", "data"),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, dataset_name)):
            return candidate
    raise ValueError(
        "dataset_directory is required for non-Sachs datasets in run_probabilistic_dag.py; "
        "set VI_DP_DAG_DATASET_DIRECTORY to preserve the original runner interface."
    )


def run(
    seed_dataset,
    dataset_name,
    i_dataset,
    split,
    seed_model,
    directory_model,
    temperature,
    hard,
    order_type,
    noise_factor,
    directory_results,
    max_epochs,
    patience,
    frequency,
    batch_size,
    lr,
    loss,
    regr,
):
    logging.info("Received the following configuration:")
    logging.info(
        f"DATASET | seed_dataset {seed_dataset} - dataset_name {dataset_name} - i_dataset {i_dataset} - split {split}"
    )
    logging.info(
        f"ARCHITECTURE | seed_model {seed_model} - temperature {temperature} - hard {hard} - order_type {order_type} - noise_factor {noise_factor}"
    )
    logging.info(
        f"TRAINING | max_epochs {max_epochs} - patience {patience} - frequency {frequency} - batch_size {batch_size} - lr {lr}"
    )

    dataset_directory = _resolve_dataset_directory(dataset_name)
    train_loader, val_loader, test_loader, true_dag_adj = get_dag_dataset(
        dataset_directory=dataset_directory,
        dataset_name=dataset_name,
        i_dataset=i_dataset,
        batch_size=batch_size,
        split=split,
        seed=seed_dataset,
    )
    del train_loader, val_loader, test_loader
    input_dim = true_dag_adj.shape[0]

    if order_type not in {"sinkhorn", "topk"}:
        raise NotImplementedError

    model = ProbabilisticDAG(
        n_nodes=input_dim,
        temperature=temperature,
        hard=hard,
        order_type=order_type,
        noise_factor=noise_factor,
        initial_adj=None,
        lr=lr,
        seed=seed_model,
    )

    full_config_dict = {
        "seed_dataset": seed_dataset,
        "dataset_name": dataset_name,
        "i_dataset": i_dataset,
        "split": list(split),
        "seed_model": seed_model,
        "temperature": temperature,
        "hard": hard,
        "order_type": order_type,
        "noise_factor": noise_factor,
        "max_epochs": max_epochs,
        "patience": patience,
        "frequency": frequency,
        "batch_size": batch_size,
        "lr": lr,
        "loss": loss,
        "regr": regr,
    }
    full_config_name = "-".join(str(v) for v in full_config_dict.values())
    model_path = f"{directory_model}/model-pdag-{full_config_name}"

    t_start = time.time()
    model, prob_abs_losses, sampled_mse_losses = train(
        model,
        true_dag_adj,
        max_epochs=max_epochs,
        frequency=frequency,
        patience=patience,
        model_path=model_path,
        full_config_dict=full_config_dict,
    )
    t_end = time.time()

    checkpoint = load_checkpoint(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = test(model, true_dag_adj)

    results = {
        "model_path": model_path,
        "result_path": f"{directory_results}/results-pdag-{full_config_name}",
        "training_time": t_end - t_start,
        "prob_abs_losses": prob_abs_losses,
        "sampled_mse_losses": sampled_mse_losses,
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
