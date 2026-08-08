import networkx as nx
import numpy as np
import jax.numpy as jnp

from src.jax_utils import ArrayDataLoader

try:
    from cdt.data import load_dataset
except ImportError:  # pragma: no cover - depends on local environment
    load_dataset = None


synthetic_datasets = {
    "data_p100_e100_n1000_GP",
    "data_p100_e400_n1000_GP",
    "data_p10_e10_n1000_GP",
    "data_p10_e40_n1000_GP",
    "data_p20_e20_n1000_GP",
    "data_p20_e80_n1000_GP",
    "data_p50_e200_n1000_GP",
    "data_p50_e50_n1000_GP",
    "data_sf_p100_e100_n1000_GP",
    "data_sf_p100_e400_n1000_GP",
    "data_sf_p10_e10_n1000_GP",
    "data_sf_p10_e40_n1000_GP",
    "data_sf_p20_e20_n1000_GP",
    "data_sf_p20_e80_n1000_GP",
    "data_sf_p50_e200_n1000_GP",
    "data_sf_p50_e50_n1000_GP",
}


def _make_splits(X, dag_adj, *, batch_size, split, seed):
    n_data = X.shape[0]
    indices = np.arange(n_data, dtype=np.int32)
    split0 = int(n_data * split[0])
    split1 = int(n_data * (split[0] + split[1]))
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_indices = indices[:split0]
    mean = np.mean(X[train_indices], axis=0, keepdims=True)
    std = np.std(X[train_indices], axis=0, keepdims=True, ddof=1)
    std[std == 0] = 1.0
    X_scaled = (X - mean) / std

    train_loader = ArrayDataLoader(
        X_scaled,
        indices=train_indices,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_loader = ArrayDataLoader(
        X_scaled,
        indices=indices[split0:split1],
        batch_size=1024,
        shuffle=False,
        seed=seed + 1,
    )
    test_loader = ArrayDataLoader(
        X_scaled,
        indices=indices[split1:],
        batch_size=1024,
        shuffle=False,
        seed=seed + 2,
    )

    return train_loader, val_loader, test_loader, jnp.asarray(dag_adj, dtype=jnp.float32)


def get_dag_dataset(dataset_directory, dataset_name, i_dataset, batch_size, split=(0.6, 0.2, 0.2), seed=1):
    assert np.sum(split) == 1.0

    if dataset_name in synthetic_datasets:
        return DAGDataset.synthetic(dataset_directory, dataset_name, i_dataset, batch_size=batch_size, split=split, seed=seed)
    if dataset_name == "sachs":
        return DAGDataset.sachs(batch_size=batch_size, split=split, seed=seed)
    if dataset_name == "syntren":
        return DAGDataset.syntren(dataset_directory, i_dataset, batch_size=batch_size, split=split, seed=seed)
    raise NotImplementedError


class DAGDataset:
    @classmethod
    def synthetic(cls, dataset_directory, dataset_name, i_dataset, batch_size, split, seed):
        adjacency = np.load(f"{dataset_directory}/{dataset_name}/DAG{i_dataset}.npy")
        X = np.load(f"{dataset_directory}/{dataset_name}/data{i_dataset}.npy")
        return _make_splits(X, adjacency, batch_size=batch_size, split=split, seed=seed)

    @classmethod
    def sachs(cls, batch_size, split, seed):
        if load_dataset is None:
            raise ImportError("cdt is required to load the Sachs dataset, matching the original implementation.")

        data, graph = load_dataset("sachs")
        adjacency = nx.to_numpy_array(graph)
        X = data.values[:853]
        return _make_splits(X, adjacency, batch_size=batch_size, split=split, seed=seed)

    @classmethod
    def syntren(cls, dataset_directory, i_dataset, batch_size, split, seed):
        adjacency = np.load(f"{dataset_directory}/syntren/DAG{i_dataset}.npy")
        X = np.load(f"{dataset_directory}/syntren/data{i_dataset}.npy")
        return _make_splits(X, adjacency, batch_size=batch_size, split=split, seed=seed)

