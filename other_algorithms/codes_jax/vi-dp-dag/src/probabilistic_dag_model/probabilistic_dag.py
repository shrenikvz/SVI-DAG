import numpy as np
import jax
import jax.nn as jnn
import jax.numpy as jnp

from src.jax_utils import adam_init, adam_update
from src.probabilistic_dag_model.soft_sort import SoftSort_p1, gumbel_sinkhorn


def _sample_binary_gumbel_softmax(logits, *, key):
    gumbel = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape, minval=1e-20, maxval=1.0)))
    y_soft = jnn.softmax(logits + gumbel, axis=0)
    indices = jnp.argmax(y_soft, axis=0)
    y_hard = jnp.moveaxis(jnn.one_hot(indices, logits.shape[0], dtype=y_soft.dtype), -1, 0)
    return jax.lax.stop_gradient(y_hard - y_soft) + y_soft


class ProbabilisticDAG:
    def __init__(
        self,
        n_nodes,
        temperature=1.0,
        hard=True,
        order_type="sinkhorn",
        noise_factor=1.0,
        initial_adj=None,
        lr=1e-3,
        seed=0,
    ):
        np.random.seed(seed)

        self.n_nodes = int(n_nodes)
        self.temperature = float(temperature)
        self.hard = bool(hard)
        self.order_type = order_type
        self.noise_factor = float(noise_factor)
        self.lr = float(lr)
        self.training = True
        self.rng_key = jax.random.PRNGKey(seed)
        self.mask = jnp.triu(jnp.ones((self.n_nodes, self.n_nodes), dtype=jnp.float32), 1)

        if self.order_type == "sinkhorn":
            perm_weights = jnp.zeros((self.n_nodes, self.n_nodes), dtype=jnp.float32)
            self.sort = None
        elif self.order_type == "topk":
            perm_weights = jnp.zeros((self.n_nodes,), dtype=jnp.float32)
            self.sort = SoftSort_p1(hard=self.hard, tau=self.temperature)
        else:
            raise NotImplementedError

        self.rng_key, edge_key = jax.random.split(self.rng_key)
        edge_log_params = jax.random.uniform(edge_key, (self.n_nodes, self.n_nodes), dtype=jnp.float32)

        self.initial_adj = None if initial_adj is None else jnp.asarray(initial_adj, dtype=jnp.float32)
        if self.initial_adj is not None:
            zero_indices = (1.0 - self.initial_adj).astype(bool)
            edge_log_params = edge_log_params.at[zero_indices].set(-300.0)
        edge_log_params = edge_log_params.at[jnp.diag_indices(self.n_nodes)].set(-300.0)

        self.params = {
            "perm_weights": perm_weights,
            "edge_log_params": edge_log_params,
        }
        self.optimizer_state = adam_init(self.params)

    def to(self, *_args, **_kwargs):
        return self

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def state_dict(self):
        return {
            "params": self.params,
            "optimizer_state": self.optimizer_state,
            "rng_key": self.rng_key,
        }

    def load_state_dict(self, state_dict):
        self.params = jax.tree_util.tree_map(jnp.asarray, state_dict["params"])
        if "optimizer_state" in state_dict:
            self.optimizer_state = jax.tree_util.tree_map(jnp.asarray, state_dict["optimizer_state"])
        if "rng_key" in state_dict:
            self.rng_key = jnp.asarray(state_dict["rng_key"])
        return self

    def next_key(self):
        self.rng_key, subkey = jax.random.split(self.rng_key)
        return subkey

    def apply_gradients(self, grads):
        if self.initial_adj is not None:
            grads = dict(grads)
            grads["edge_log_params"] = grads["edge_log_params"] * self.initial_adj
        self.params, self.optimizer_state = adam_update(
            self.params,
            grads,
            self.optimizer_state,
            lr=self.lr,
        )

    def sample_edges(self, params=None, *, key):
        active_params = self.params if params is None else params
        p_log = jnn.log_sigmoid(jnp.stack((active_params["edge_log_params"], -active_params["edge_log_params"]), axis=0))
        dag = _sample_binary_gumbel_softmax(p_log, key=key)[0]
        return dag

    def sample_permutation(self, params=None, *, key):
        active_params = self.params if params is None else params
        if self.order_type == "sinkhorn":
            log_alpha = jnn.log_sigmoid(active_params["perm_weights"])
            permutation, _ = gumbel_sinkhorn(
                log_alpha,
                key=key,
                noise_factor=self.noise_factor,
                temp=self.temperature,
                hard=self.hard,
            )
            permutation = jnp.squeeze(permutation)
        elif self.order_type == "topk":
            logits = jnn.log_softmax(active_params["perm_weights"], axis=0)[None, :]
            gumbels = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape, minval=1e-20, maxval=1.0)))
            permutation = jnp.squeeze(self.sort((logits + gumbels) / 1.0))
        else:
            raise NotImplementedError
        return permutation

    def sample(self, params=None, *, key):
        active_params = self.params if params is None else params
        perm_key, edge_key = jax.random.split(key)
        permutation = self.sample_permutation(active_params, key=perm_key)
        permutation_inv = permutation.T
        dag_adj = self.sample_edges(active_params, key=edge_key)
        dag_adj = dag_adj * (permutation_inv @ self.mask @ permutation)
        return dag_adj

    def log_prob(self, dag_adj):
        raise NotImplementedError

    def deterministic_permutation(self, hard=True, params=None):
        active_params = self.params if params is None else params
        if self.order_type == "sinkhorn":
            log_alpha = jnn.log_sigmoid(active_params["perm_weights"])
            permutation, _ = gumbel_sinkhorn(
                log_alpha,
                temp=self.temperature,
                hard=hard,
                noise_factor=0,
            )
            permutation = jnp.squeeze(permutation)
        elif self.order_type == "topk":
            sort = SoftSort_p1(hard=hard, tau=self.temperature)
            permutation = jnp.squeeze(sort(active_params["perm_weights"].reshape(1, -1)))
        else:
            raise NotImplementedError
        return permutation

    def get_threshold_mask(self, threshold, params=None):
        active_params = self.params if params is None else params
        permutation = self.deterministic_permutation(params=active_params)
        permutation_inv = permutation.T
        dag = (jax.nn.sigmoid(active_params["edge_log_params"]) > threshold).astype(jnp.float32)
        return dag * (permutation_inv @ self.mask @ permutation)

    def get_prob_mask(self, params=None):
        active_params = self.params if params is None else params
        permutation = self.deterministic_permutation(params=active_params)
        permutation_inv = permutation.T
        edge_probs = jax.nn.sigmoid(active_params["edge_log_params"])
        return edge_probs * (permutation_inv @ self.mask @ permutation)

    def print_parameters(self, prob=True):
        print("Permutation Weights")
        print(np.asarray(jax.nn.sigmoid(self.params["perm_weights"]) if prob else self.params["perm_weights"]))
        print("Edge Probs")
        print(np.asarray(jax.nn.sigmoid(self.params["edge_log_params"]) if prob else self.params["edge_log_params"]))

