import numpy as np
import jax
import jax.numpy as jnp

from src.probabilistic_dag_model.masked_autoencoder import MaskedAutoencoder, MaskedAutoencoderFast
from src.probabilistic_dag_model.probabilistic_dag import ProbabilisticDAG


masked_autoencoder = {True: MaskedAutoencoderFast, False: MaskedAutoencoder}


class ProbabilisticDAGAutoencoder:
    def __init__(
        self,
        input_dim,
        output_dim,
        loss="ELBO",
        regr=0,
        prior_p=0.001,
        seed=123,
        ma_hidden_dims=(64, 64, 64),
        ma_architecture="linear",
        ma_lr=1e-3,
        ma_fast=False,
        pd_temperature=1.0,
        pd_hard=True,
        pd_order_type="sinkhorn",
        pd_noise_factor=1.0,
        pd_initial_adj=None,
        pd_lr=1e-3,
    ):
        np.random.seed(seed)

        self.loss = loss
        self.regr = float(regr)
        self.prior_p = float(prior_p)
        self.training = True
        self.grad_loss = None
        self.pd_initial_adj = None if pd_initial_adj is None else jnp.asarray(pd_initial_adj, dtype=jnp.float32)
        self._last_batch = None
        self._last_mask_type = None
        self._last_threshold = 0.5
        self._last_mask_key = None

        self.mask_autoencoder = masked_autoencoder[ma_fast](
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=ma_hidden_dims,
            architecture=ma_architecture,
            lr=ma_lr,
            seed=seed,
        )

        if pd_order_type not in {"sinkhorn", "topk"}:
            raise NotImplementedError

        self.probabilistic_dag = ProbabilisticDAG(
            n_nodes=input_dim,
            temperature=pd_temperature,
            hard=pd_hard,
            order_type=pd_order_type,
            noise_factor=pd_noise_factor,
            initial_adj=pd_initial_adj,
            lr=pd_lr,
            seed=seed,
        )

    def to(self, *_args, **_kwargs):
        return self

    def train(self):
        self.training = True
        self.mask_autoencoder.train()
        self.probabilistic_dag.train()
        return self

    def eval(self):
        self.training = False
        self.mask_autoencoder.eval()
        self.probabilistic_dag.eval()
        return self

    def state_dict(self):
        return {
            "mask_autoencoder": self.mask_autoencoder.state_dict(),
            "probabilistic_dag": self.probabilistic_dag.state_dict(),
            "pd_initial_adj": self.pd_initial_adj,
        }

    def load_state_dict(self, state_dict):
        self.mask_autoencoder.load_state_dict(state_dict["mask_autoencoder"])
        self.probabilistic_dag.load_state_dict(state_dict["probabilistic_dag"])
        self.pd_initial_adj = None if state_dict.get("pd_initial_adj") is None else jnp.asarray(state_dict["pd_initial_adj"])
        return self

    def _elbo_loss_with_params(self, X_pred, X, pd_params):
        elbo_loss = jnp.mean((X_pred - X) ** 2)
        if self.regr > 0:
            target = self.prior_p * jnp.ones_like(pd_params["edge_log_params"])
            regularizer = jnp.mean(target * (jnp.log(target) - pd_params["edge_log_params"]))
            elbo_loss = elbo_loss + self.regr * regularizer
        return elbo_loss

    def _current_mask(self, pd_params):
        if self.pd_initial_adj is not None:
            return self.pd_initial_adj
        if self._last_mask_type == "deterministic":
            return self.probabilistic_dag.get_threshold_mask(self._last_threshold, params=pd_params)
        if self._last_mask_type == "id":
            return jnp.eye(self.mask_autoencoder.input_dim, dtype=jnp.float32)
        return self.probabilistic_dag.sample(params=pd_params, key=self._last_mask_key)

    def forward(self, X, compute_loss=True):
        X = jnp.asarray(X, dtype=jnp.float32)
        self._last_batch = X
        X_pred = self.mask_autoencoder.apply(X)
        if compute_loss:
            self.grad_loss = self._elbo_loss_with_params(X_pred, X, self.probabilistic_dag.params)
        return X_pred

    __call__ = forward

    def update_mask(self, type=None, threshold=0.5):
        self._last_mask_type = type
        self._last_threshold = threshold
        self._last_mask_key = None

        if type == "deterministic":
            new_mask = self.probabilistic_dag.get_threshold_mask(threshold)
        elif type == "id":
            new_mask = jnp.eye(self.mask_autoencoder.input_dim, dtype=jnp.float32)
        else:
            self._last_mask_key = self.probabilistic_dag.next_key()
            new_mask = self.probabilistic_dag.sample(key=self._last_mask_key)

        if self.pd_initial_adj is not None:
            new_mask = self.pd_initial_adj
        self.mask_autoencoder.mask = new_mask

    def ELBO_loss(self, X_pred, X):
        return self._elbo_loss_with_params(X_pred, X, self.probabilistic_dag.params)

    def step(self):
        if self._last_batch is None:
            raise RuntimeError("forward must be called before step, matching the original training flow.")

        if self.pd_initial_adj is None:
            def loss_fn(ma_params, pd_params):
                mask = self._current_mask(pd_params)
                X_pred = self.mask_autoencoder.apply(self._last_batch, params=ma_params, mask=mask)
                return self._elbo_loss_with_params(X_pred, self._last_batch, pd_params)

            loss_value, (ma_grads, pd_grads) = jax.value_and_grad(loss_fn, argnums=(0, 1))(
                self.mask_autoencoder.params,
                self.probabilistic_dag.params,
            )
            self.mask_autoencoder.apply_gradients(ma_grads)
            self.probabilistic_dag.apply_gradients(pd_grads)
        else:
            def loss_fn(ma_params):
                X_pred = self.mask_autoencoder.apply(self._last_batch, params=ma_params, mask=self.pd_initial_adj)
                return self._elbo_loss_with_params(X_pred, self._last_batch, self.probabilistic_dag.params)

            loss_value, ma_grads = jax.value_and_grad(loss_fn)(self.mask_autoencoder.params)
            self.mask_autoencoder.apply_gradients(ma_grads)

        self.grad_loss = loss_value

