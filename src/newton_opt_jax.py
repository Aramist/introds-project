import typing as tp
from collections.abc import Callable

import jax
import numpy as np
import optax
from flax import nnx
from jax import numpy as jnp


def compute_hessian(
    module: nnx.Module,
    inputs: jax.Array,
    targets: jax.Array,
    loss_fn: Callable[[jax.Array, jax.Array], jax.Array],
) -> dict[str, dict[str, jax.Array]]:
    """Evaluates the Hessian of the neural network w.r.t. its parameters at the input.

    Args:
        module (nnx.Module): Neural network
        inputs (jax.Array): Input data
        targets (jax.Array): Labels
        loss_fn (Callable[[jax.Array, jax.Array], jax.Array]): Loss function. Should be of
            the form loss_fn(predictions, targets)

    Returns:
        dict[str, dict[str, jax.Array]]: Nested dictionary containing the Hessian matrices for each parameter pair.
    """

    def wrapper(params: nnx.State, x: jax.Array, y: jax.Array) -> jax.Array:
        nnx.update(module, params)
        preds = module(x)
        return loss_fn(preds, y)

    hessian_fn = jax.hessian(wrapper, argnums=0)
    _, params, _ = nnx.split(module, nnx.Param, ...)
    return hessian_fn(params, inputs, targets)


class MLP(nnx.Module):
    d_in: int
    d_out: int
    n_hidden: int
    n_layers: int
    layers: nnx.List[nnx.Module]
    rngs: nnx.Rngs

    def __init__(
        self,
        num_layers: int,
        d_in: int,
        d_hidden: int,
        d_out: int,
        *,
        use_bias: bool = True,
        rngs: nnx.Rngs | None = None,
    ):
        """A simple ReLU MLP with the given number of hidden layers.

        Args:
            num_layers (int): Number of hidden layers. If 0, the model is linear.
            d_in (int): Dimensionality of input data.
            d_hidden (int): Width of hidden layers.
            d_out (int): Dimensionality of output.
            use_bias (bool, optional): Whether to use bias in linear layers. Defaults to True.
            rngs (jax.random.PRNGKey, optional): Random key for initialization. Defaults to None.
        """
        super().__init__()

        if rngs is None:
            self.rngs = nnx.Rngs(
                default=jax.random.PRNGKey(0),
                params=jax.random.PRNGKey(1),
                dropout=jax.random.PRNGKey(2),
            )
        else:
            self.rngs = rngs

        self.layers = nnx.List()
        channels = [d_in] + [d_hidden] * num_layers + [d_out]

        for in_dim, out_dim in zip(channels[:-1], channels[1:]):
            self.layers.append(
                nnx.Linear(in_dim, out_dim, use_bias=use_bias, rngs=self.rngs)
            )
            self.layers.append(nnx.relu)
        # drop last relu
        self.layers.pop()

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    dataset_size = 20_000
    d_data = 10
    rng = jax.random.PRNGKey(0)

    model = MLP(
        num_layers=3,
        d_in=d_data,
        d_hidden=64,
        d_out=1,
    )

    def mse_loss(predictions: jax.Array, targets: jax.Array) -> jax.Array:
        # predictions: (*batch, 1)
        # targets: (*batch,)
        # returns scalar

        return jnp.mean((predictions.squeeze() - targets) ** 2)

    # Generate dummy training data
    X = jax.random.normal(rng, (dataset_size, d_data))
    y = jnp.einsum(
        "...i, i -> ...", X, jnp.linspace(-1, 1, d_data)
    ) + 0.1 * jax.random.normal(rng, (dataset_size,))

    print(model(X))
    print(compute_hessian(model, X[:100], y[:100], mse_loss))
