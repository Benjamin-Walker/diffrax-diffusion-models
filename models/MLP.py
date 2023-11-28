import math
from typing import Tuple

import equinox as eqx
import jax.numpy as jnp


class MLP(eqx.Module):
    mlp: eqx.nn.MLP
    data_shape: Tuple[int]
    t1: float

    def __init__(self, data_shape, width_size, depth, t1, langevin, *, key):
        data_size = math.prod(data_shape)
        if langevin:
            in_size = 2 * data_size + 1
        else:
            in_size = data_size + 1
        self.mlp = eqx.nn.MLP(in_size, data_size, width_size, depth, key=key)
        self.data_shape = data_shape
        self.t1 = t1

    def __call__(self, t, y, v=None, *, key=None):
        del key
        t = 2 * (1 - (1 - t / self.t1) ** 4) - 1
        y = y.reshape(-1)
        if v is None:
            in_ = jnp.append(y, t)
        else:
            v = v.reshape(-1)
            in_ = jnp.concatenate([t[None], y, v])
        out = self.mlp(in_)
        return out.reshape(self.data_shape)
