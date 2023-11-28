import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import optax
from diffrax import diffeqsolve, DirectAdjoint, ODETerm, Tsit5
from jax import vmap

from datasets import diamond
from models.MLP import MLP
from models.UNet import UNet


def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)


def int_beta(t):
    return 0.1 * t + (19.9 / 2) * (t**2)


def get_drift():
    def drift(t, y, args):
        _, beta = jax.jvp(int_beta, (t,), (jnp.ones_like(t),))
        return -0.5 * beta * y

    return drift


def marginal_prob(x, t):
    log_mean_coeff = -0.5 * int_beta(t)
    mean = jnp.exp(log_mean_coeff) * x
    std = jnp.sqrt(jnp.maximum(1 - jnp.exp(2.0 * log_mean_coeff), 1e-5))
    return mean, std


batch_marginal_prob = vmap(marginal_prob, in_axes=(0, 0), out_axes=0)


def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


@eqx.filter_jit
@eqx.filter_value_and_grad
def loss_fn(model, data, key):
    key, step_key = jrandom.split(key)
    batch_size = data.shape[0]
    t = jrandom.uniform(step_key, (batch_size,), minval=0, maxval=t1 / batch_size)
    t = t + (t1 / batch_size) * jnp.arange(batch_size)
    key, step_key = jrandom.split(key)
    z = jrandom.normal(step_key, data.shape)
    mean, std = batch_marginal_prob(data, t)
    std = jnp.expand_dims(std, 1)
    perturbed_data = mean + batch_mul(std, z)
    pred_z = jax.vmap(model)(t, perturbed_data)
    losses = jnp.square(pred_z + batch_mul(z, 1 / std))
    weight = 1 - jnp.exp(-t)
    losses = weight * jnp.mean(losses.reshape((losses.shape[0], -1)), axis=-1)
    loss = jnp.mean(losses)

    return loss


@eqx.filter_jit
def make_step(model, opt_state, data, step_key):
    loss, grads = loss_fn(model, data, step_key)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return model, loss


key = jrandom.PRNGKey(5677)
t0, t1 = 0.0, 1.0
key, init_key = jrandom.split(key)
drift = get_drift()
key, loader_key = jax.random.split(key)
dataset = diamond(loader_key)

Use_UNet = False
if Use_UNet:
    model = UNet(
        key=init_key,
        data_shape=dataset.data_shape,
        is_biggan=False,
        dim_mults=[1, 2, 4],
        hidden_size=64,
        heads=4,
        dim_head=32,
        dropout_rate=0.0,
        num_res_blocks=2,
        attn_resolutions=[16],
        t1=t1,
        langevin=False,
    )
else:
    model = MLP(
        key=init_key,
        data_shape=dataset.data_shape,
        width_size=128,
        depth=3,
        t1=t1,
        langevin=False,
    )

learning_rate = 1e-4

optim = optax.adam(learning_rate)
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

epochs = 1000
steps_per_epoch = 100

for epoch in range(epochs):
    running_loss = 0
    for step, data in zip(range(steps_per_epoch), dataset.train_dataloader.loop(128)):
        key, step_key = jax.random.split(key)
        model, loss = make_step(model, opt_state, data, step_key)
        running_loss += loss.item()

    print(f"epoch={epoch}, loss={running_loss / steps_per_epoch}")

    if epoch % 100 == 0:

        def vector_field(t, y, args):
            _, beta = jax.jvp(int_beta, (t,), (jnp.ones_like(t),))
            return drift(t, y, args) - 0.5 * beta * model(t, y)

        term = ODETerm(vector_field)
        solver = Tsit5()
        key, unif_key = jax.random.split(key)
        plt.figure()
        if Use_UNet:
            sol = diffeqsolve(
                term,
                solver,
                t0=t1,
                t1=t0,
                dt0=-0.1,
                y0=jrandom.normal(unif_key, dataset.data_shape),
                adjoint=DirectAdjoint(),
            )
            sample = dataset.mean + dataset.std * sol.ys[0].squeeze()
            sample = jnp.clip(sample, dataset.min, dataset.max)
            plt.imshow(sample, cmap="Greys")
            plt.axis("off")
            plt.tight_layout()
        else:
            unif_key, key = jrandom.split(key)
            y0s = jrandom.normal(unif_key, (1000,) + dataset.data_shape)
            sol = jax.vmap(
                lambda y: diffeqsolve(
                    term, solver, t0=t1, t1=t0, dt0=-0.1, y0=y, adjoint=DirectAdjoint()
                )
            )(y0s)
            sample = dataset.mean + dataset.std * sol.ys[:, 0]
            sample = jnp.clip(sample, dataset.min, dataset.max)
            plt.scatter(sample[:, 0], sample[:, 1])
        plt.savefig(f"Samples/Diamond_Epoch_{epoch}.png")
        plt.close()
