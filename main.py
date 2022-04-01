import jax.random as jrandom
from diffrax import diffeqsolve, ODETerm, Tsit5, NoAdjoint
from jax import vmap
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import equinox as eqx
import tensorflow_datasets as tfds
from UNet import Unet, LinearTimeSelfAttention
import tensorflow as tf

key = jrandom.PRNGKey(5677)

tf.config.experimental.set_visible_devices([], 'GPU')


def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    return train_ds, test_ds


t0, t1 = 0., 10.

def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)


def get_drift():

    def drift(t, y, args):
        return -0.5*y

    return drift


def marginal_prob(x, t):
    log_mean_coeff = -0.5*t
    mean = jnp.exp(log_mean_coeff) * x
    std = jnp.sqrt(jnp.maximum(1 - jnp.exp(2. * log_mean_coeff), 1e-5))
    return mean, std


batch_marginal_prob = vmap(marginal_prob, in_axes=(None, None, 0, 0), out_axes=0)


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
    losses = jnp.square(pred_z + batch_mul(z, 1/std))
    weight = (1 - jnp.exp(-t))
    losses = weight * jnp.mean(losses.reshape((losses.shape[0], -1)), axis=-1)
    loss = jnp.mean(losses)

    return loss


@eqx.filter_jit
def make_step(model, opt_state, data, step_key):

    loss, grads = loss_fn(model, data, step_key)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return model, loss


train_ds, test_ds = get_datasets()
data_mean = jnp.mean(train_ds['image'])
data_std = jnp.std(train_ds['image'])
data_max = jnp.max(train_ds['image'])
data_min = jnp.min(train_ds['image'])
train_ds['image'] = (train_ds['image'] - data_mean) / data_std
key, init_key = jrandom.split(key)

drift = get_drift()
learning_rate = 1e-4

model = Unet(
        key=init_key,
        input_size=(1, 28, 28),
        dim_mults=(1, 2, 4, ),
        attention_cls=LinearTimeSelfAttention,
        )

optim = optax.adam(learning_rate)
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
key, loader_key = jax.random.split(key)
iter_data = dataloader((train_ds['image'].transpose(0, 3, 1, 2),
                        jnp.expand_dims(train_ds['label'], 1)), 256,
                        key=loader_key)

epochs = 20
steps_per_epoch = 1000

for epoch in range(epochs):
    running_loss = 0
    for step, (x, y) in zip(range(steps_per_epoch), iter_data):
        key, step_key = jax.random.split(key)
        model, loss = make_step(model, opt_state, x, step_key)
        running_loss += loss.item()

    print(f"epoch={epoch}, loss={running_loss / steps_per_epoch}")


def vector_field(t, y, args):
    return drift(t, y + model(t, y), args)


term = ODETerm(vector_field)
solver = Tsit5()
key, unif_key = jax.random.split(key)
sol = diffeqsolve(term, solver, t0=t1, t1=t0, dt0=-0.1, y0=jrandom.normal(unif_key, (1, 28, 28)), adjoint=NoAdjoint())
sample = data_mean + data_std * sol.ys[0].squeeze()
sample = jnp.clip(sample, data_min, data_max)
plt.imshow(sample, cmap="Greys")
plt.axis("off")
plt.tight_layout()
plt.savefig('my_sample.png')
