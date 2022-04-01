# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""U-Nets for continuous-time Denoising Diffusion Probabilistic Models.
This file only serves as a helper for `examples/cont_ddpm.py`.
To use this file, run the following to install extra requirements:
pip install einops
"""
import math

import jax
import jax.numpy as jnp
import equinox as eqx
import flax.linen as fnn
from einops import rearrange
from equinox.module import Module
from conv import ConvTranspose2d as CT2D
import equinox.nn as nn
from equinox.module import Module, static_field


def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)


class Mish(Module):
    def __call__(self, x):
        return _mish(x)


@jax.jit
def _mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


class SinusoidalPosEmb(Module):

    emb: float

    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.emb = jnp.exp(jnp.arange(half_dim) * -emb)

    def __call__(self, x):
        emb = x[:, None] * self.emb[None, :]
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb


class SelfAttention(Module):

    #group_norm: "fnn.normalization.GroupNorm"
    qkv: "eqx.nn.conv.Conv2d"
    out: "eqx.nn.conv.Conv2d"


    def __init__(self, dim, key, groups=32, **kwargs):
        super().__init__()
        key1, key2 = jax.random.split(key, 2)
        #self.group_norm = fnn.GroupNorm(groups, dim)
        self.qkv = eqx.nn.Conv2d(dim, dim * 3, kernel_size=1, key=key1)
        self.out = eqx.nn.Conv2d(dim, dim, kernel_size=1, stride=1, key=key2)

    def __call__(self, x):
        b, c, h, w = x.size()
        #x = self.group_norm(x)
        q, k, v = tuple(t.view(b, c, h * w) for t in self.qkv(x).chunk(chunks=3, dim=1))
        attn_matrix = (batch_mul(k.permute(0, 2, 1), q) / math.sqrt(c)).softmax(dim=-2)
        out = batch_mul(v, attn_matrix).view(b, c, h, w)
        return self.out(out)


class LinearTimeSelfAttention(Module):

    #group_norm: "fnn.normalization.GroupNorm"
    heads: int
    to_qkv: "eqx.nn.conv.Conv2d"
    to_out: "eqx.nn.conv.Conv2d"

    def __init__(self, dim, key, heads=4, dim_head=32, groups=32,):
        super().__init__()
        key1, key2 = jax.random.split(key, 2)
        #self.group_norm = fnn.GroupNorm(groups, dim)
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = eqx.nn.Conv2d(dim, hidden_dim * 3, 1, key=key1)
        self.to_out = eqx.nn.Conv2d(hidden_dim, dim, 1, key=key2)

    def __call__(self, x):
        b, c, h, w = x.shape
        #x = self.group_norm(x)
        x = x.squeeze()
        qkv = self.to_qkv(x)
        qkv = jnp.expand_dims(qkv, axis=0)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = jax.nn.softmax(k, axis=-1)
        context = jnp.einsum('bhdn,bhen->bhde', k, v)
        out = jnp.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        out = out.squeeze()
        return self.to_out(out)


class ResnetBlock(Module):

    dim: int
    dim_out: int
    groups: int
    dropout_rate: float
    time_emb_dim: int
    mlp_layers: list
    block1_layers: list
    block2_layers: list
    res_conv: "eqx.nn.conv.Conv2d"

    def __init__(self, dim, dim_out, *, time_emb_dim, key, groups=8, dropout_rate=0.):
        super().__init__()
        key1, key2, key3, key4 = jax.random.split(key, 4)
        # groups: Used in group norm.
        self.dim = dim
        self.dim_out = dim_out
        self.groups = groups
        self.dropout_rate = dropout_rate
        self.time_emb_dim = time_emb_dim

        self.mlp_layers = [Mish(),
                           eqx.nn.Linear(time_emb_dim, dim_out, key=key1)]

        # Norm -> non-linearity -> conv format follows
        # https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/nn.py#L55
        self.block1_layers = [#fnn.GroupNorm(groups, dim),
                              Mish(),
                              eqx.nn.Conv2d(dim, dim_out, 3, padding=1, key=key2)]
        self.block2_layers = [#fnn.GroupNorm(groups, dim_out),
                              Mish(),
                              eqx.nn.Conv2d(dim_out, dim_out, 3, padding=1, key=key3)]
        self.res_conv = eqx.nn.Conv2d(dim, dim_out, 1, key=key4)

    def __call__(self, x, t):
        h = x
        for layer in self.block1_layers:
            h = layer(h)
        for layer in self.mlp_layers:
            t = layer(t)
        h += t[..., None, None]
        for layer in self.block2_layers:
            h = layer(h)
        return h + self.res_conv(x)

    def __repr__(self):
        return (f"{self.__class__.__name__}(dim={self.dim}, dim_out={self.dim_out}, time_emb_dim="
                f"{self.time_emb_dim}, groups={self.groups}, dropout_rate={self.dropout_rate})")


class Residual(Module):

    fn: "LinearTimeSelfAttention"

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def __call__(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Downsample(Module):

    conv: "eqx.nn.conv.Conv2d"

    def __init__(self, dim, key):
        super().__init__()
        self.conv = eqx.nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, key=key)

    def __call__(self, x):
        return self.conv(x)


class Upsample(Module):

    conv: "conv.ConvTranspose2d"

    def __init__(self, dim, key):
        super().__init__()
        self.conv = CT2D(dim, dim, kernel_size=4, stride=2, padding=1, key=key)

    def __call__(self, x):
        return self.conv(x)


class Unet(Module):

    time_pos_emb: "SinusoidalPosEmb"
    mlp_layers: list
    first_conv: "eqx.nn.conv.Conv2d"
    down_res_blocks: list
    down_attn_blocks: list
    down_spatial_blocks: list
    mid_block1: "ResnetBlock"
    mid_attn: "Residual"
    mid_block2: "ResnetBlock"
    ups_res_blocks: list
    ups_attn_blocks: list
    ups_spatial_blocks: list
    final_conv_layers: list

    def __init__(self,
                 key,
                 input_size=(3, 32, 32),
                 hidden_channels=64,
                 dim_mults=(1, 2, 4, 8),
                 groups=32,
                 heads=4,
                 dim_head=32,
                 dropout_rate=0.,
                 num_res_blocks=2,
                 attn_resolutions=(16,),
                 attention_cls=SelfAttention):
        super().__init__()
        key1, key2, key3, key4, key5, key6, key7, key8, key9, key10, key11, \
            key12, key13, key14, key15, key16 = jax.random.split(key, 16)
        in_channels, in_height, in_width = input_size
        dims = [hidden_channels, *map(lambda m: hidden_channels * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(hidden_channels)
        self.mlp_layers = [eqx.nn.Linear(hidden_channels, hidden_channels * 4, key=key1),
                           Mish(),
                           eqx.nn.Linear(hidden_channels * 4, hidden_channels, key=key2)]

        self.first_conv = eqx.nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, key=key3)

        h, w = in_height, in_width
        self.down_res_blocks = []
        self.down_attn_blocks = []
        self.down_spatial_blocks = []
        for ind, (dim_in, dim_out) in enumerate(in_out):
            res_blocks = [
                [ResnetBlock(
                    dim=dim_in,
                    dim_out=dim_out,
                    time_emb_dim=hidden_channels,
                    groups=groups,
                    dropout_rate=dropout_rate,
                    key=key4
                )]
            ]
            for _ in range(num_res_blocks - 1):
                res_blocks.append([
                ResnetBlock(
                    dim=dim_out,
                    dim_out=dim_out,
                    time_emb_dim=hidden_channels,
                    groups=groups,
                    dropout_rate=dropout_rate,
                    key=key5
                )])
            self.down_res_blocks.append(res_blocks)

            attn_blocks = []
            if h in attn_resolutions and w in attn_resolutions:
                attn_blocks.append(
                    [Residual(attention_cls(dim_out, heads=heads, dim_head=dim_head, groups=groups, key=key6))
                     for _ in range(num_res_blocks)]
                )
            self.down_attn_blocks.append(attn_blocks)

            if ind < (len(in_out) - 1):
                spatial_blocks = [Downsample(dim_out, key=key7)]
                h, w = h // 2, w // 2
            else:
                spatial_blocks = []
            self.down_spatial_blocks.append(spatial_blocks)

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(
            dim=mid_dim,
            dim_out=mid_dim,
            time_emb_dim=hidden_channels,
            groups=groups,
            dropout_rate=dropout_rate,
            key=key8
        )
        self.mid_attn = Residual(attention_cls(mid_dim, heads=heads, dim_head=dim_head, groups=groups, key=key9))
        self.mid_block2 = ResnetBlock(
            dim=mid_dim,
            dim_out=mid_dim,
            time_emb_dim=hidden_channels,
            groups=groups,
            dropout_rate=dropout_rate,
            key=key10
        )

        self.ups_res_blocks = []
        self.ups_attn_blocks = []
        self.ups_spatial_blocks = []
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            res_blocks = []
            for _ in range(num_res_blocks):
                res_blocks.append([
                    ResnetBlock(
                        dim=dim_out * 2,
                        dim_out=dim_out,
                        time_emb_dim=hidden_channels,
                        groups=groups,
                        dropout_rate=dropout_rate,
                        key=key11
                    )
                ])
            res_blocks.append([
                ResnetBlock(
                    dim=dim_out + dim_in,
                    dim_out=dim_in,
                    time_emb_dim=hidden_channels,
                    groups=groups,
                    dropout_rate=dropout_rate,
                    key=key12
                )
            ])
            self.ups_res_blocks.append(res_blocks)

            attn_blocks = []
            if h in attn_resolutions and w in attn_resolutions:
                attn_blocks.append(
                    [Residual(attention_cls(dim_out, heads=heads, dim_head=dim_head, groups=groups, key=key13))
                     for _ in range(num_res_blocks)]
                )
                attn_blocks.append(
                    Residual(attention_cls(dim_in, heads=heads, dim_head=dim_head, groups=groups, key=key14))
                )
            self.ups_attn_blocks.append(attn_blocks)

            spatial_blocks = []
            if ind < (len(in_out) - 1):
                spatial_blocks.append(Upsample(dim_in, key=key15))
                h, w = h * 2, w * 2
            self.ups_spatial_blocks.append(spatial_blocks)

        self.final_conv_layers = [
            #fnn.GroupNorm(groups, hidden_channels),
            Mish(),
            eqx.nn.Conv2d(hidden_channels, in_channels, 1, key=key16)
        ]

    def __call__(self, t, x):
        t = t.reshape(1)
        t = self.time_pos_emb(t)
        t = t.squeeze()
        for layer in self.mlp_layers:
            t = layer(t)

        hs = [self.first_conv(x)]
        for i, (res_blocks, attn_blocks, spatial_blocks) in enumerate(
                zip(self.down_res_blocks, self.down_attn_blocks, self.down_spatial_blocks)):
            if len(attn_blocks) > 0:
                for res_block, attn_block in zip(res_blocks, attn_blocks):
                    for layer in res_block:
                        h = layer(hs[-1], t)
                    for layer in attn_block:
                        h = layer(h)
                    hs.append(h)
            else:
                for res_block in res_blocks:
                    for layer in res_block:
                        h = layer(hs[-1], t)
                    hs.append(h)
            if len(spatial_blocks) > 0:
                spatial_block, = spatial_blocks
                hs.append(spatial_block(hs[-1]))

        h = hs[-1]
        h = self.mid_block1(h, t)
        h = h.reshape(1, 256, 7, 7)
        h = self.mid_attn(h)
        h = h.reshape(256, 7, 7)
        h = self.mid_block2(h, t)

        for i, (res_blocks, attn_blocks, spatial_blocks) in enumerate(
                zip(self.ups_res_blocks, self.ups_attn_blocks, self.ups_spatial_blocks)):
            if len(attn_blocks) > 0:
                for res_block, attn_block in zip(res_blocks, attn_blocks):
                    for layer in res_block:
                        h = layer(jnp.concatenate((h, hs.pop()), axis=1), t)
                    for layer in attn_block:
                        h = layer(h)
            else:
                for res_block in res_blocks:
                    for layer in res_block:
                        h = layer(jnp.concatenate((h, hs.pop()), axis=0), t)
            if len(spatial_blocks) > 0:
                spatial_block, = spatial_blocks
                h = spatial_block(h)

        for layer in self.final_conv_layers:
            h = layer(h)

        return h


class SNN(eqx.Module):
    """A simple NN model."""
    layers: list

    def __init__(self, key):
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)
        self.layers = [eqx.nn.Linear(784, 2048, key=key1),
                       eqx.nn.Linear(2048, 1024, key=key2),
                       eqx.nn.Linear(1024, 512, key=key3),
                       eqx.nn.Linear(512, 256, key=key4),
                       eqx.nn.Linear(256, 784, key=key5)]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return jax.nn.sigmoid(self.layers[-1](x))


class ConvTranspose2D(Module):

    in_c: int = static_field()
    out_c: int = static_field()
    kernel: int = static_field()
    stride: int = static_field()
    padding: int = static_field()
    key: jax.random.PRNGKey = static_field()

    def __init__(self, in_c, out_c, kernel, stride, padding, key):
        self.in_c = in_c
        self.out_c = out_c
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.key = key

    def __call__(self, x):

        input_dim = x.shape[1] + (x.shape[1] - 1) * (self.stride - 1)
        pad_row = jnp.zeros((x.shape[0], input_dim, x.shape[1]))
        pad_row = pad_row.at[:, ::2].set(x)
        pad_row_col = jnp.zeros((x.shape[0], input_dim, input_dim))
        pad_row_col = pad_row_col.at[:, :, ::2].set(pad_row)
        padding_amount = self.kernel - self.padding - 1
        pad_row_col_out = jnp.pad(pad_row_col, [(0, 0), (padding_amount, padding_amount),
                                                (padding_amount, padding_amount)])

        conv = nn.Conv2d(self.in_c, self.out_c, self.kernel, 1, 0, key=self.key)
        out = conv(pad_row_col_out)
        return out


class CNN(eqx.Module):
    """A CNN model."""
    layers_down: list
    layers_up: list
    layers_fc: list

    def __init__(self, key):
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)
        self.layers_down = [nn.Conv2d(1, 128, 3, 1, 1, key=key1),
                            nn.Conv2d(128, 64, 4, 2, 1, key=key2),
                            nn.Conv2d(64, 32, 4, 2, 1, key=key3)]

        self.layers_fc = [nn.Linear(1568, 980, key=key4)]

        self.layers_up = [CT2D(20 + 32, 128, 4, 2, 0, 1, key=key5),
                          CT2D(128 + 64, 1, 4, 2, 0, 1, key=key6)]

    def __call__(self, x):

        x0 = jax.nn.relu(self.layers_down[0](x))
        x1 = jax.nn.relu(self.layers_down[1](x0))
        x2 = jax.nn.relu(self.layers_down[2](x1))
        x3 = x2.reshape(1568, )
        x3 = self.layers_fc[0](x3)
        x3 = x3.reshape(20, 7, 7)
        x3 = jnp.concatenate((x3, x2), axis=0)
        x4 = jax.nn.relu(self.layers_up[0](x3))
        x4 = jnp.concatenate((x4, x1), axis=0)
        return jax.nn.sigmoid(self.layers_up[1](x4))
