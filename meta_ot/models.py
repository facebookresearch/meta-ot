# Copyright (c) Meta Platforms, Inc. and affiliates.

from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn

from typing import Any, Callable, Sequence, Tuple

from ott.core.icnn import PositiveDense

class PotentialMLP(nn.Module):
    n_output: int
    n_hidden: int
    n_hidden_layer: int

    @nn.compact
    def __call__(self, a, b):
        z = jnp.concatenate((a, b))
        for i in range(self.n_hidden_layer):
            # TODO: get dtype from caller
            z = nn.relu(nn.Dense(self.n_hidden, dtype=jnp.float64)(z))
        f = nn.Dense(self.n_output, dtype=jnp.float64)(z)
        return f


class LatentGaussianEnc(nn.Module):
    n_hidden: int
    n_hidden_layer: int
    latent_dim: int

    @nn.compact
    def __call__(self, density):
        z = density
        for i in range(self.n_hidden_layer):
            z = nn.relu(nn.Dense(self.n_hidden)(z))
        mean = nn.Dense(self.latent_dim)(z)
        cov = nn.Dense(self.latent_dim*self.latent_dim)(z).reshape(self.latent_dim, self.latent_dim)

        # TODO: The default initialization here gives narrow distributions.
        # Explore better ways of inititializing/regularzing this
        # TODO: Index into diagonal instead of adding jnp.eye?
        cov = 0.1*cov.dot(cov.T) + jnp.eye(self.latent_dim)

        return Gaussian(mean, cov)

class LatentGaussianDec(nn.Module):
    n_discrete: int
    n_hidden: int
    n_hidden_layer: int

    @nn.compact
    def __call__(self, A, b):
        z = jnp.concatenate((A.reshape(-1), b))
        for i in range(self.n_hidden_layer):
            z = nn.relu(nn.Dense(self.n_hidden)(z))
        f = nn.Dense(self.n_discrete)(z)
        return f


class LatentGaussianPotential(nn.Module):
    n_discrete: int
    latent_dim: int
    latent_enc: LatentGaussianEnc
    latent_dec: LatentGaussianDec

    @nn.compact
    def __call__(self, a, b, debug=False):
        latent_a = self.latent_enc(a)
        latent_b = self.latent_enc(b)
        A, b = gaussian_transport(latent_mu, latent_nu)
        f_pred = self.latent_dec(A, b)

        if debug:
            return f_pred, latent_a, latent_b, A, b
        else:
            return f_pred

class ICNN(nn.Module):
    dim_hidden: Sequence[int]
    init_std: float = 0.1
    init_fn: Callable = jax.nn.initializers.normal
    act: str = 'leaky_relu' # Store a string here rather than the function for pickling
    quad_rank: int = 3

    def setup(self):
        num_hidden = len(self.dim_hidden)

        w_zs = list()
        for i in range(1, num_hidden):
            w_zs.append(PositiveDense(
                self.dim_hidden[i], kernel_init=self.init_fn(self.init_std),
                use_bias=False))
        w_zs.append(PositiveDense(
            1, kernel_init=self.init_fn(self.init_std), use_bias=False))
        self.w_zs = w_zs

        w_xs = list()
        for i in range(num_hidden):
            w_xs.append(nn.Dense(
                self.dim_hidden[i], kernel_init=self.init_fn(self.init_std),
                use_bias=True))

        w_xs.append(nn.Dense(
            1, kernel_init=self.init_fn(self.init_std), use_bias=True))
        self.w_xs = w_xs

    @nn.compact
    def __call__(self, x):
        assert x.ndim == 1
        n_input = x.shape[0]

        if self.act == 'leaky_relu':
            act_fn = jax.nn.leaky_relu
        else:
            assert False

        z = act_fn(self.w_xs[0](x))

        for Wz, Wx in zip(self.w_zs[:-1], self.w_xs[1:-1]):
            z = act_fn(jnp.add(Wz(z), Wx(x)))

        L = self.param('L', nn.initializers.normal(), (self.quad_rank, n_input))
        quad = x.dot(L.transpose().dot(L)).dot(x)

        y = jnp.add(self.w_zs[-1](z), self.w_xs[-1](x)) + quad
        return jnp.squeeze(y)


class MetaICNN(nn.Module):
    num_icnn_params: int
    bottleneck_size: int = 512
    fc_num_hidden_units: int = 512
    fc_num_hidden_layers: int = 2

    def setup(self):
        assert self.bottleneck_size % 2 == 0
        self.resnet = ResNet18(num_classes=self.bottleneck_size // 2)

    @nn.compact
    def __call__(self, x, y, train=True, unravel_fn=None):
        assert x.ndim == y.ndim
        batched = x.ndim == 4
        if not batched:
            x = jnp.expand_dims(x, 0)
            y = jnp.expand_dims(y, 0)
        assert x.ndim == y.ndim == 4

        zx = self.resnet(x, train=train)
        zy = self.resnet(y, train=train)

        z = jnp.concatenate((zx, zy), axis=-1)
        for i in range(self.fc_num_hidden_layers):
            z = nn.relu(nn.Dense(features=self.fc_num_hidden_units)(z))
        z = nn.Dense(features=2*self.num_icnn_params)(z)

        if not batched:
            z = z[0]

        D_params_flat, D_conj_params_flat = jnp.split(z, 2, axis=-1)
        if unravel_fn is not None:
            assert not batched
            D_params = unravel_fn(D_params_flat)
            D_conj_params = unravel_fn(D_conj_params_flat)
            return D_params, D_conj_params
        else:
            return D_params_flat, D_conj_params_flat


ModuleDef = Any

class ResNetBlock(nn.Module):
  """ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x,):
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters, (1, 1),
                           self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.conv(self.filters, (1, 1))(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3), self.strides)(y)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters * 4, (1, 1))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters * 4, (1, 1),
                           self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class ResNet(nn.Module):
  """ResNetV1."""
  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  num_classes: int
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  conv: ModuleDef = nn.Conv

  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = partial(self.conv, use_bias=False, dtype=self.dtype)
    norm = partial(nn.BatchNorm,
                   use_running_average=not train,
                   momentum=0.9,
                   epsilon=1e-5,
                   dtype=self.dtype)

    x = conv(self.num_filters, (7, 7), (2, 2),
             padding=[(3, 3), (3, 3)],
             name='conv_init')(x)
    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(self.num_filters * 2 ** i,
                           strides=strides,
                           conv=conv,
                           norm=norm,
                           act=self.act)(x)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)
    return x


ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2],
                   block_cls=ResNetBlock)
