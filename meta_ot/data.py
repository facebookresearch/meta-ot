# Copyright (c) Meta Platforms, Inc. and affiliates.

import random

import jax
import jax.numpy as jnp

import numpy as np
import numpy.random as npr

import torch
from torchvision import transforms
from torchvision.datasets import MNIST

from collections import namedtuple

from ott.core import problems
from ott.geometry.pointcloud import PointCloud

from dataclasses import dataclass

from PIL import Image

from . import utils

import os
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

PairData = namedtuple('PairData', 'a b')
PairDataImgs = namedtuple('PairDataImgs', 'a b afull bfull')

@dataclass
class MNISTPairSampler:
    train: bool = True
    batch_size: int = 128
    epsilon: float = 1e-2
    debug: bool = False

    def __post_init__(self):
        dataset = MNIST(
            '/tmp/mnist/',
            download=True,
            train=self.train,
        )
        data = dataset.data
        data = jnp.float64(data)/255.
        data = data.reshape(-1, 784)
        data = data/data.sum(axis=1, keepdims=True)
        self.data = data

        x_grid = []
        for i in jnp.linspace(1, 0, num=28):
            for j in jnp.linspace(0, 1, num=28):
                x_grid.append([j, i])
        x_grid = jnp.array(x_grid)
        self.geom = PointCloud(x=x_grid, y=x_grid, epsilon=self.epsilon, online=True)

        @jax.jit
        def _sample(key):
            k1, k2, key = jax.random.split(key, num=3)
            I = jax.random.randint(k1, shape=[self.batch_size], minval=0, maxval=len(data))
            J = jax.random.randint(k2, shape=[self.batch_size], minval=0, maxval=len(data))
            a = data[I]
            b = data[J]
            return PairData(a, b)
        self._sample = _sample

        if self.debug:
            key = jax.random.PRNGKey(0)
            self._debug_data = self._sample(key)


    def __call__(self, key):
        if self.debug:
            return self._debug_data
        else:
            return self._sample(key)

@dataclass
class WorldPairSampler:
    batch_size: int = 128
    epsilon: float = 1e-3
    population_fname: str = SCRIPT_DIR + '/../data/pop-15min.tif'
    supply_bernoulli_p: float = 0.5
    n_demand: int = 10000
    n_supply: int = 100
    debug: bool = False

    def __post_init__(self):
        import rasterio
        # Using 2020 Tiff data at 15-minute resolution from:
        # https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-density-adjusted-to-2015-unwpp-country-totals-rev11/data-download#
        src = rasterio.open(self.population_fname)

        # Population
        P = src.read(1)
        P[P < 0] = 0.
        Pflat = P.ravel()
        Pflat = Pflat / Pflat.max() # For numerical stability
        Pflat = Pflat / Pflat.sum()
        self.P = P
        self.Pflat = Pflat

        # Uniform over ~landmass
        Uflat = Pflat.copy()
        Uflat[Uflat > 0] = 1.
        Uflat /= Uflat.sum()
        self.Uflat = Uflat

        # Sample spherical and euclidean locations from p
        def sample(p, num_samples, seed=0):
            npr.seed(seed)
            sample_Is = npr.choice(len(p), p=p, size=num_samples)
            samples_theta = P.shape[0] - sample_Is / P.shape[1]
            samples_theta = (samples_theta / P.shape[0]) * np.pi
            samples_phi = sample_Is % P.shape[1]
            samples_phi = (samples_phi / P.shape[1]) * 2 * np.pi - np.pi
            samples_spherical = np.vstack((samples_phi, samples_theta)).T
            samples_euclidean = utils.spherical_to_euclidean(samples_spherical)
            return samples_spherical, samples_euclidean

        self.demand_locs_spherical, self.demand_locs_euclidean = sample(Pflat, self.n_demand, seed=0)
        self.supply_locs_spherical, self.supply_locs_euclidean = sample(Uflat, self.n_supply, seed=1)

        self.geom = PointCloud(
            x=self.supply_locs_euclidean, y=self.demand_locs_euclidean,
            epsilon=self.epsilon, online=True, cost_fn=utils.SphereDist())

        @jax.jit
        def _sample(key):
            k1, k2, k3, key = jax.random.split(key, num=4)

            demand_probs = jax.random.uniform(k1, [self.batch_size, self.n_demand])
            demand_probs /= demand_probs.sum(axis=1, keepdims=True)

            mask = jax.random.bernoulli(
                k2, p=self.supply_bernoulli_p, shape=[self.batch_size, self.n_supply])
            supply_probs = mask * jax.random.uniform(k3, [self.batch_size, self.n_supply])
            supply_probs /= supply_probs.sum(axis=1, keepdims=True)

            return PairData(supply_probs, demand_probs)

        self._sample = _sample

        if self.debug:
            key = jax.random.PRNGKey(0)
            self._debug_data = self._sample(key)


    def __call__(self, key):
        if self.debug:
            return self._debug_data
        else:
            return self._sample(key)


class ImageSampler:
    mean = jnp.array([0.485, 0.456, 0.406])
    std = jnp.array([0.229, 0.224, 0.225])

    def __init__(self, image_path, square_size=224, num_rgb_sample=None, key=None):
        self.path = image_path
        self.image = Image.open(image_path).convert('RGB')
        self.flat_norm_image = (jnp.asarray(self.image).transpose(2, 0, 1).reshape(3, -1) / 255.).T

        if num_rgb_sample is not None:
            I = jax.random.choice(key, len(self.flat_norm_image), shape=[num_rgb_sample])
            self.flat_norm_image = self.flat_norm_image[I]

        image_square = self.image.resize((square_size, square_size))
        self.image_square = self.normalize_image(jnp.asarray(image_square) / 255.)

    def normalize_image(self, image):
        return (image - self.mean) / self.std

    def unnormalize_image(self, image):
        return image * self.std + self.mean


    def sample(self, key, batch_size):
        I = jax.random.choice(key, len(self.flat_norm_image), shape=[batch_size])
        batch = self.flat_norm_image[I]
        return batch


class ImagePairSampler:
    def __init__(self, image_paths, num_rgb_sample=None, key=None):
        samplers = []
        for path in image_paths:
            if key is not None:
                key, = jax.random.split(key, 1)
            samplers.append(ImageSampler(path, num_rgb_sample=num_rgb_sample, key=key))
        self.samplers = samplers

    def sample_image_pair(self, val_pairs):
        X_sampler, Y_sampler = random.sample(self.samplers, 2)
        if val_pairs is not None:
            # Sample until it's not a validation pair
            if (X_sampler.path, Y_sampler.path) in val_pairs:
                return self.sample_image_pair(val_pairs)
        return X_sampler, Y_sampler

    def sample_image_pair_batch(self, batch_size=1, val_pairs=None):
        # TODO: Could clean all these names up, maybe with something like:
        # ImagePair = namedtuple('ImagePair', 'X_square X_flat Y_square Y_flat')
        X_samplers, Y_samplers, X_squares, Y_squares, X_fulls, Y_fulls = [], [], [], [], [], []
        for i in range(batch_size):
            X_sampler, Y_sampler = self.sample_image_pair(val_pairs)
            X_samplers.append(X_sampler)
            Y_samplers.append(Y_sampler)
            X_squares.append(X_sampler.image_square)
            Y_squares.append(Y_sampler.image_square)
            X_fulls.append(X_sampler.flat_norm_image)
            Y_fulls.append(Y_sampler.flat_norm_image)
        X_squares = jnp.stack(X_squares)
        Y_squares = jnp.stack(Y_squares)
        X_fulls = jnp.stack(X_fulls)
        Y_fulls = jnp.stack(Y_fulls)
        return X_samplers, Y_samplers, X_squares, Y_squares, X_fulls, Y_fulls
