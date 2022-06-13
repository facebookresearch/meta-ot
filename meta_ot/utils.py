# Copyright (c) Meta Platforms, Inc. and affiliates.

import random

import numpy as np
import torch
import jax
import jax.numpy as jnp

from PIL import Image, ImageOps

from ott.geometry import costs
import ot

from collections import namedtuple
Gaussian = namedtuple("Gaussian", "mean cov")


def spherical_to_euclidean(theta_phi):
    single= theta_phi.ndim == 1
    if single:
        theta_phi = jnp.expand_dims(theta_phi, 0)
    theta, phi = jnp.split(theta_phi, 2, 1)
    return jnp.concatenate((
        jnp.sin(phi) * jnp.cos(theta),
        jnp.sin(phi) * jnp.sin(theta),
        jnp.cos(phi)
    ), 1)


def euclidean_to_spherical(xyz):
    single = xyz.ndim == 1
    if single:
        xyz = jnp.expand_dims(xyz, 0)
    x, y, z = jnp.split(xyz, 3, 1)
    return jnp.concatenate((
        jnp.arctan2(y, x),
        jnp.arccos(z)
    ), 1)


@jax.tree_util.register_pytree_node_class
class SphereDist(costs.CostFn):
    def pairwise(self, x, y):
        cosine_eps = 1e-4
        return jnp.arccos(jnp.vdot(x, y) / (1.+cosine_eps))


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.999):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
