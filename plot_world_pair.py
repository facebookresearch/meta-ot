#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse

import pickle as pkl

import numpy as np
from matplotlib import pyplot as plt
plt.style.use('bmh')

import os

import jax
import jax.numpy as jnp

from ott.core import quad_problems, problems, sinkhorn
# from ott.tools import transport
# from ott.geometry.pointcloud import PointCloud

from meta_ot.data import WorldPairSampler
from meta_ot import utils

from jax.config import config
config.update("jax_enable_x64", True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_root', type=str)
    parser.add_argument('--pkl_tag', type=str, default='latest')
    parser.add_argument('--num_test_samples', type=int, default=10)
    args = parser.parse_args()

    exp = pkl.load(open(f'{args.exp_root}/{args.pkl_tag}.pkl', 'rb'))

    key = jax.random.PRNGKey(0)
    sampler = WorldPairSampler(epsilon=1e-3) #n_supply=10, n_demand=10)
    pair = sampler(key)
    a = pair.a[0]
    b = pair.b[0]

    for i in range(args.num_test_samples):
        plot(pair.a[i], pair.b[i], sampler, f'truth.{i}', args.exp_root)
        plot(pair.a[i], pair.b[i], sampler, f'meta.{i}', args.exp_root, exp)


def plot(a, b, sampler, tag, work_dir, exp=None):
    if exp is None:
        solver = sinkhorn.make(lse_mode=True, jit=True,
                            inner_iterations=10, max_iterations=10000) #, threshold=-1.)
        ot_prob = problems.LinearProblem(sampler.geom, a=a, b=b)
        out = solver(ot_prob)
        assert out.converged

        T = out.matrix.argmax(axis=0)
    else:
        P = exp.pred_transport(a=a, b=b)
        T = P.argmax(axis=0)

    demand_to_supply = sampler.supply_locs_euclidean[T]

    fig, ax = plt.subplots(figsize=(6,4))
    colors = plt.style.library['bmh']['axes.prop_cycle'].by_key()['color']

    ax.imshow(sampler.Uflat.reshape(sampler.P.shape), cmap='gray_r',
            extent=[-np.pi, np.pi, 0, np.pi], alpha=0.15)

    I = a > 0
    # ax.scatter(demand_locs_spherical[:,0], demand_locs_spherical[:,1], s=1)
    ax.scatter(sampler.supply_locs_spherical[I,0], sampler.supply_locs_spherical[I,1], s=4., color='k', zorder=10)

    for i in range(sampler.n_demand):
        t = np.expand_dims(np.linspace(0, 1, 1000), 1)
        geodesic = t*sampler.demand_locs_euclidean[i] + (1-t)*demand_to_supply[i]
        geodesic = geodesic / np.linalg.norm(geodesic, axis=1, keepdims=True)
        geodesic = utils.euclidean_to_spherical(geodesic)
        geodesic = np.array(geodesic)

        # Remove discontinuities in the geodesic
        n = np.linalg.norm(geodesic[:-1] - geodesic[1:], axis=1)
        geodesic[1:][n > 0.1] = np.nan
        ax.plot(geodesic[:,0], geodesic[:,1], color=colors[0], alpha=.1, linewidth=1)


    fig.tight_layout()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fname = f'{work_dir}/{tag}.pdf'
    fig.savefig(fname, transparent=True)
    os.system(f'pdfcrop {fname} {fname}')


if __name__ == '__main__':
    main()
