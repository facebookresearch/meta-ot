#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np

import jax
import jax.numpy as jnp

import scipy as sp
import scipy.io
import os

import matplotlib.pyplot as plt
plt.style.use('bmh')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Roman"]})


from ott.geometry.geometry import Geometry
from ott.core import quad_problems, problems, sinkhorn
from ott.tools import transport
from ott.geometry.pointcloud import PointCloud

import functools

from itertools import product

from meta_ot import data

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Neutral', call_pdb=1)


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def main():
    num_interp = 8
    epsilons = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
    num_estimationum_estimation_iter = 10

    nrow, ncol = len(epsilons), 1
    fig, axs = plt.subplots(
        nrow, ncol, figsize=(7, 4),
        gridspec_kw={'wspace': 0, 'hspace': 0},
        dpi=200)

    for ax, epsilon in zip(axs, epsilons):
        test_sampler = data.MNISTPairSampler(
            train=False, batch_size=28,
            epsilon=epsilon,
        )
        geom = test_sampler.geom

        key = jax.random.PRNGKey(0)
        samples = test_sampler(key)

        hists_1 = interp(geom, samples.a[1], samples.b[1], num_interp, num_estimationum_estimation_iter)
        hists_2 = interp(geom, samples.b[1], samples.b[3], num_interp, num_estimationum_estimation_iter, ignore_first=True)
        hists = np.hstack([hists_1, hists_2])

        ax.imshow(hists, cmap='Blues')
        fig.tight_layout()

        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.set_ylabel(rf'$$\epsilon={epsilon}$$')

        fig.tight_layout()
        fname = 'mnist-epsilons.pdf'
        plt.savefig(fname, transparent=True)
    os.system(f'pdfcrop {fname} {fname}')

def crop_hist_col(hist):
    hist_col_sum = hist.sum(axis=0)
    trim_left = np.argmax(hist_col_sum > 3.)
    trim_right = np.argmax(np.flip(hist_col_sum) > 3.)
    hist = hist[:,trim_left:-trim_right]
    return hist

def interp(geom, a, b, num_interp, num_estimation_iter, ignore_first=False):
    ot_prob = problems.LinearProblem(geom, a=a, b=b)
    solver = sinkhorn.make(lse_mode=True)
    out = solver(ot_prob)

    out_log = jnp.log(out.matrix.reshape(-1))
    flat_image_size = 28*28

    @functools.partial(jax.jit, static_argnums=[2])
    def get_hist_iter(key, t, batch_size):
        map_samples = jax.random.categorical(
            key, logits=out_log, shape=[batch_size])
        a_samples = geom.x[map_samples // flat_image_size]
        b_samples = geom.y[map_samples % flat_image_size]
        proj_samples = (1.-t)*a_samples + t*b_samples
        hist_i, _, _ = jnp.histogram2d(proj_samples[:,1], proj_samples[:,0],
                            bins = jnp.linspace(0., 1., num=28+1))
        return hist_i

    def get_hist(key, t, batch_size=1000):
        hist = jnp.zeros((28,28))
        for i in range(num_estimation_iter):
            k1, key = jax.random.split(key, 2)
            hist += get_hist_iter(k1, t, batch_size)
        hist /= hist.sum()
        hist = jnp.flipud(hist)

        hist = np.array(hist)
        thresh = np.quantile(hist, 0.9)
        hist[hist > thresh] = thresh
        hist = hist / hist.max()
        return hist

    hists = []
    ts = jnp.linspace(0, 1, num=num_interp)
    if ignore_first:
        ts = ts[1:]
    key = jax.random.PRNGKey(0)
    for i, t in enumerate(ts):
        hist = get_hist(key, t, batch_size=1000)
        hist = crop_hist_col(hist)
        hists.append(hist)

    hists = np.hstack(hists)
    return hists


if __name__ == '__main__':
    main()
