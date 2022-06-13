#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np

import jax
import jax.numpy as jnp

import scipy as sp
import scipy.io
import os

import argparse

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

import pickle as pkl

from meta_ot import data

import cv2

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Neutral', call_pdb=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_root', type=str)
    parser.add_argument('--pkl_tag', type=str, default='latest')
    parser.add_argument('--num_test_samples', type=int, default=10)
    args = parser.parse_args()

    exp = pkl.load(open(f'{args.exp_root}/{args.pkl_tag}.pkl', 'rb'))
    test_sampler = data.MNISTPairSampler(train=False, batch_size=28)
    geom = test_sampler.geom

    key = jax.random.PRNGKey(0)
    samples = test_sampler(key)
    num_interp = 8
    num_estimation_iter = 20
    batch_size = 500
    rows = [
        (samples.b[5], samples.b[1], False),
        (samples.b[1], samples.b[26], True),
    ]

    def plot(fname, exp=None):
        hists = []
        for a, b, endpoint in rows:
            hist_i = interp(
                geom, a, b, num_interp - (1 if not endpoint else 0),
                num_estimation_iter, batch_size=batch_size, exp=exp,
                endpoint=endpoint)
            hists.append(hist_i)

        hists = np.hstack(hists)

        fname = f'{args.exp_root}/{fname}'
        plt.imsave(fname, hists, cmap='Blues')

        # Because the base color of 'Blues' isn't compltely white...
        os.system(f"convert {fname} -fuzz 1% -transparent '#F7FBFF'-trim {fname}")

    plot('mnist-interp-true.png')
    plot('mnist-interp-pred.png', exp=exp)


def crop_hist_col(hist):
    hist_col_sum = hist.sum(axis=0)
    trim_left = np.argmax(hist_col_sum > 3.)
    trim_right = np.argmax(np.flip(hist_col_sum) > 3.)
    hist = hist[:,trim_left:-trim_right]
    return hist


def interp(geom, a, b, num_interp, num_estimation_iter, endpoint=True, batch_size=500, ignore_first=False, exp=None):
    if exp is None:
        ot_prob = problems.LinearProblem(geom, a=a, b=b)
        solver = sinkhorn.make(lse_mode=True)
        out = solver(ot_prob)
        log_P_flat = jnp.log(out.matrix.reshape(-1))
    else:
        P = exp.pred_transport(a=a, b=b)
        log_P_flat = jnp.log(P.reshape(-1))

    flat_image_size = 28*28

    @functools.partial(jax.jit, static_argnums=[2])
    def get_hist_iter(key, t):
        map_samples = jax.random.categorical(
            key, logits=log_P_flat, shape=[batch_size])
        a_samples = geom.x[map_samples // flat_image_size]
        b_samples = geom.y[map_samples % flat_image_size]
        proj_samples = (1.-t)*a_samples + t*b_samples
        hist_i, _, _ = jnp.histogram2d(proj_samples[:,1], proj_samples[:,0],
                            bins = jnp.linspace(0., 1., num=28+1))
        return hist_i

    def get_hist(key, t):
        hist = jnp.zeros((28,28))
        for i in range(num_estimation_iter):
            k1, key = jax.random.split(key, 2)
            hist += get_hist_iter(k1, t)
        hist /= hist.sum()
        hist = jnp.flipud(hist)

        hist = np.array(hist)
        thresh = np.quantile(hist, 0.9)
        hist[hist > thresh] = thresh
        hist = hist / hist.max()
        return hist

    hists = []
    ts = jnp.linspace(0, 1, num=num_interp, endpoint=endpoint)
    if ignore_first:
        ts = ts[1:]
    key = jax.random.PRNGKey(0)
    for i, t in enumerate(ts):
        hist = get_hist(key, t)
        hist = crop_hist_col(hist)
        hists.append(hist)

    hists = np.hstack(hists)
    return hists


if __name__ == '__main__':
    main()
