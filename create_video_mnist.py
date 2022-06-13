#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np

import jax
import jax.numpy as jnp

import scipy as sp
import scipy.io
import os
import shutil

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
    parser.add_argument('--num_row', type=int, default=9)
    parser.add_argument('--num_col', type=int, default=10)
    parser.add_argument('--num_interp', type=int, default=50)
    parser.add_argument('--num_estimation_iter', type=int, default=20)
    args = parser.parse_args()

    num_samples = args.num_row * args.num_col
    test_sampler = data.MNISTPairSampler(train=False, batch_size=num_samples)
    geom = test_sampler.geom

    key = jax.random.PRNGKey(0)
    samples = test_sampler(key)
    batch_size = 500

    exp = pkl.load(open(f'{args.exp_root}/{args.pkl_tag}.pkl', 'rb'))
    hists = []
    for i, (a, b) in enumerate(zip(samples.a, samples.b)):
        print(f'Coupling {i}/{num_samples}')
        hist_i = interp(
            geom, a, b, args.num_interp,
            args.num_estimation_iter, batch_size=batch_size, exp=exp)
        hists.append(hist_i)

    frames_dir = f'{args.exp_root}/frames'
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)
    for frame_i in range(args.num_interp):
        frame_hists = [hist[frame_i] for hist in hists]
        rows = []
        start_idx = 0
        for i in range(args.num_row):
            end_idx = start_idx + args.num_col
            rows.append(np.hstack(frame_hists[start_idx:end_idx]))
            start_idx = end_idx
        frame = np.vstack(rows)
        fname = f'{frames_dir}/{frame_i:04d}.png'
        plt.imsave(fname, frame, cmap='Blues')

        # Because the base color of 'Blues' isn't compltely white...
        os.system(f"convert {fname} -fuzz 1% -transparent '#F7FBFF' {fname}")
        os.system(f"convert {fname} -background White -alpha remove -alpha off {fname}")
        os.system(f'convert -font /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf -size 600x20 -gravity Center label:"Optimally transport between MNIST digits" {fname} -resize 600x -append {fname}')
        os.system(f'convert {fname} -resize 600x546 -background white -gravity south -extent 600x546 {fname}')

    os.system(f'ffmpeg -i {frames_dir}/%04d.png {args.exp_root}/mnist-vid.mp4 -y')
    os.system(f'ffmpeg -i {args.exp_root}/mnist-vid.mp4 -vf reverse {args.exp_root}/mnist-vid-rev.mp4 -y')


def interp(geom, a, b, num_interp, num_estimation_iter, batch_size=500, ignore_first=False, exp=None):
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
        thresh = np.quantile(hist, 0.95)
        assert thresh > 0.
        hist[hist > thresh] = thresh
        hist = hist / hist.max()
        return hist

    hists = []
    ts = jnp.linspace(0, 1, num=num_interp)
    if ignore_first:
        ts = ts[1:]
    key = jax.random.PRNGKey(0)
    for i, t in enumerate(ts):
        hist = get_hist(key, t)
        hists.append(hist)

    return hists


if __name__ == '__main__':
    main()
