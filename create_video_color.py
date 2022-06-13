#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import copy
from omegaconf import OmegaConf
import pandas as pd

import jax
import jax.numpy as jnp
import torch
from torch import nn
import numpy as np

from meta_ot.data import ImageSampler, ImagePairSampler
from meta_ot import conjugate
from meta_ot.models import ResNet18
from train_color_single import push_grad

import time
import pickle as pkl
import os
import shutil
from collections import defaultdict

from PIL import Image

from sklearn.manifold import TSNE
import torchvision

from train_color_single import Workspace as SingleWorkspace

import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
plt.style.use('bmh')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Roman"]})

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Neutral', call_pdb=1)

DIR = os.path.dirname(os.path.realpath(__file__))

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                        color_scheme='Linux',
                                        call_pdb=1)

def plot(X_sampler, Y_sampler, loc, apply_jit, push_conj_jit, crop_height=400, gamma=1.):
    D_params, D_conj_params = apply_jit(X_sampler.image_square, Y_sampler.image_square)

    w, h = X_sampler.image.size
    im1 = X_sampler.image.resize((int(crop_height * w/h), crop_height), Image.ANTIALIAS)
    w, h = Y_sampler.image.size
    im2 = Y_sampler.image.resize((int(crop_height * w/h), crop_height), Image.ANTIALIAS)
    # im1_push = push_image(self.D, D_params, im1, gamma=gamma)
    im2_push = push_image(push_conj_jit, D_conj_params, im2, gamma=gamma)
    im1 = jnp.array(im1)
    im2 = jnp.array(im2)
    # out =  jnp.concatenate((im1, im2, im1_push, im2_push), axis=1)
    out =  jnp.concatenate((im1, im2, im2_push), axis=1)
    # im2_push
    plt.imsave(loc, out)


def push_image(push_jit, params, image, batch_size=5000, gamma=1.):
    # TODO: Could make this much more efficient for the interpolations
    # by not recomputing the transport map for every gamma
    X = np.asarray(image).transpose(2, 0, 1)
    orig_shape = X.shape
    X = (X.reshape(3, -1) / 255.).T
    X_pushed = np.zeros_like(X)
    pos = 0;
    while pos < len(X):
        X_pushed[pos:pos+batch_size] = (1-gamma)*X[pos:pos+batch_size] + gamma*push_jit(X[pos:pos+batch_size], params)
        pos += batch_size

    im_X_pushed = (
        np.clip(X_pushed.T.reshape(orig_shape).transpose(1, 2, 0), 0, 1) * 255
    ).astype(np.uint8)
    return im_X_pushed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_root', type=str)
    parser.add_argument('--pkl_tag', type=str, default='latest')
    parser.add_argument('--num_interp_steps', type=int, default=100)
    args = parser.parse_args()

    exp_path = f'{args.exp_root}/{args.pkl_tag}.pkl'
    assert os.path.exists(exp_path)
    with open(exp_path, 'rb') as f:
        exp = pkl.load(f)


    root_vid_dir = f'{args.exp_root}/vids'
    if os.path.exists(root_vid_dir):
        shutil.rmtree(root_vid_dir)
    os.makedirs(root_vid_dir)


    @jax.jit
    def meta_apply(X, Y):
        D_params_meta, D_conj_params_meta = exp.meta_icnn.apply(
            {'params': exp.meta_params, 'batch_stats': exp.meta_batch_stats},
            X, Y, train=False,
            unravel_fn=exp.unravel_icnn_params_fn)
        return D_params_meta, D_conj_params_meta

    push_conj_jit = jax.jit(lambda X, D_conj_params: push_grad(exp.D_conj, D_conj_params, X))


    _, val_pairs = exp.load_val()
    val_idxs = [0,4,7]
    vid_dirs = []
    for i, val_idx in enumerate(val_idxs):
        (X_path, Y_path) = val_pairs[val_idx]
        vid_i_dir = f'{root_vid_dir}/{i:02d}'
        vid_dirs.append(vid_i_dir)
        os.makedirs(vid_i_dir)

        X_sampler, Y_sampler = ImageSampler(X_path), ImageSampler(Y_path)
        print(X_sampler.path, Y_sampler.path)

        D_params_meta, D_conj_params_meta = meta_apply(
            X_sampler.image_square, Y_sampler.image_square)
        gammas = np.linspace(0., 1., num=args.num_interp_steps)
        for gamma_idx, gamma in enumerate(gammas):
            plot(X_sampler, Y_sampler, loc=f'{vid_i_dir}/{gamma_idx:04d}.png',
                 apply_jit=meta_apply, push_conj_jit=push_conj_jit,
                 gamma=gamma)

    frames_dir = f'{root_vid_dir}/frames'
    os.makedirs(frames_dir)
    for frame in os.listdir(vid_dirs[0]):
        frame_fname = f'{frames_dir}/{frame}'
        s = f'convert -font /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf -size 600x20 -gravity Center label:"Transfer the color palette from the first image to the second" '
        for vid_dir in vid_dirs:
            s += f'{vid_dir}/{frame} -resize 600x '
        s += f'-append {frame_fname}'
        os.system(s)

        # os.system(f'convert {frame_fname} -resize 600x546 -background white -gravity north -extent 600x546 {frame_fname}')

    final_loc = frame_fname

    # Add extra frames at the end
    for j in range(args.num_interp_steps // 2):
        idx = args.num_interp_steps + j
        loc=f'{frames_dir}/{idx:04d}.png'
        os.system(f'ln -s {final_loc} {loc}')

    os.system(f'ffmpeg -i {frames_dir}/%04d.png {root_vid_dir}/color.mp4 -y')


if __name__ == '__main__':
    main()
