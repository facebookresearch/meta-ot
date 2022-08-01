#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

from omegaconf import OmegaConf
import argparse

import shutil
import os
import glob
import pickle as pkl
from collections import defaultdict

import pandas as pd
import numpy as np

import jax
import jax.numpy as jnp

import time

from ott.geometry.geometry import Geometry
from ott.core import quad_problems, linear_problems, sinkhorn, initializers
from ott.tools import transport
from ott.geometry.pointcloud import PointCloud

import matplotlib.pyplot as plt
plt.style.use('bmh')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Roman"]})

from meta_ot import data
from datetime import datetime

from jax.config import config
# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

DIR = os.path.dirname(os.path.realpath(__file__))

def main():
    import sys
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                         color_scheme='Linux',
                                         call_pdb=1)

    parser = argparse.ArgumentParser()
    parser.add_argument('sweep_root', type=str)
    args = parser.parse_args()

    exp_logs = []
    for exp_dir in glob.glob(args.sweep_root + '/*/'):
        print(exp_dir)

        try:
            conf = OmegaConf.load(f'{exp_dir}/.hydra/config.yaml')
            df = pd.read_csv(exp_dir + '/log.csv')
            exp_logs.append(df)
        except:
            continue

    fig, ax = plt.subplots(1, 1, figsize=(4,2.))

    for df in exp_logs:
        ax.plot(df.iter/1000, df.dual_obj, color='k', alpha=0.3)

    # ax.set_xlim(0, 50)
    # ax.set_ylim(0, ylims[k])
    ax.set_xlabel('1k Training Iterations')
    ax.set_ylabel('Dual')
    ax.set_title('Wasserstein-2')
    fig.tight_layout()

    fname = f'paper/fig/color-training-iter.pdf'
    fig.savefig(fname, transparent=True)
    os.system(f'pdfcrop {fname} {fname}')
    print(f'Saving to {fname}')

    fig, ax = plt.subplots(1, 1, figsize=(4,2.))

    for df in exp_logs:
        ax.plot(df.time/60., df.dual_obj, color='k', alpha=0.3)
    ax.set_xlabel('Training Time (minutes)')
    ax.set_ylabel('Dual')
    ax.set_title('Wasserstein-2')

    # ax.set_ylim(0, ylims[k])
    # ax.set_xlim(0, time_xlims[k])
    fig.tight_layout()
    fname = f'paper/fig/color-training-time.pdf'
    fig.savefig(fname, transparent=True)
    os.system(f'pdfcrop {fname} {fname}')
    print(f'Saving to {fname}')



if __name__ == '__main__':
    main()
