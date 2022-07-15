#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse

import shutil
import os
import pickle as pkl
from collections import defaultdict

import numpy as np

import jax
import jax.numpy as jnp

import time

from ott.geometry.geometry import Geometry
from ott.core import quad_problems, problems, sinkhorn
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

    expPath = './exp/local/'
    dirs = sorted(next(os.walk(expPath))[1])
    latest = sorted(next(os.walk(expPath + dirs[-1]))[1])[-1]

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root', type=str, default=expPath + dirs[-1] + '/' + latest)
    parser.add_argument('--pkl_tag', type=str, default='latest')
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--timestamp', action='store_true')
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--num_test_samples', type=int, default=10)
    parser.add_argument('--no_errs', action='store_true')
    args = parser.parse_args()

    exp = pkl.load(open(f'{args.exp_root}/{args.pkl_tag}.pkl', 'rb'))

    if args.test_data is None:
        args.test_data = exp.cfg.data

    if args.test_data == 'mnist':
        test_sampler = data.MNISTPairSampler(train=args.train)
        geom = test_sampler.geom
    elif args.test_data == 'usps28':
        test_sampler = data.USPSPairSampler(train=args.train, reshape=True)
        geom = test_sampler.geom
    elif args.test_data == 'doodles':
        test_sampler = data.DoodlePairSampler(train=args.train)
        geom = test_sampler.geom
    elif args.test_data == 'random':
        test_sampler = data.RandomSampler()
        geom = test_sampler.geom
    elif args.test_data == 'world':
        test_sampler = data.WorldPairSampler()
        geom = test_sampler.geom
    else:
        assert False

    key = jax.random.PRNGKey(0)
    batch = test_sampler(key)

    if not args.no_errs:
        @jax.jit
        def compute_errs(a, b):
            if args.test_data in ['mnist', 'random', 'usps28', 'doodles']:
                max_iterations = 25
            elif exp.cfg.data == 'world':
                max_iterations = 1000
            else:
                assert False
            solver = sinkhorn.make(lse_mode=True, inner_iterations=1, max_iterations=max_iterations, threshold=-1.)
            ot_prob = problems.LinearProblem(geom, a=a, b=b)
            out = solver(ot_prob)

            f_pred = exp.potential_model.apply({'params': exp.params}, a, b)
            g_pred = exp.g_from_f(a, b, f_pred)
            init = (f_pred, g_pred)
            state = solver.init_state(ot_prob, init)
            out_meta = solver(ot_prob, init)
            return out, out_meta


        fig, ax = plt.subplots(1, 1, figsize=(4,2.))
        colors = plt.style.library['bmh']['axes.prop_cycle'].by_key()['color']

        if args.timestamp:
            fname = f'{args.exp_root}/errs_' + str(datetime.now()) + '.pdf'
        else:
            fname = f'{args.exp_root}/errs.pdf'

        print(f'Saving to {fname}')
        for i in range(args.num_test_samples):
            print(f'Sample {i}')
            a, b = batch.a[i], batch.b[i]
            ot, meta_ot = compute_errs(a, b)
            ax.plot(ot.errors, color=colors[0], alpha=0.7)
            ax.plot(meta_ot.errors, color=colors[1], alpha=0.7)
            # ax.set_xscale('log')
            # ax.set_yscale('log')

            ax.set_ylabel('Error')
            ax.set_xlabel('Sinkhorn Iterations')
            to_title = {
                'mnist': 'MNIST',
                'usps28': 'USPS28',
                'doodles': 'Google Doodles',
                'world': 'World',
                'random': 'Random'
            }
            ax.set_title(to_title[exp.cfg.data])
            fig.tight_layout()
            fig.savefig(fname, transparent=True)
            os.system(f'pdfcrop {fname} {fname}')

        plt.close(fig)

    # Runtime profiling
    max_iterations = 100000
    threshold = 1e-3
    inner_iterations = 10
    solver = sinkhorn.make(
        lse_mode=True, inner_iterations=inner_iterations,
        max_iterations=max_iterations, threshold=threshold)

    @jax.jit
    def run_standard(a, b):
        ot_prob = problems.LinearProblem(geom, a=a, b=b)
        out = solver(ot_prob)
        return out.converged

    @jax.jit
    def run_meta_pred(a, b):
        f_pred = exp.potential_model.apply({'params': exp.params}, a, b)
        return True

    @jax.jit
    def run_meta(a, b):
        f_pred = exp.potential_model.apply({'params': exp.params}, a, b)
        g_pred = exp.g_from_f(a, b, f_pred)
        init = (f_pred, g_pred)
        ot_prob = problems.LinearProblem(geom, a=a, b=b)
        state = solver.init_state(ot_prob, init)
        out_meta = solver(ot_prob, init)
        return out_meta.converged


    def profile(f, tag):
        # Warm-up the solvers
        f(batch.a[0], batch.b[0])

        times = []
        for i in range(args.num_test_samples):
            print(f'Sample {i}')
            a, b = batch.a[i], batch.b[i]
            start = time.time()
            converged = f(a, b)
            times.append(time.time() - start)
            assert converged

        print(f'{tag}: {np.mean(times):.1e} +/- {np.std(times):.1e}')

    profile(run_standard, 'standard')
    profile(run_meta_pred, 'meta_init_pred')
    profile(run_meta, 'meta')


if __name__ == '__main__':
    main()
