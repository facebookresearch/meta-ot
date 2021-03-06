#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse

import shutil
import os
import re
import pickle as pkl
from collections import defaultdict

import numpy as np
import functools

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
    parser.add_argument('exp_root', type=str)
    parser.add_argument('--pkl_tag', type=str, default='latest')
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--timestamp', action='store_true')
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--num_test_samples', type=int, default=10)
    parser.add_argument('--no_errs', action='store_true')
    parser.add_argument('--no_labels', action='store_true')
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

            errs = {}
            solver = sinkhorn.make(lse_mode=True, inner_iterations=1, max_iterations=max_iterations, threshold=-1.)
            ot_prob = linear_problems.LinearProblem(geom, a=a, b=b)
            out = solver(ot_prob)
            errs['zeros'] = out.errors

            solver_gauss_init = sinkhorn.make(
                lse_mode=True, inner_iterations=1, max_iterations=max_iterations, threshold=-1.,
                potential_initializer=initializers.GaussianInitializer())
            out_gauss_init = solver_gauss_init(ot_prob)
            errs['gauss'] = out_gauss_init.errors

            f_pred = exp.potential_model.apply({'params': exp.params}, a, b)
            g_pred = exp.g_from_f(a, b, f_pred)
            init = (f_pred, g_pred)
            state = solver.init_state(ot_prob, init)
            out_meta = solver(ot_prob, init)
            errs['meta_ot'] = out_meta.errors

            return errs

        fname = f'{args.exp_root}/errs.pdf'
        print(f'Saving to {fname}')

        all_errs = defaultdict(list)
        for i in range(args.num_test_samples):
            print(f'Sample {i}')
            a, b = batch.a[i], batch.b[i]
            errs = compute_errs(a, b)
            for k,v in errs.items():
                all_errs[k].append(v)

            fig, ax = plt.subplots(1, 1, figsize=(4,2.))
            colors = plt.style.library['bmh']['axes.prop_cycle'].by_key()['color']

            def summarize_errs(errs):
                errs = np.stack(errs)
                mean = np.mean(errs, axis=0)
                std = np.std(errs, axis=0)
                return mean, std

            def plot_single(errs, color):
                mean, std = summarize_errs(errs)
                xs = np.arange(len(mean))
                ax.plot(xs, mean, color=color, alpha=0.7)
                ax.fill_between(xs, mean-std, mean+std, color=color, alpha=0.3)

            for key, color in zip(['zeros', 'meta_ot', 'gauss'], colors):
                plot_single(all_errs[key], color)

            to_title = {
                'mnist': 'MNIST',
                'usps28': 'USPS28',
                'doodles': 'Google Doodles',
                'world': 'Spherical',
                'random': 'Random'
            }
            if not args.no_labels:
                ax.set_ylabel('Error')
                ax.set_xlabel('Sinkhorn Iterations')
                ax.set_title(to_title[exp.cfg.data])
            else:
                ax.set_ylim(0., 0.4)
            fig.tight_layout()
            fig.savefig(fname, transparent=True)
            os.system(f'pdfcrop {fname} {fname}')

            plt.close(fig)

    # Runtime profiling
    max_iterations = 100000
    inner_iterations = 10

    @jax.jit
    def run_standard(threshold, a, b):
        solver = sinkhorn.make(
            lse_mode=True, inner_iterations=inner_iterations,
            max_iterations=max_iterations, threshold=threshold)
        ot_prob = linear_problems.LinearProblem(geom, a=a, b=b)
        out = solver(ot_prob)
        return out.converged

    @jax.jit
    def run_gauss_init(threshold, a, b):
        solver_gauss_init = sinkhorn.make(
            lse_mode=True, inner_iterations=inner_iterations,
            max_iterations=max_iterations,
            threshold=threshold,
            potential_initializer=initializers.GaussianInitializer())
        ot_prob = linear_problems.LinearProblem(geom, a=a, b=b)
        out = solver_gauss_init(ot_prob)
        return out.converged

    @jax.jit
    def run_meta_pred(a, b):
        f_pred = exp.potential_model.apply({'params': exp.params}, a, b)
        return True

    @jax.jit
    def run_meta(threshold, a, b):
        solver = sinkhorn.make(
            lse_mode=True, inner_iterations=inner_iterations,
            max_iterations=max_iterations, threshold=threshold)
        f_pred = exp.potential_model.apply({'params': exp.params}, a, b)
        g_pred = exp.g_from_f(a, b, f_pred)
        init = (f_pred, g_pred)
        ot_prob = linear_problems.LinearProblem(geom, a=a, b=b)
        state = solver.init_state(ot_prob, init)
        out_meta = solver(ot_prob, init)
        return out_meta.converged


    def profile(f):
        # Warm-up the solvers
        f(batch.a[0], batch.b[0])

        times = []
        for i in range(args.num_test_samples):
            # print(f'Sample {i}')
            a, b = batch.a[i], batch.b[i]
            start = time.time()
            converged = f(a, b)
            times.append(time.time() - start)
            assert converged

        result = fr'\pair{{{np.mean(times):.1e}}}{{{np.std(times):.1e}}}'
        result = re.sub('e(-[0-9]*)', r'\\cdot10^{\1}', result)
        result = result.replace('-0', '-')
        return result

    print('init pred:')
    print(profile(run_meta_pred))

    thresholds = [1e-2, 1e-3, 1e-4, 1e-5]
    prof_results = defaultdict(list)
    solver_funcs = {
        'standard': run_standard,
        'gauss_init': run_gauss_init,
        'meta': run_meta
    }
    for threshold in thresholds:
        for k in ['standard', 'gauss_init', 'meta']:
            f = functools.partial(solver_funcs[k], threshold)
            prof_results[k].append(profile(f))

    print(' & '.join([f'{v:.2e}' for v in thresholds]))
    for k in ['standard', 'gauss_init', 'meta']:
        print(k + ' & ' + ' & '.join(prof_results[k]))


if __name__ == '__main__':
    main()
