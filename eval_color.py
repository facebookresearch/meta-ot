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

class Workspace:
    def __init__(self, args):
        self.args = args
        exp_path = f'{args.exp_root}/{args.pkl_tag}.pkl'
        assert os.path.exists(exp_path)
        with open(exp_path, 'rb') as f:
            self.exp = pkl.load(f)

        self.val_dir = f'{self.exp.work_dir}/val'
        if os.path.exists(self.val_dir):
            shutil.rmtree(self.val_dir)
        os.makedirs(self.val_dir)


    def run(self):
        self.plot_train_objs()
        # self.vis_embeddings()
        self.finetune_val()

    def vis_embeddings(self):
        key = jax.random.PRNGKey(0)
        k1, key = jax.random.split(key, 2)
        pair_sampler = ImagePairSampler(self.exp.image_paths, num_rgb_sample=self.exp.cfg.num_rgb_sample, key=k1)
        images = jnp.stack([
            sampler.image_square for sampler in pair_sampler.samplers])
        images_th = torch.from_numpy(np.array(images)).cuda().permute(0, 3, 1, 2)
        resnet18_pretrained_th = torchvision.models.resnet18().cuda().eval()
        resnet18_pretrained_th.fc = nn.Identity() # Remove classifier
        pretrained_zs = resnet18_pretrained_th(images_th).detach().cpu().numpy()

        resnet = ResNet18(num_classes=self.exp.meta_icnn.bottleneck_size // 2)
        meta_zs = resnet.apply(
            {'params': self.exp.meta_params['resnet'],
             'batch_stats': self.exp.meta_batch_stats['resnet']},
            images, train=False)

        full_images = [sampler.image for sampler in pair_sampler.samplers]

        def plot_tsne(zs, loc):
            tsne = TSNE(n_components=2, perplexity=5, init='pca', method='exact').fit_transform(zs)
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            for full_image, x, y in zip(full_images, tsne[:,0], tsne[:,1]):
                full_image = full_image.resize((40,40))
                full_image = np.asarray(full_image) / 255.
                ax.add_artist(
                    AnnotationBbox(
                        OffsetImage(full_image, zoom=.5), (x, y), xycoords='data',
                        frameon=False,
                    )
                )
                ax.update_datalim(np.column_stack([x, y]))
                ax.autoscale()

            fname = f'{self.exp.work_dir}/{loc}.pdf'
            print(f'Saving to {fname}')
            fig.tight_layout()
            ax.grid(False)
            ax.axis('off')
            fig.savefig(fname, transparent=True)
            plt.close(fig)

        plot_tsne(pretrained_zs, 'tsne-pretrained')
        plot_tsne(meta_zs, 'tsne-meta')

    def plot_train_objs(self):
        log = pd.read_csv(f'{self.args.exp_root}/log.csv')
        fig, ax = plt.subplots(1, 1, figsize=(4,2.5))
        colors = plt.style.library['bmh']['axes.prop_cycle'].by_key()['color']
        ax.plot(log['iter'], log['val_dual_obj'])
        fname = f'{self.exp.work_dir}/train-objs.pdf'
        print(f'Saving to {fname}')
        ax.set_xlabel('Train iteration')
        ax.set_ylabel('Dual objective')
        fig.tight_layout()
        fig.savefig(fname)
        plt.close(fig)


    def finetune_val(self):
        cfg = OmegaConf.load('conf/train_color_single.yaml')
        self.W_single = SingleWorkspace(cfg)
        self.W_single.pretrain_identity(num_iter=15001)
        self.D_init_params = copy.deepcopy(self.W_single.D_params)
        self.D_conj_init_params = copy.deepcopy(self.W_single.D_conj_params)

        fig, ax = plt.subplots(1, 1, figsize=(4,2))
        colors = plt.style.library['bmh']['axes.prop_cycle'].by_key()['color']
        fname = f'{self.exp.work_dir}/val-objs.pdf'
        print(f'Saving to {fname}')
        log_freq = 250

        sampler_pairs = []
        _, val_pairs = self.exp.load_val()
        for i, (X_path, Y_path) in enumerate(val_pairs):
            X_sampler, Y_sampler = ImageSampler(X_path), ImageSampler(Y_path)
            self.exp.plot(X_sampler, Y_sampler, loc=f'{self.val_dir}/{i:04d}_meta_init.png')
            sampler_pairs.append((X_sampler, Y_sampler))

        @jax.jit
        def meta_apply(X, Y):
            D_params_meta, D_conj_params_meta = self.exp.meta_icnn.apply(
                {'params': self.exp.meta_params, 'batch_stats': self.exp.meta_batch_stats},
                X, Y, train=False,
                unravel_fn=self.exp.unravel_icnn_params_fn)
            return D_params_meta, D_conj_params_meta

        num_train_iter = 2001
        times = defaultdict(list)
        dual_objs = defaultdict(list)
        for i, (X_sampler, Y_sampler) in enumerate(sampler_pairs):
            print(X_sampler.path, Y_sampler.path)
            hist_vanilla = self.W_single.train(
                X_sampler, Y_sampler, self.D_init_params, self.D_conj_init_params,
                save=False, plot=False, log_freq=log_freq,
                num_train_iter=num_train_iter)
            self.W_single.plot(X_sampler, Y_sampler, loc=f'{self.val_dir}/{i:04d}_vanilla_final.png')

            start = time.time()
            D_params_meta, D_conj_params_meta = meta_apply(
                X_sampler.image_square, Y_sampler.image_square)
            meta_model_time = time.time() - start

            hist_meta = self.W_single.train(
                X_sampler, Y_sampler, D_params_meta, D_conj_params_meta,
                save=False, plot=False, log_freq=log_freq,
                num_train_iter=num_train_iter)
            self.W_single.plot(X_sampler, Y_sampler, loc=f'{self.val_dir}/{i:04d}_meta_final.png')

            vanilla_dual_objs = hist_vanilla['dual_objs']
            meta_dual_objs = hist_meta['dual_objs']

            # Normalize
            all_objs = jnp.concatenate((vanilla_dual_objs, meta_dual_objs))
            max_obj = all_objs.max()
            min_obj = all_objs.min()
            obj_range = max_obj - min_obj
            vanilla_dual_objs = (vanilla_dual_objs - min_obj) / obj_range
            meta_dual_objs = (meta_dual_objs - min_obj) / obj_range

            iters = hist_vanilla['iters']
            ax.plot(iters, vanilla_dual_objs, color=colors[0], alpha=0.7, zorder=1)
            ax.plot(iters, meta_dual_objs, color=colors[1], alpha=0.7, zorder=2)
            ax.set_xlim(0, None)

            ax.set_ylabel('Dual Objective')
            ax.set_xlabel('W2GN Iterations')
            fig.tight_layout()
            fig.savefig(fname, transparent=True)
            os.system(f'pdfcrop {fname} {fname}')

            if i > 0:
                # Track cumulative stats, could be cleaned up
                idx_1k = 4
                idx_2k = 8
                assert hist_vanilla['iters'][idx_1k] == 1000
                assert hist_vanilla['iters'][idx_2k] == 2000
                vanilla_init_time = hist_vanilla['times'][0]
                meta_init_time = hist_meta['times'][0]
                times['vanilla_1k'].append(hist_vanilla['times'][idx_1k] - vanilla_init_time)
                times['vanilla_2k'].append(hist_vanilla['times'][idx_2k] - vanilla_init_time)
                times['meta_0'].append(meta_model_time)
                times['meta_1k'].append(hist_meta['times'][idx_1k] - meta_init_time + meta_model_time)
                times['meta_2k'].append(hist_meta['times'][idx_2k] - meta_init_time + meta_model_time)

                dual_objs['meta_0'].append(meta_dual_objs[0])
                dual_objs['meta_1k'].append(meta_dual_objs[idx_1k])
                dual_objs['meta_2k'].append(meta_dual_objs[idx_2k])
                dual_objs['vanilla_0'].append(vanilla_dual_objs[0])
                dual_objs['vanilla_1k'].append(vanilla_dual_objs[idx_1k])
                dual_objs['vanilla_2k'].append(vanilla_dual_objs[idx_2k])

            print('=====')
            for k, vals in sorted(times.items()):
                vals = np.array(vals)
                duals = np.array(dual_objs[k])
                print(fr'{k} & {vals.mean():.2e}s \pm {vals.std():.2e}s & {duals.mean():.2e} \pm {duals.std():.2e} \\')
            print('=====')

            if i > args.num_test_samples:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_root', type=str)
    parser.add_argument('--pkl_tag', type=str, default='latest')
    parser.add_argument('--num_test_samples', type=int, default=20)
    args = parser.parse_args()

    W_eval = Workspace(args)
    W_eval.run()
