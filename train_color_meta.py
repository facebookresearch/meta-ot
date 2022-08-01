#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# https://github.com/iamalexkorotin/Wasserstein2GenerativeNetworks/blob/master/notebooks/W2GN_color_transfer.ipynb


import hydra
from hydra.utils import instantiate

import csv
import copy
import glob
import os
import random
import functools

import pickle as pkl

from typing import Callable

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

import matplotlib.pyplot as plt
from PIL import Image
import time

from collections import namedtuple

from meta_ot.models import ICNN
from meta_ot.data import ImageSampler, ImagePairSampler
from meta_ot.utils import RunningAverageMeter
from train_color_single import push_grad, push_image
from meta_ot import conjugate

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Neutral', call_pdb=1)

DIR = os.path.dirname(os.path.realpath(__file__))


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.train_iter = 0
        self.output = 0


        self.data_dir = DIR + '/data/' + self.cfg.data
        self.pairs_f = self.data_dir + '/pairs.txt'
        self.val_images, self.val_pairs = self.load_val()
        self.debug_image_paths = self.val_pairs[0]

        assert os.path.exists(self.data_dir)
        self.image_paths = glob.glob(self.data_dir + '/*.jpg')
        n_image = len(self.image_paths)
        assert n_image > 0

        self.key = jax.random.PRNGKey(self.cfg.seed)

        self.D = ICNN(dim_hidden=[128])
        self.D_conj = ICNN(dim_hidden=[128]) # Assume they have the same structure

        # Initialize an ICNN to get parameter count and the unraveling function
        k1, k2, self.key = jax.random.split(self.key, 3)
        self.input_dim = 3
        self.D_init_params = self.D.init(k1, jnp.ones(self.input_dim))['params']
        num_icnn_params = sum(x.size for x in jax.tree_leaves(self.D_init_params))
        print(f'ICNN #params: {num_icnn_params}')
        _, self.unravel_icnn_params_fn = jax.flatten_util.ravel_pytree(self.D_init_params)

        self.meta_icnn = instantiate(self.cfg.meta_icnn, num_icnn_params=num_icnn_params)
        k1, self.key = jax.random.split(self.key, 2)
        t = jnp.ones((self.cfg.meta_batch_size, 224, 224, 3))
        self.meta_params = self.meta_icnn.init(k1, t, t)['params']
        self.meta_batch_stats = self.meta_icnn.init(k1, t, t)['batch_stats']
        self.num_meta_params = sum(x.size for x in jax.tree_leaves(self.meta_params))
        print(f'Meta ICNN #params: {self.num_meta_params}')

        self.conj_solver = conjugate.Solver(self.D, tol=1e-4, projection_name='unit_box')


    def D_batch(self, params, data):
        return jax.vmap(lambda X: self.D.apply({'params': params}, X))(data)

    def D_conj_batch(self, params, data):
        return jax.vmap(lambda X: self.D_conj.apply({'params': params}, X))(data)


    def pretrain_identity(self, sampler):
        # Pre-train to satisfy push(D, x) \approx push(D_conj, x) \approx x
        k1, self.key = jax.random.split(self.key, 2)
        opt = optax.adam(learning_rate=self.cfg.pretrain_lr)
        if self.cfg.max_grad_norm:
            opt = optax.chain(optax.clip_by_global_norm(self.cfg.max_grad_norm), opt)
        meta_state = train_state.TrainState.create(
            apply_fn=self.meta_icnn.apply, params=self.meta_params, tx=opt)

        def pretrain_loss_fn_single(D_params_flat, D_conj_params_flat,
                                    X_square, Y_square, X):
            D_params = self.unravel_icnn_params_fn(D_params_flat)
            D_conj_params = self.unravel_icnn_params_fn(D_conj_params_flat)

            push_X = push_grad(self.D, D_params, X)
            loss = ((push_X-X)**2).sum(axis=1).mean()
            loss += self.cfg.l2_penalty*(D_params_flat**2).mean()

            push_X = push_grad(self.D_conj, D_conj_params, X)
            loss += ((push_X-X)**2).sum(axis=1).mean()
            loss += self.cfg.l2_penalty*(D_conj_params_flat**2).mean()

            return loss

        def pretrain_loss_fn_batch(meta_params, meta_batch_stats,
                                   X_squares, Y_squares, X):
            (D_params_flat, D_conj_params_flat), model_state = \
              self.meta_icnn.apply(
                  {'params': meta_params, 'batch_stats': meta_batch_stats},
                  X_squares, Y_squares, mutable=['batch_stats'], train=True)
            losses = jax.vmap(
                pretrain_loss_fn_single, in_axes=(0,0,0,0,None)
            )(D_params_flat, D_conj_params_flat, X_squares, Y_squares, X)
            return jnp.mean(losses), model_state['batch_stats']

        @jax.jit
        def pretrain_update(meta_state, meta_batch_stats, X_squares, Y_squares, key):
            X = jax.random.uniform(key, [self.cfg.inner_batch_size, self.input_dim])
            X = 2.*(X-.5) + .5
            grad_fn = jax.value_and_grad(pretrain_loss_fn_batch, has_aux=True)
            (loss, meta_batch_stats), grads = grad_fn(
                meta_state.params, meta_batch_stats, X_squares, Y_squares, X)
            return loss, meta_state.apply_gradients(grads=grads), meta_batch_stats

        loss_meter = RunningAverageMeter()
        for i in range(self.cfg.max_num_pretrain_iter):
            X_samplers, Y_samplers, X_squares, Y_squares, X_fulls, Y_fulls = \
                sampler.sample_image_pair_batch(self.cfg.meta_batch_size)
            k1, self.key = jax.random.split(self.key, 2)
            loss, meta_state, self.meta_batch_stats = pretrain_update(
                meta_state, self.meta_batch_stats, X_squares, Y_squares, k1)
            loss_meter.update(loss.item())
            if i % 1000 == 0:
                print(f'iter={i} pretrain_loss={loss_meter.avg:.2e}')
                self.meta_params = meta_state.params
                self.plot(X_samplers[0], Y_samplers[0], loc='latest-train.png')
            if loss_meter.avg < self.cfg.pretrain_loss_threshold:
                break

        self.meta_params = meta_state.params


    def run(self):
        logf, writer = self._init_logging()

        k1, k2, self.key = jax.random.split(self.key, 3)
        sampler = ImagePairSampler(self.image_paths, num_rgb_sample=self.cfg.num_rgb_sample, key=k1)
        debug_sampler = ImagePairSampler(self.debug_image_paths, num_rgb_sample=self.cfg.num_rgb_sample, key=k2)

        val_samplers = []
        for i, (X_path, Y_path) in enumerate(self.val_pairs):
            X_sampler, Y_sampler = ImageSampler(X_path), ImageSampler(Y_path)
            val_samplers.append((X_sampler, Y_sampler))

        random.seed(0)
        X_samplers, Y_samplers, X_squares, Y_squares, X_fulls, Y_fulls = \
            sampler.sample_image_pair_batch(
                self.cfg.meta_batch_size, self.val_pairs)
        fixed_train_samplers = list(zip(*(X_samplers, Y_samplers)))

        self.pretrain_identity(sampler)

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0,
            peak_value=self.cfg.lr,
            warmup_steps=5000,
            decay_steps=self.cfg.num_train_iter,
            end_value=0.0,
        )
        opt = optax.adam(learning_rate=lr_schedule)
        if self.cfg.max_grad_norm:
            opt = optax.chain(optax.clip_by_global_norm(self.cfg.max_grad_norm), opt)
        meta_state = train_state.TrainState.create(
            apply_fn=self.meta_icnn.apply, params=self.meta_params, tx=opt)

        def loss_fn_single(key, D_params_flat, D_conj_params_flat,
                           X_square, Y_square, X_full, Y_full):
            D_params = self.unravel_icnn_params_fn(D_params_flat)
            D_conj_params = self.unravel_icnn_params_fn(D_conj_params_flat)

            k1, k2, key = jax.random.split(key, 3)
            X = X_full[jax.random.choice(
                k1, len(X_full), shape=[self.cfg.inner_batch_size])]
            Y = Y_full[jax.random.choice(
                k2, len(Y_full), shape=[self.cfg.inner_batch_size])]

            # Approximate dual (correlation) objective
            X_hat = push_grad(self.D_conj, D_conj_params, Y)
            X_hat_detach = jax.lax.stop_gradient(X_hat)
            dual_loss = (self.D_batch(D_params, X) + \
                jax.vmap(jnp.dot)(X_hat_detach, Y) - \
                self.D_batch(D_params, X_hat_detach)).mean()

            # Cycle regularization
            Y_hat = push_grad(self.D, D_params, X)
            cycle_loss = \
              ((push_grad(self.D, D_params, X_hat) - Y) ** 2).mean() + \
              ((push_grad(self.D_conj, D_conj_params, Y_hat) - X) ** 2).mean()

            loss = dual_loss + self.cfg.cycle_loss_weight * cycle_loss + \
              self.cfg.l2_penalty*(D_params_flat**2).mean() + \
              self.cfg.l2_penalty*(D_conj_params_flat**2).mean()
            return loss, (dual_loss, cycle_loss)

        def loss_fn_batch(meta_params, meta_batch_stats, key, X_squares, Y_squares,
                          X_fulls, Y_fulls):
            (D_params_flat, D_conj_params_flat), model_state = \
              self.meta_icnn.apply(
                  {'params': meta_params, 'batch_stats': meta_batch_stats},
                  X_squares, Y_squares, mutable=['batch_stats'], train=True)
            losses, (corr_losses, cycle_losses) = jax.vmap(loss_fn_single, in_axes=(None,0,0,0,0,0,0))(
                key, D_params_flat, D_conj_params_flat, X_squares, Y_squares, X_fulls, Y_fulls)
            loss = jnp.mean(losses)
            corr_loss = jnp.mean(corr_losses)
            cycle_loss = jnp.mean(cycle_losses)
            return loss, (corr_loss, cycle_loss, model_state['batch_stats'])

        @jax.jit
        def update_batch(key, meta_state, meta_batch_stats,
                         X_squares, Y_squares, X_fulls, Y_fulls):
            grad_fn = jax.value_and_grad(loss_fn_batch, has_aux=True)
            (loss, (corr_loss, cycle_loss, batch_stats)), grads = grad_fn(
                meta_state.params, meta_batch_stats, key, X_squares, Y_squares, X_fulls, Y_fulls)
            return loss, corr_loss, cycle_loss, meta_state.apply_gradients(grads=grads), batch_stats

        loss_meter = RunningAverageMeter()
        corr_loss_meter = RunningAverageMeter()
        cycle_loss_meter = RunningAverageMeter()
        best_dual = None
        start_time = time.time()
        for i in range(int(self.cfg.num_train_iter)):
            X_samplers, Y_samplers, X_squares, Y_squares, X_fulls, Y_fulls = \
                sampler.sample_image_pair_batch(
                    self.cfg.meta_batch_size, self.val_pairs)
            k1, self.key = jax.random.split(self.key, 2)
            loss, corr_loss, cycle_loss, meta_state, self.meta_batch_stats = \
                update_batch(
                    k1, meta_state, self.meta_batch_stats,
                    X_squares, Y_squares, X_fulls, Y_fulls)
            loss_meter.update(loss.item())
            corr_loss_meter.update(corr_loss.item())
            cycle_loss_meter.update(cycle_loss.item())

            if i % 1000 == 0:
                self.meta_params = meta_state.params
                train_dual_obj = self.dual_objs(fixed_train_samplers)
                # val_dual_obj = self.val_dual_objs(val_samplers)
                print(f'iter={i} train_loss={loss_meter.avg:.2e} corr_loss={corr_loss_meter.avg:.2e} cycle_loss={cycle_loss_meter.avg:.2e} train_dual_obj={train_dual_obj:.2e}')
                writer.writerow({
                    'iter': i,
                    'time': time.time()-start_time,
                    'loss': loss_meter.avg,
                    'corr_loss': corr_loss_meter.avg,
                    'cycle_loss': cycle_loss_meter.avg,
                    'dual_obj': train_dual_obj,
                })
                logf.flush()
                self.plot(X_samplers[0], Y_samplers[0], loc='latest-train.png')
                X_sampler, Y_sampler = debug_sampler.samplers
                self.plot(X_sampler, Y_sampler, loc='latest-val.png')
                self.save(tag='latest')
                if best_dual is None or train_dual_obj > best_dual:
                    best_dual = train_dual_obj
                    self.save(tag='best')


    def dual_objs(self, samplers):
        # TODO: Better jit this?
        vs = []
        for (X_sampler, Y_sampler) in samplers:
            D_params, D_conj_params = self.meta_icnn.apply(
                {'params': self.meta_params, 'batch_stats': self.meta_batch_stats},
                X_sampler.image_square, Y_sampler.image_square, train=False,
                unravel_fn=self.unravel_icnn_params_fn)
            vs.append(self.dual_obj(X_sampler, Y_sampler, D_params).item())
        return np.mean(vs)


    def dual_obj(self, X_sampler, Y_sampler, D_params, num_samples=1024, seed=0):
        # TODO: Better jit this so it's not jitted every time it's called?
        def D_conj(y):
            out = self.conj_solver.solve(D_params, y)
            return out.grad
        D_conj_batch = jax.jit(jax.vmap(D_conj))
        D_batch = jax.jit(functools.partial(self.D_batch, D_params))

        key = jax.random.PRNGKey(seed)
        k1, k2, key = jax.random.split(key, 3)
        X = X_sampler.sample(k1, num_samples)
        Y = Y_sampler.sample(k2, num_samples)

        X_hat = jax.lax.stop_gradient(D_conj_batch(Y))

        dual_obj = D_batch(X).mean() + \
          (jax.vmap(jnp.dot)(X_hat, Y) - D_batch(X_hat)).mean()

        return -dual_obj


    def plot(self, X_sampler, Y_sampler, loc, crop_height=400):
        D_params, D_conj_params = self.meta_icnn.apply(
            {'params': self.meta_params, 'batch_stats': self.meta_batch_stats},
            X_sampler.image_square, Y_sampler.image_square, train=False,
            unravel_fn=self.unravel_icnn_params_fn)

        w, h = X_sampler.image.size
        im1 = X_sampler.image.resize((int(crop_height * w/h), crop_height), Image.ANTIALIAS)
        w, h = Y_sampler.image.size
        im2 = Y_sampler.image.resize((int(crop_height * w/h), crop_height), Image.ANTIALIAS)
        im1_push = push_image(self.D, D_params, im1)
        im2_push = push_image(self.D_conj, D_conj_params, im2)
        im1 = jnp.array(im1)
        im2 = jnp.array(im2)
        out =  jnp.concatenate((im1, im2, im1_push, im2_push), axis=1)
        plt.imsave(loc, out)


    def _init_logging(self):
        logf = open('log.csv', 'a')
        fieldnames = [
            'iter', 'time', 'loss', 'corr_loss', 'cycle_loss', 'dual_obj']
        writer = csv.DictWriter(logf, fieldnames=fieldnames)
        if os.stat('log.csv').st_size == 0:
            writer.writeheader()
            logf.flush()
        return logf, writer


    def load_val(self):
        val_images = []
        val_pairs = []
        with open(self.pairs_f, 'rb') as f:
            for line in f.readlines():
                im1, im2 = line.decode().strip().split(',')
                im1 = f'{self.data_dir}/{im1}.jpg'
                im2 = f'{self.data_dir}/{im2}.jpg'
                val_images += [im1, im2]
                assert os.path.exists(im1) and os.path.exists(im2)
                val_pairs.append((im1, im2))
        return val_images, val_pairs

    def save(self, tag='latest'):
        path = os.path.join(self.work_dir, f'{tag}.pkl')
        with open(path, 'wb') as f:
            pkl.dump(self, f)


    def __getstate__(self):
        d = copy.copy(self.__dict__)
        # For pickling: cannot save functions as part of the object
        del d['unravel_icnn_params_fn']
        return d


    def __setstate__(self, d):
        self.__dict__ = d
        _, self.unravel_icnn_params_fn = jax.flatten_util.ravel_pytree(self.D_init_params)



@hydra.main(config_path='conf', config_name='train_color_meta.yaml')
def main(cfg):
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    from train_color_meta import Workspace as W # For pickling
    fname = os.getcwd() + '/latest.pkl'
    workspace = W(cfg)

    workspace.run()


if __name__ == '__main__':
    main()

