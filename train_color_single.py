#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# https://github.com/iamalexkorotin/Wasserstein2GenerativeNetworks/blob/master/notebooks/W2GN_color.ipynb


import hydra
from hydra.utils import instantiate


import functools

import copy
import time
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')

import numpy as np

import pickle as pkl
from PIL import Image

import torch

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

import matplotlib.pyplot as plt

from meta_ot.models import ICNN
from meta_ot.utils import RunningAverageMeter
from meta_ot.data import ImageSampler
from meta_ot import conjugate


import os
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

        self.key = jax.random.PRNGKey(self.cfg.seed)

        self.D = ICNN(dim_hidden=[128])
        self.D_conj = ICNN(dim_hidden=[128])

        k1, k2, self.key = jax.random.split(self.key, 3)
        self.input_dim = 3
        self.D_params = self.D.init(k1, jnp.ones(self.input_dim))['params']
        self.D_conj_params = self.D_conj.init(k2, jnp.ones(self.input_dim))['params']

        self.conj_solver = conjugate.Solver(self.D, tol=1e-4, projection_name='unit_box')


    def D_batch(self, params, data):
        return jax.vmap(lambda X: self.D.apply({'params': params}, X))(data)

    def D_conj_batch(self, params, data):
        return jax.vmap(lambda X: self.D_conj.apply({'params': params}, X))(data)

    def pretrain_identity(self, num_iter=None):
        # Pre-train to satisfy push(D, x) \approx x
        k1, self.key = jax.random.split(self.key, 2)
        pre_D_opt = optax.adam(learning_rate=1e-3, b1=0.8, b2=0.9)
        state_D = train_state.TrainState.create(apply_fn=self.D.apply, params=self.D_params, tx = pre_D_opt)

        def pretrain_loss_fn(D_params, x):
            push_x = push_grad(self.D, D_params, x)

            flat_D_params, _ = jax.flatten_util.ravel_pytree(D_params)
            loss = ((push_x-x)**2).sum(axis=1).mean() + \
              self.cfg.l2_penalty*(flat_D_params**2).mean()
            return loss


        @jax.jit
        def pretrain_update(state, key):
            X = jax.random.uniform(key, [self.cfg.batch_size, self.input_dim])
            X = 2.*(X-.5) + .5
            grad_fn = jax.value_and_grad(pretrain_loss_fn)
            loss, grads = grad_fn(state.params, X)
            return loss, state.apply_gradients(grads=grads)

        if num_iter is None:
            num_iter = 15001
        for i in range(num_iter):
            k1, self.key = jax.random.split(self.key, 2)
            loss, state_D = pretrain_update(state_D, k1)
            if i % 1000 == 0:
                print(f'iter={i} pretrain_loss={loss:.2e}')
                # print_param_stats(state_D.params)

        self.D_params = state_D.params
        self.D_conj_params = copy.deepcopy(state_D.params)


    def run(self):
        if self.cfg.image_1.startswith('/'):
            # Use absolute paths
            X_sampler = ImageSampler(self.cfg.image_1)
            Y_sampler = ImageSampler(self.cfg.image_2)
        else:
            # Use relative paths from the repo directory
            X_sampler = ImageSampler(DIR + '/' + self.cfg.image_1)
            Y_sampler = ImageSampler(DIR + '/' + self.cfg.image_2)

        self.pretrain_identity()
        self.train(X_sampler, Y_sampler)

    def train(self, X_sampler, Y_sampler, init_D_params=None, init_D_conj_params=None,
              save=True, plot=True, log_freq=1000, num_train_iter=None,
              lr=1e-3):
        # Sample data for dual computation
        self.key = jax.random.PRNGKey(self.cfg.seed)
        k1, k2, self.key = jax.random.split(self.key, 3)
        X_dual = X_sampler.sample(k1, self.cfg.batch_size)
        Y_dual = Y_sampler.sample(k2, self.cfg.batch_size)
        dual_fn = jax.jit(lambda D_params: self.dual_obj(X_dual, Y_dual, D_params))

        if init_D_params is not None:
            self.D_params = init_D_params

        if init_D_conj_params is not None:
            self.D_conj_params = init_D_conj_params

        if num_train_iter is None:
            num_train_iter = self.cfg.num_train_iter

        D_opt = optax.adam(learning_rate=lr)
        D_conj_opt = optax.adam(learning_rate=lr)
        state_D = train_state.TrainState.create(
            apply_fn=self.D.apply, params=self.D_params, tx=D_opt)
        state_D_conj = train_state.TrainState.create(
            apply_fn=self.D_conj.apply, params=self.D_conj_params, tx=D_conj_opt)

        def train_loss_fn(D_params, D_conj_params, X, Y):
            # Approximate dual (correlation) objective
            X_hat = push_grad(self.D_conj, D_conj_params, Y)
            X_hat_detach = jax.lax.stop_gradient(X_hat)
            dual_loss = (self.D_batch(D_params, X) + \
                jax.vmap(jnp.dot)(X_hat_detach, Y) - \
                self.D_batch(D_params, X_hat_detach)).mean()

            # Cycle Regularization
            Y_hat = push_grad(self.D, D_params, X)
            cycle_loss = \
              ((push_grad(self.D, D_params, X_hat) - Y) ** 2).mean() + \
              ((push_grad(self.D_conj, D_conj_params, Y_hat) - X) ** 2).mean()

            flat_D_params, _ = jax.flatten_util.ravel_pytree(D_params)
            flat_D_conj_params, _ = jax.flatten_util.ravel_pytree(D_conj_params)

            loss = dual_loss + self.cfg.cycle_loss_weight * cycle_loss + \
                self.cfg.l2_penalty*(flat_D_params**2).mean() + \
                self.cfg.l2_penalty*(flat_D_conj_params**2).mean()
            return loss, [dual_loss, cycle_loss]


        @jax.jit
        def update(key, state_D, state_D_conj):
            k1, k2, key = jax.random.split(key, 3)
            X = X_sampler.sample(k1, self.cfg.batch_size)
            Y = Y_sampler.sample(k2, self.cfg.batch_size)
            grad_fn = jax.value_and_grad(train_loss_fn, argnums=(0,1), has_aux=True)
            (loss, (corr_loss, cycle_loss)), (grads_D, grads_D_conj) = grad_fn(
                state_D.params, state_D_conj.params, X, Y)
            return loss, corr_loss, cycle_loss, state_D.apply_gradients(grads=grads_D), \
                state_D_conj.apply_gradients(grads=grads_D_conj)

        hist = defaultdict(list)
        loss_meter = RunningAverageMeter()
        corr_loss_meter = RunningAverageMeter()
        cycle_loss_meter = RunningAverageMeter()

        for i in range(num_train_iter):
            k1, self.key = jax.random.split(self.key, 2)
            loss, corr_loss, cycle_loss, state_D, state_D_conj = update(k1, state_D, state_D_conj)
            loss_meter.update(loss.item())
            corr_loss_meter.update(corr_loss.item())
            cycle_loss_meter.update(cycle_loss.item())
            if i % log_freq == 0:
                dual_obj = dual_fn(self.D_params)
                print(f'iter={i} train_loss={loss_meter.avg:.2e} corr_loss={corr_loss_meter.avg:.2e} cycle_loss={cycle_loss_meter.avg:.2e} dual_obj={dual_obj:.2e}')
                self.D_params = state_D.params
                self.D_conj_params = state_D_conj.params
                hist['dual_objs'].append(dual_obj.item())
                hist['times'].append(time.time())
                hist['iters'].append(i)
                if plot:
                    self.plot(X_sampler, Y_sampler, loc='latest.png')
                if save:
                    self.save()

        hist['dual_objs'] = jnp.array(hist['dual_objs'])
        return hist


    def dual_obj(self, X, Y, D_params, num_samples=1024):
        def D_conj(y):
            out = self.conj_solver.solve(D_params, y)
            return out.grad

        D_conj_batch = jax.vmap(D_conj)
        D_batch = functools.partial(self.D_batch, D_params)

        X_hat = jax.lax.stop_gradient(D_conj_batch(Y))

        dual_obj = D_batch(X).mean() + \
          (jax.vmap(jnp.dot)(X_hat, Y) - D_batch(X_hat)).mean()

        return -dual_obj


    def plot(self, X_sampler, Y_sampler, loc, crop_height=400):
        w, h = X_sampler.image.size
        im1 = X_sampler.image.resize((int(crop_height * w/h), crop_height), Image.ANTIALIAS)
        w, h = Y_sampler.image.size
        im2 = Y_sampler.image.resize((int(crop_height * w/h), crop_height), Image.ANTIALIAS)
        im1_push = push_image(self.D, self.D_params, im1)
        im2_push = push_image(self.D_conj, self.D_conj_params, im2)
        im1 = jnp.array(im1)
        im2 = jnp.array(im2)
        out =  jnp.concatenate((im1, im2, im1_push, im2_push), axis=1)
        plt.imsave(loc, out)


    def save(self, tag='latest'):
        path = os.path.join(self.work_dir, f'{tag}.pkl')
        with open(path, 'wb') as f:
            pkl.dump(self, f)


def push_grad(net, params, data):
    return jax.vmap(lambda X: jax.grad(net.apply, argnums=1)(
        {'params': params}, X))(data)


def push_image(net, params, image, batch_size=5000):
    push_jit = jax.jit(lambda X: push_grad(net, params, X))
    X = np.asarray(image).transpose(2, 0, 1)
    orig_shape = X.shape
    X = (X.reshape(3, -1) / 255.).T
    X_pushed = np.zeros_like(X)
    pos = 0;
    while pos < len(X):
        X_pushed[pos:pos+batch_size] = push_jit(X[pos:pos+batch_size])
        pos += batch_size

    im_X_pushed = (
        np.clip(X_pushed.T.reshape(orig_shape).transpose(1, 2, 0), 0, 1) * 255
    ).astype(np.uint8)
    return im_X_pushed




@hydra.main(config_path='conf', config_name='train_color_single.yaml')
def main(cfg):
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    from train_color_single import Workspace as W # For pickling
    fname = os.getcwd() + '/latest.pkl'
    workspace = W(cfg)

    workspace.run()


if __name__ == '__main__':
    main()
