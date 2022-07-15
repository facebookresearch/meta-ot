#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import hydra
from hydra.utils import instantiate

import numpy as np

import csv
import os
import pickle as pkl

import ot
import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.training import train_state
import optax

from ott.core import quad_problems, problems, sinkhorn
from ott.geometry import PointCloud

import matplotlib.pyplot as plt
plt.style.use('bmh')

import functools

from meta_ot import utils
from meta_ot import data


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

        if self.cfg.data in ['mnist', 'random', 'doodles', 'usps28']:
            na = nb = 784
            self.n_output = na
        elif self.cfg.data == 'world':
            na = 100
            nb = 10000
            self.n_output = na
        else:
            assert False

        self.key = jax.random.PRNGKey(self.cfg.seed)
        self.potential_model = instantiate(
            self.cfg.potential_model, n_output=self.n_output)
        init_key, self.key = jax.random.split(self.key)
        a_placeholder = jnp.zeros(na)
        b_placeholder = jnp.zeros(nb)
        self.params = self.potential_model.init(
            init_key, a_placeholder, b_placeholder)['params']


    def run(self):
        logf, writer = self._init_logging()

        tx = instantiate(self.cfg.optim)
        state = train_state.TrainState.create(
            apply_fn=self.potential_model.apply, params=self.params, tx=tx)

        if self.cfg.data == 'mnist':
            train_sampler = data.MNISTPairSampler(train=True, batch_size=self.cfg.batch_size, debug=False)
            self.geom = train_sampler.geom
        elif self.cfg.data == 'world':
            train_sampler = data.WorldPairSampler(batch_size=self.cfg.batch_size, debug=False)
            self.geom = train_sampler.geom
        elif self.cfg.data == 'usps28':
            train_sampler = data.USPSPairSampler(train=True, batch_size=self.cfg.batch_size, debug=False, reshape=True)
            self.geom = train_sampler.geom
        elif self.cfg.data == 'doodles':
            train_sampler = data.DoodlePairSampler(train=True, batch_size=self.cfg.batch_size, debug=False)
            self.geom = train_sampler.geom
        elif self.cfg.data == 'random':
            train_sampler = data.RandomSampler(batch_size=self.cfg.batch_size, debug=False)
            self.geom = train_sampler.geom
        else:
            assert False

        def loss_batch(params, batch):
            loss_fn = functools.partial(self.dual_obj_loss, params=params)
            loss = jax.vmap(loss_fn)(a=batch.a, b=batch.b)
            return jnp.mean(loss)

        @jax.jit
        def update(state, key):
            batch = train_sampler(key)
            grad_fn = jax.value_and_grad(loss_batch)
            loss, grads = grad_fn(state.params, batch)
            return loss, state.apply_gradients(grads=grads)

        loss_meter = utils.RunningAverageMeter()
        while self.train_iter < self.cfg.num_train_iter:
            k1, self.key = jax.random.split(self.key)
            loss, state = update(state, k1)
            loss_meter.update(loss.item())
            if self.train_iter % 100 == 0:
                print(f'[{self.train_iter}] train_loss={loss_meter.val:.2e}')
                self.params = state.params
                self.save()
                losses = []
            self.train_iter += 1

    def g_from_f(self, a, b, f):
        g = self.geom.update_potential(
            f, jnp.zeros_like(b),
            jnp.log(b), 0, axis=0)
        return g

    def pred_transport(self, a, b):
        f_pred = self.potential_model.apply({'params': self.params}, a, b)
        g_pred = self.g_from_f(a, b, f_pred)
        P = self.geom.transport_from_potentials(f_pred, g_pred)
        return P

    def dual_obj_from_f(self, a, b, f):
        g = self.g_from_f(a, b, f)
        g = jnp.where(jnp.isfinite(g), g, 0.)
        dual_obj = f.dot(a) + g.dot(b)
        return dual_obj

    def dual_obj_loss(self, a, b, params):
        f_pred = self.potential_model.apply({'params': params}, a, b)
        dual_value = self.dual_obj_from_f(a, b, f_pred)
        loss = -dual_value
        return loss

    def save(self, tag='latest'):
        path = os.path.join(self.work_dir, f'{tag}.pkl')
        with open(path, 'wb') as f:
            pkl.dump(self, f)

    def _init_logging(self):
        logf = open('log.csv', 'a')
        fieldnames = ['iter', 'loss', 'kl', 'ess']
        writer = csv.DictWriter(logf, fieldnames=fieldnames)
        if os.stat('log.csv').st_size == 0:
            writer.writeheader()
            logf.flush()
        return logf, writer


@hydra.main(config_path='conf', config_name='train_discrete')
def main(cfg):
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    from train_discrete import Workspace as W # For pickling
    fname = os.getcwd() + '/latest.pkl'
    # TODO: Fix resuming
    # if os.path.exists(fname):
    #     print(f'Resuming fom {fname}')
    #     with open(fname, 'rb') as f:
    #         workspace = pkl.load(f)
    # else:
    #     workspace = W(cfg)
    workspace = W(cfg)

    workspace.run()


if __name__ == '__main__':
    main()
