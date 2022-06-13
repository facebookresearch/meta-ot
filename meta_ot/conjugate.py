# Copyright (c) Meta Platforms, Inc. and affiliates.

import functools

import jax
import jax.numpy as jnp

import copy
from dataclasses import dataclass
from collections import namedtuple
from typing import Optional, Callable


ConjStatus = namedtuple("ConjStatus", "val grad num_iter")

projection_fns = {
    'identity': lambda x: x,
    'unit_box': lambda x: x.clip(0., 1.),
}

def get_projection_fn(name):
    if name in projection_fns.keys():
        return projection_fns[name]
    else:
        raise NotImplementedError()

@dataclass
class Solver:
    D: 'Agent'
    min_iter: int = 0
    max_iter: int = 100
    tol: float = 1e-5
    initial_step_size: float = 1.
    max_linesearch_iter: int = 30
    armijo_gamma: float = 0.1
    linesearch_decay: float = 2
    damp: float = 1e-2
    normalize_step: bool = False
    verbose: bool = False
    projection_name: str = 'identity'

    def __post_init__(self):
        self.projection_fn = get_projection_fn(self.projection_name)

    def conj_min_obj(self, x, D_params, y):
        # f^c(y) = min_x f(x) - y^T x
        return self.D.apply({'params': D_params}, x) - x.dot(y)

    def loop_cond(self, v):
        x, last_obj, it, err, last_step_size = v
        return (it < self.min_iter) | ((it < self.max_iter) & (err > self.tol))

    def loop_body(self, v, D_params, y):
        n = y.size
        x, last_obj, it, err, last_step_size = v
        conj_min_obj = functools.partial(
            self.conj_min_obj, y=y, D_params=D_params)
        g = jax.grad(conj_min_obj)(x)
        step = g

        # step_size = self.initial_step_size
        step_size = jnp.minimum(self.initial_step_size, last_step_size * self.linesearch_decay)
        proposal = self.projection_fn(x - step_size * step)
        proposal_obj = conj_min_obj(proposal)

        ls_v = (step_size, 0, proposal, proposal_obj)
        linesearch_loop_cond = functools.partial(
            self.linesearch_loop_cond, g=g, last_obj=last_obj,
            last_x=x)
        linesearch_loop_body = functools.partial(
            self.linesearch_loop_body, last_x=x, step=step,
            conj_min_obj=conj_min_obj)
        step_size, step_iter, new_x, new_obj = jax.lax.while_loop(
            linesearch_loop_cond, linesearch_loop_body, ls_v)
        err = ((x - new_x)**2).mean()
        return new_x, new_obj, it + 1, err, step_size

    def linesearch_loop_cond(self, v, g, last_obj, last_x):
        step_size, step_iter, proposal, proposal_obj = v
        armijo_val = (last_obj - proposal_obj) / g.dot(last_x - proposal)
        return (armijo_val < self.armijo_gamma) & \
            (step_iter < self.max_linesearch_iter)

    def linesearch_loop_body(self, v, last_x, step, conj_min_obj):
        step_size, step_iter, proposal, proposal_obj = v
        step_size /= self.linesearch_decay
        proposal = self.projection_fn(last_x - step_size * step)
        proposal_obj = conj_min_obj(proposal)
        step_iter += 1
        return (step_size, step_iter, proposal, proposal_obj)

    def solve(self, D_params, y):
        assert y.ndim == 1
        # x_init = jnp.full(y.shape, 0.)
        x_init = y
        v = (x_init, self.conj_min_obj(x=x_init, D_params=D_params, y=y), 0, 1., self.initial_step_size)

        loop_body = functools.partial(self.loop_body, D_params=D_params, y=y)
        loop_cond = lambda v: self.loop_cond(v)
        x, obj, n_iter, err, step_size = jax.lax.while_loop(loop_cond, loop_body, v)

        return ConjStatus(val=-obj, grad=x, num_iter=n_iter)

    def loop_body_debug(self, v, D_params, y):
        n = y.size
        x, last_obj, it, err, last_step_size = v
        conj_min_obj = functools.partial(
            self.conj_min_obj, y=y, D_params=D_params)
        g = jax.grad(conj_min_obj)(x)
        step = g

        step_size = jnp.minimum(self.initial_step_size, last_step_size * self.linesearch_decay)
        proposal = self.projection_fn(x - step_size * step)
        proposal_obj = conj_min_obj(proposal)

        linesearch_loop_cond = functools.partial(
            self.linesearch_loop_cond, g=g, last_obj=last_obj,
            last_x=x)
        linesearch_loop_body = functools.partial(
            self.linesearch_loop_body, last_x=x, step=step,
            conj_min_obj=conj_min_obj)
        ls_v = (step_size, 0, proposal, proposal_obj)
        step_size, step_iter, new_x, new_obj = ls_v
        while linesearch_loop_cond(ls_v):
            ls_v = linesearch_loop_body(ls_v)
            step_size, step_iter, new_x, new_obj = ls_v
        if self.verbose:
            print(f'  + n_ls_steps={step_iter} step_size={step_size:.2e}')
        err = ((x - new_x)**2).mean()
        return new_x, new_obj, it + 1, err, step_size

    def solve_debug(self, D_params, y):
        assert y.ndim == 1
        # x_init = jnp.full(y.shape, 0.)
        # x_init = jnp.array([0., 0.]) # TODO
        x_init = y
        v = (x_init, self.conj_min_obj(x=x_init, D_params=D_params, y=y), 0, 1., self.initial_step_size)

        loop_body = functools.partial(self.loop_body_debug, D_params=D_params, y=y)
        while self.loop_cond(v):
            x, obj, n_iter, err, step_size = v
            if self.verbose:
                print(f'[{n_iter}] obj={obj:.2e} err={err:.2e}')
            v = loop_body(v)
        x, obj, n_iter, err, step_size = v
        if self.verbose:
            print(f'[{n_iter}] obj={obj:.2e} err={err:.2e}')

        return ConjStatus(val=-obj, grad=x, num_iter=n_iter)

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        # For pickling: cannot save functions as part of the object
        del d['projection_fn']
        return d


    def __setstate__(self, d):
        self.__dict__ = d
        self.projection_fn = get_projection_fn(self.projection_name)
