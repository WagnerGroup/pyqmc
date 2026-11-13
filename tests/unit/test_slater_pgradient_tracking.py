# MIT License
#
# Copyright (c) 2019-2024 The PyQMC Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

"""Slater.pgradient recomputes the AO values from its own copy of the configurations, which
updateinternals keeps up to date. After a series of moves it must agree with a fresh recompute."""
import numpy as np

import pyqmc.api as pyq
from pyqmc.wf.slater import Slater


def _moves(wf, configs, rng, nsweeps=3, scale=0.8):
    """Single-electron moves, partly accepted, through updateinternals as VMC does.
    The large steps make electrons cross the cell boundary in a periodic system."""
    nconf, nelec, _ = configs.configs.shape
    for _ in range(nsweeps):
        for e in range(nelec):
            newpos = configs.configs[:, e, :] + rng.normal(scale=scale, size=(nconf, 3))
            epos = configs.make_irreducible(e, newpos)
            accept = rng.random(nconf) < 0.6
            wf.updateinternals(e, epos, configs, mask=accept)
            configs.move(e, epos, accept)


def _check(wf, configs):
    _moves(wf, configs, np.random.default_rng(11))
    tracked = wf.pgradient()
    wf.recompute(configs)
    fresh = wf.pgradient()
    assert set(tracked) == set(fresh)
    for k in fresh:
        assert np.allclose(tracked[k], fresh[k], atol=1e-10), k


def test_pgradient_after_moves_pbc(H_pbc_sto3g_krks):
    mol, mf = H_pbc_sto3g_krks
    supercell = pyq.get_supercell(mol, S=(np.ones((3, 3)) - 2 * np.eye(3)))
    wf = Slater(supercell, mf, eval_gto_precision=1e-8)
    configs = pyq.initial_guess(supercell, 12)
    wf.recompute(configs)
    wrap_before = configs.wrap.copy()
    _check(wf, configs)
    assert np.any(configs.wrap != wrap_before)  # the test did exercise wrapping


def test_pgradient_after_moves_open(LiH_sto3g_rhf):
    mol, mf = LiH_sto3g_rhf
    wf = Slater(mol, mf)
    configs = pyq.initial_guess(mol, 12)
    wf.recompute(configs)
    _check(wf, configs)
