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

import pickle
import numpy as np
import pyqmc.api as pyq
import pyqmc.rng
from pyqmc.method.linemin import correlated_compute_worker


class PickleClient:
    """Runs each task in-process on a pickled copy of its arguments, as a process
    pool would. A Generator sent to several tasks would replay in each copy."""

    class Future:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    def submit(self, f, *args, **kwargs):
        args, kwargs = pickle.loads(pickle.dumps((args, kwargs)))
        return self.Future(f(*args, **kwargs))


def _vmc_ecp(mol, wf, rng):
    configs = pyq.initial_guess(mol, 20, rng=rng)
    df, _ = pyq.vmc(
        wf,
        configs,
        nblocks=2,
        nsteps_per_block=2,
        accumulators={"energy": pyq.EnergyAccumulator(mol)},
        rng=rng,
    )
    return df["energyecp"]


def test_reproducible(H2_ccecp_rhf):
    """rng=s fixes every draw, including the ECP rotations, and pyqmc leaves
    numpy's global random state alone."""
    mol, mf = H2_ccecp_rhf
    wf, _ = pyq.generate_wf(mol, mf)
    global_state = np.random.get_state()[1].copy()
    a = _vmc_ecp(mol, wf, pyqmc.rng.get_rng(3))
    b = _vmc_ecp(mol, wf, pyqmc.rng.get_rng(3))
    c = _vmc_ecp(mol, wf, pyqmc.rng.get_rng(4))
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)
    assert np.array_equal(global_state, np.random.get_state()[1])


def test_partitions_draw_different_streams(H2_ccecp_rhf):
    """Partitions that start from identical walkers end up in different places."""
    mol, mf = H2_ccecp_rhf
    wf, _ = pyq.generate_wf(mol, mf)
    npartitions, nper = 3, 10
    one = pyq.initial_guess(mol, nper, rng=0)
    configs = pyq.initial_guess(mol, npartitions * nper, rng=0)
    configs.configs[:] = np.tile(one.configs, (npartitions, 1, 1))
    _, configs = pyq.vmc(
        wf,
        configs,
        nblocks=1,
        nsteps_per_block=1,
        accumulators={},
        client=PickleClient(),
        npartitions=npartitions,
        rng=0,
    )
    parts = configs.configs.reshape(npartitions, nper, -1)
    for i in range(1, npartitions):
        assert not np.allclose(parts[0], parts[i])


def test_correlated_sampling_replays_ecp(H2_ccecp_rhf):
    """Every parameter set in correlated sampling sees the same ECP rotations, so
    identical parameter sets give identical energies."""
    mol, mf = H2_ccecp_rhf
    wf, to_opt = pyq.generate_wf(mol, mf)
    acc = pyq.gradient_generator(mol, wf, to_opt)
    x = acc.transform.serialize_parameters(wf.parameters)
    configs = pyq.initial_guess(mol, 20, rng=0)
    data = correlated_compute_worker(wf, configs, [x, x, x], acc, [0, 1], rng=0)
    assert np.array_equal(data["total"][0], data["total"][1])
    assert np.array_equal(data["total"][0], data["total"][2])
