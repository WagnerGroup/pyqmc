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

import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import copy

import h5py
import numpy as np
import pytest

import pyqmc.api as pyq
from pyqmc.method.ensemble_minsr import MinSRWfbyWf, optimize_ensemble
from pyqmc.observables.accumulators import LinearTransform


def make_ensemble(mol, mf, mc, nstates):
    """A CASCI state for each root, with its determinant coefficients free."""
    mcs = [copy.copy(mc) for _ in range(nstates)]
    energy = pyq.EnergyAccumulator(mol)
    wfs = []
    updater = []
    for i in range(nstates):
        mcs[i].ci = mc.ci[i]
        wf, to_opt = pyq.generate_slater(
            mol, mf, mc=mcs[i], optimize_determinants=True, tol=1e-20
        )
        wfs.append(wf)
        updater.append([MinSRWfbyWf(energy, LinearTransform(wf.parameters, to_opt))])
    return wfs, updater


@pytest.mark.slow
def test_ensemble_minsr(H2_casci, tmp_path):
    """Two states optimized together: energies are recorded for both, the states
    stay distinct, and the hdf output has the same layout as the SR version."""
    mol, mf, mc = H2_casci
    hdf_file = str(tmp_path / "ensemble_minsr.hdf5")

    np.random.seed(0)
    wfs, updater = make_ensemble(mol, mf, mc, nstates=2)
    configs = pyq.initial_guess(mol, 200)

    wfs = optimize_ensemble(
        wfs,
        configs,
        updater,
        hdf_file=hdf_file,
        tau=0.1,
        max_iterations=3,
        npartitions=1,
        verbose=False,
        minsr_kwargs={"nblocks": 1, "nsteps_per_block": 3},
        overlap_kwargs={"nblocks": 2, "nsteps": 3},
    )

    with h5py.File(hdf_file, "r") as hdf:
        keys = set(hdf.keys())
        assert {"energy0", "energy1", "overlap0", "overlap1", "iteration"} <= keys
        assert list(hdf["wavefunction"][()]) == [0, 1] * 3
        assert list(hdf["iteration"][()]) == [0, 0, 1, 1, 2, 2]
        # energy{i} is only written on the rows belonging to state i, so it has
        # one entry per iteration rather than one per row
        e0 = hdf["energy0"][()]
        e1 = hdf["energy1"][()]
        err0 = hdf["energy_error0"][()]
        overlap1 = hdf["overlap1"][()]
        assert len(e0) == 3 and len(e1) == 3 and len(overlap1) == 3
        assert np.all(np.isfinite(e0)) and np.all(np.isfinite(e1))
        assert np.all(err0 > 0)
        # the excited state stays above the ground state
        assert e1[-1] > e0[-1]
        # normalized off-diagonal overlap stays below 1, i.e. the states are distinct
        norm = np.sqrt(np.abs(overlap1[-1][0, 0] * overlap1[-1][1, 1]))
        assert np.abs(overlap1[-1][1, 0]) / norm < 0.5

    # restarting continues from the recorded iteration
    wfs2, updater2 = make_ensemble(mol, mf, mc, nstates=2)
    optimize_ensemble(
        wfs2,
        pyq.initial_guess(mol, 200),
        updater2,
        hdf_file=hdf_file,
        tau=0.1,
        max_iterations=4,
        npartitions=1,
        verbose=False,
        minsr_kwargs={"nblocks": 1, "nsteps_per_block": 3},
        overlap_kwargs={"nblocks": 2, "nsteps": 3},
    )
    with h5py.File(hdf_file, "r") as hdf:
        assert list(hdf["iteration"][()]) == [0, 0, 1, 1, 2, 2, 3, 3]


@pytest.mark.slow
def test_ensemble_minsr_with_client(H2_casci, tmp_path):
    """With a client the sampling happens in worker processes, so the threads
    share the wave functions instead of copying them."""
    from concurrent.futures import ProcessPoolExecutor

    mol, mf, mc = H2_casci
    hdf_file = str(tmp_path / "ensemble_minsr_client.hdf5")

    np.random.seed(0)
    wfs, updater = make_ensemble(mol, mf, mc, nstates=2)
    ncore = 2
    with ProcessPoolExecutor(max_workers=ncore) as client:
        wfs = optimize_ensemble(
            wfs,
            pyq.initial_guess(mol, 200),
            updater,
            hdf_file=hdf_file,
            client=client,
            tau=0.1,
            max_iterations=2,
            npartitions=ncore,
            verbose=False,
            minsr_kwargs={"nblocks": 1, "nsteps_per_block": 3},
            overlap_kwargs={"nblocks": 2, "nsteps": 3},
        )
    with h5py.File(hdf_file, "r") as hdf:
        assert np.all(np.isfinite(hdf["energy0"][()]))
        assert np.all(np.isfinite(hdf["energy1"][()]))
        assert len(hdf["energy1"][()]) == 2


@pytest.mark.slow
def test_ensemble_minsr_matches_sr(H2_casci):
    """The ensemble minSR step reproduces the ensemble SR step on real sampled
    data, including the overlap penalty."""
    from pyqmc.method.ensemble_optimization_wfbywf import (
        StochasticReconfigurationWfbyWf,
    )
    from pyqmc.method.minsr import sample_minsr_data
    from pyqmc.method.sample_many import sample_overlap

    mol, mf, mc = H2_casci
    np.random.seed(0)
    wfs, updater = make_ensemble(mol, mf, mc, nstates=2)
    configs = pyq.initial_guess(mol, 300)
    wfi = 1  # the excited state, which carries the overlap penalty
    up = updater[wfi][0]
    eps, tau = 1e-2, 0.1
    penalty = np.ones((2, 2)) * 0.5

    # one energy sample and one overlap sample, shared by both updates
    sample1, configs = sample_minsr_data(
        wfs[wfi], configs, up.transform, up.enacc, up.nodal_cutoff, nsteps_per_block=3
    )
    weighted, unweighted, _ = sample_overlap(
        wfs, configs, up.allwfs(), nblocks=2, nsteps=3
    )

    up.eps = eps
    avg, _ = up.block_average(sample1, weighted, unweighted["overlap"])
    dp_minsr = up.delta_p([tau], avg, penalty)[0][0]

    # the same data, averaged the way the SR accumulator would have
    dppsi, eloc = sample1["dppsi"], sample1["total"]
    sr = StochasticReconfigurationWfbyWf(up.enacc, up.transform, eps=eps)
    sr_sample1 = {
        "total": np.array([np.mean(eloc)] * 2),
        "dppsi": np.array([np.mean(dppsi, axis=0)] * 2),
        "dpH": np.array([np.mean(eloc[:, np.newaxis] * dppsi, axis=0)] * 2),
        "dpidpj": np.array(
            [np.einsum("ij,ik->jk", dppsi, dppsi) / dppsi.shape[0]] * 2
        ),
    }
    sr_avg, _ = sr.block_average(sr_sample1, weighted, unweighted["overlap"])
    dp_sr = sr.delta_p([tau], sr_avg, penalty)[0][0]

    assert np.allclose(dp_sr, dp_minsr, atol=1e-8), np.abs(dp_sr - dp_minsr).max()
