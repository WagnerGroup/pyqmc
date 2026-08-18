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
from pyqmc.method.ensemble_cgsr import CGSRWfbyWf
from pyqmc.method.ensemble_optimization import optimize_ensemble
from pyqmc.observables.accumulators import LinearTransform

# small samplings so the test is quick
KWS = dict(
    vmc_kwargs={"nblocks": 1, "nsteps_per_block": 3},
    overlap_kwargs={"nblocks": 2, "nsteps_per_block": 3},
    initial_vmc_warmup_kwargs={"nblocks": 1, "nsteps_per_block": 3},
    initial_overlap_warmup_kwargs={"nblocks": 1, "nsteps_per_block": 3},
    refresh_vmc_warmup_kwargs={"nblocks": 1, "nsteps_per_block": 2},
    refresh_overlap_warmup_kwargs={"nblocks": 1, "nsteps_per_block": 2},
)


def make_ensemble(mol, mf, mc, nstates):
    """A CASCI state for each root, with its determinant coefficients free."""
    mcs = [copy.copy(mc) for _ in range(nstates)]
    wfs, transforms = [], []
    for i in range(nstates):
        mcs[i].ci = mc.ci[i]
        wf, to_opt = pyq.generate_slater(
            mol, mf, mc=mcs[i], optimize_determinants=True, tol=1e-20
        )
        wfs.append(wf)
        transforms.append(LinearTransform(wf.parameters, to_opt))
    return wfs, transforms


@pytest.mark.slow
def test_ensemble_cgsr(H2_casci, tmp_path):
    """Two states optimized together through the shared driver with method='cgsr'."""
    mol, mf, mc = H2_casci
    hdf_file = str(tmp_path / "ensemble_cgsr.hdf5")

    np.random.seed(0)
    wfs, transforms = make_ensemble(mol, mf, mc, nstates=2)
    optimize_ensemble(
        wfs, pyq.initial_guess(mol, 200), transforms, hdf_file,
        enacc=pyq.EnergyAccumulator(mol), method="cgsr", tau=0.1,
        max_iterations=3, npartitions=1, verbose=False, **KWS
    )

    with h5py.File(hdf_file, "r") as hdf:
        e0, e1 = hdf["energy0"][()], hdf["energy1"][()]
        overlap1 = hdf["overlap1"][()]
        assert len(e0) == 3 and len(e1) == 3
        assert np.all(np.isfinite(e0)) and np.all(np.isfinite(e1))
        assert e1[-1] > e0[-1]  # the excited state stays above the ground state
        norm = np.sqrt(np.abs(overlap1[-1][0, 0] * overlap1[-1][1, 1]))
        assert np.abs(overlap1[-1][1, 0]) / norm < 0.5  # states stay distinct


@pytest.mark.slow
def test_ensemble_cgsr_matches_minsr(H2_casci, tmp_path):
    """CG and minSR solve the same ensemble equations, so from the same seed they
    take the same path, overlap penalty included."""
    mol, mf, mc = H2_casci

    energies = {}
    for method in ["minsr", "cgsr"]:
        np.random.seed(0)
        wfs, transforms = make_ensemble(mol, mf, mc, nstates=2)
        updaters = transforms
        if method == "cgsr":  # tighten CG so the comparison is not solver-limited
            updaters = [
                [CGSRWfbyWf(pyq.EnergyAccumulator(mol), t, eps=1e-2, rtol=1e-11)]
                for t in transforms
            ]
        hdf_file = str(tmp_path / f"ensemble_{method}.hdf5")
        optimize_ensemble(
            wfs, pyq.initial_guess(mol, 200), updaters, hdf_file,
            enacc=pyq.EnergyAccumulator(mol), method=method, tau=0.1,
            max_iterations=3, npartitions=1, verbose=False, **KWS
        )
        with h5py.File(hdf_file, "r") as hdf:
            energies[method] = np.array([hdf["energy0"][()], hdf["energy1"][()]])

    assert np.abs(energies["cgsr"] - energies["minsr"]).max() < 1e-8


@pytest.mark.slow
def test_ensemble_cgsr_with_client(H2_casci, tmp_path):
    """The solve is serial and local, so a client changes nothing about it, but
    the sampling still has to work through worker processes."""
    from concurrent.futures import ProcessPoolExecutor

    mol, mf, mc = H2_casci
    hdf_file = str(tmp_path / "ensemble_cgsr_client.hdf5")

    np.random.seed(0)
    wfs, transforms = make_ensemble(mol, mf, mc, nstates=2)
    with ProcessPoolExecutor(max_workers=2) as client:
        optimize_ensemble(
            wfs, pyq.initial_guess(mol, 200), transforms, hdf_file,
            enacc=pyq.EnergyAccumulator(mol), method="cgsr", tau=0.1,
            max_iterations=2, client=client, npartitions=2, verbose=False, **KWS
        )
    with h5py.File(hdf_file, "r") as hdf:
        assert np.all(np.isfinite(hdf["energy0"][()]))
        assert len(hdf["energy1"][()]) == 2
