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
import pyqmc.rng

import pyqmc.api as pyq
from pyqmc.method.ensemble_optimization import optimize_ensemble
from pyqmc.observables.accumulators import LinearTransform

KWS = dict(
    vmc_kwargs={"nblocks": 1, "nsteps_per_block": 3},
    overlap_kwargs={"nblocks": 2, "nsteps_per_block": 3},
    initial_vmc_warmup_kwargs={"nblocks": 1, "nsteps_per_block": 3},
    initial_overlap_warmup_kwargs={"nblocks": 1, "nsteps_per_block": 3},
    refresh_vmc_warmup_kwargs={"nblocks": 1, "nsteps_per_block": 2},
    refresh_overlap_warmup_kwargs={"nblocks": 1, "nsteps_per_block": 2},
)


def test_ensemble_complex_orbitals(H2_casci, tmp_path):
    """Ensemble optimization with complex orbitals and real determinant coefficients.

    The sampled overlap diagonal then has complex dtype; renormalization must keep
    det_coeff real so that the checkpoint can be written on every iteration.
    """
    mol, mf, mc = H2_casci
    mf = copy.copy(mf)
    phases = np.exp(1j * np.linspace(0.3, 1.1, mf.mo_coeff.shape[1]))
    mf.mo_coeff = mf.mo_coeff * phases[np.newaxis, :]

    wfs, transforms = [], []
    for i in range(2):
        mci = copy.copy(mc)
        mci.ci = mc.ci[i]
        mci.mo_coeff = mf.mo_coeff
        wf, to_opt = pyq.generate_slater(mol, mf, mc=mci, optimize_determinants=True, tol=1e-20)
        assert wf.dtype == complex
        wfs.append(wf)
        transforms.append(LinearTransform(wf.parameters, to_opt))

    hdf_file = str(tmp_path / "ensemble_complex.hdf5")
    pyqmc.rng.seed(0)
    optimize_ensemble(
        wfs, pyq.initial_guess(mol, 100), transforms, hdf_file,
        enacc=pyq.EnergyAccumulator(mol), method="sr", tau=0.05,
        max_iterations=2, npartitions=1, verbose=False, **KWS
    )
    for wf in wfs:
        assert not np.iscomplexobj(wf.parameters["det_coeff"])
    with h5py.File(hdf_file, "r") as hdf:
        assert len(hdf["energy0"][()]) == 2
