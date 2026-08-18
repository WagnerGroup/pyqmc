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
import numpy as np
import pandas as pd
import pytest

from pyqmc.api import (
    EnergyAccumulator,
    cgsr_optimization,
    generate_wf,
    initial_guess,
    minsr_optimization,
)
from pyqmc.observables.accumulators import LinearTransform


@pytest.mark.slow
def test_cgsr(H2_ccecp_uhf):
    """Optimize a Slater-Jastrow wave function with CG-SR and check that it's
    better than Hartree-Fock, and that CG actually converged every step."""
    mol, mf = H2_ccecp_uhf
    mol.output, mol.stdout = None, None

    np.random.seed(0)
    wf, to_opt = generate_wf(mol, mf)
    wf, df = cgsr_optimization(
        wf,
        initial_guess(mol, 1000),
        LinearTransform(wf.parameters, to_opt),
        EnergyAccumulator(mol),
        tstep=0.1,
        max_iterations=20,
        verbose=False,
    )

    df = pd.DataFrame(df)
    # the last iteration is one noisy estimate, so average the tail
    assert mf.energy_tot() > df["energy"].values[-5:].mean()
    assert df["cg_converged"].all()
    # the whole point: the iteration count is set by conditioning, not nparams
    assert df["cg_iterations"].max() < LinearTransform(wf.parameters, to_opt).nparams * 3


@pytest.mark.slow
def test_cgsr_matches_minsr_trajectory(H2_ccecp_uhf):
    """CG and minSR solve the same equations, so from the same seed they take the
    same optimization path."""
    mol, mf = H2_ccecp_uhf
    mol.output, mol.stdout = None, None

    kws = dict(
        tstep=0.05,
        eps=1e-2,
        max_iterations=6,
        vmcoptions={"nsteps_per_block": 5},
        verbose=False,
    )
    energies = {}
    for name, optimize, extra in [
        ("minsr", minsr_optimization, {}),
        ("cgsr", cgsr_optimization, dict(rtol=1e-10)),
    ]:
        np.random.seed(0)
        wf, to_opt = generate_wf(mol, mf)
        _, df = optimize(
            wf,
            initial_guess(mol, 400),
            LinearTransform(wf.parameters, to_opt),
            EnergyAccumulator(mol),
            **kws,
            **extra,
        )
        energies[name] = np.array([d["energy"] for d in df])

    assert np.abs(energies["cgsr"] - energies["minsr"]).max() < 1e-8


@pytest.mark.slow
def test_cgsr_sub_iterations_and_restart(H2_ccecp_uhf, tmp_path):
    """A list of transforms optimizes each parameter group in its own
    sub-iteration, each with its own warm start, and restarts resume correctly."""
    import h5py

    mol, mf = H2_ccecp_uhf
    mol.output, mol.stdout = None, None
    hdf_file = str(tmp_path / "cgsr_sub.hdf5")

    np.random.seed(0)
    wf, to_opt = generate_wf(mol, mf)
    groups = ["wf2acoeff", "wf2bcoeff"]
    transforms = [LinearTransform(wf.parameters, {k: to_opt[k]}) for k in groups]
    frozen = [k for k in wf.parameters if k not in groups]
    start = {k: np.array(v) for k, v in wf.parameters.items()}

    kws = dict(tstep=0.1, vmcoptions={"nsteps_per_block": 5}, hdf_file=hdf_file)
    wf, df = cgsr_optimization(
        wf, initial_guess(mol, 400), transforms, EnergyAccumulator(mol),
        max_iterations=4, **kws
    )
    assert [d["sub_iteration"] for d in df] == [0, 1] * 4
    # parameters outside the transforms are untouched
    for k in frozen:
        assert np.array_equal(np.asarray(wf.parameters[k]), start[k])

    with h5py.File(hdf_file, "r") as hdf:
        assert list(hdf["sub_iteration"][()]) == [0, 1] * 4
        assert np.all(np.isfinite(hdf["energy"][()]))

    # restarting continues rather than starting over
    wf2, to_opt2 = generate_wf(mol, mf)
    transforms2 = [LinearTransform(wf2.parameters, {k: to_opt2[k]}) for k in groups]
    cgsr_optimization(
        wf2, initial_guess(mol, 400), transforms2, EnergyAccumulator(mol),
        max_iterations=6, **kws
    )
    with h5py.File(hdf_file, "r") as hdf:
        assert len(hdf["energy"][()]) > 8
