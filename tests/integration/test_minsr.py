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
import pyqmc.rng
import pandas as pd
import pytest

from pyqmc.api import (
    EnergyAccumulator,
    generate_wf,
    gradient_generator,
    initial_guess,
    optimize,
    MinSRUpdate,
)
from pyqmc.observables.accumulators import LinearTransform


@pytest.mark.slow
def test_minsr(H2_ccecp_uhf):
    """Optimize a Slater-Jastrow wave function with minSR and check that it's
    better than Hartree-Fock."""
    mol, mf = H2_ccecp_uhf
    mol.output, mol.stdout = None, None

    pyqmc.rng.seed(0)
    wf, to_opt = generate_wf(mol, mf)
    nconf = 1000
    wf, df = optimize(
        wf,
        initial_guess(mol, nconf),
        MinSRUpdate(EnergyAccumulator(mol), LinearTransform(wf.parameters, to_opt)),
        tstep=0.1,
        max_iterations=20,
        verbose=True,
    )

    df = pd.DataFrame(df)
    mfen = mf.energy_tot()
    # the last iteration is one noisy estimate, so average the tail
    enfinal = df["energy"].values[-5:].mean()
    assert mfen > enfinal, (mfen, enfinal)


@pytest.mark.slow
def test_minsr_sub_iterations(H2_ccecp_uhf, tmp_path):
    """A list of transforms optimizes each parameter group in its own
    sub-iteration, and restarts pick up where the last sub-iteration left off."""
    import h5py

    mol, mf = H2_ccecp_uhf
    mol.output, mol.stdout = None, None
    hdf_file = str(tmp_path / "minsr_sub.hdf5")

    pyqmc.rng.seed(0)
    wf, to_opt = generate_wf(mol, mf)
    # one transform per Jastrow parameter group
    groups = ["wf2acoeff", "wf2bcoeff"]
    updaters = [
        MinSRUpdate(EnergyAccumulator(mol), LinearTransform(wf.parameters, {k: to_opt[k]}))
        for k in groups
    ]
    start = {k: np.array(v) for k, v in wf.parameters.items()}
    frozen = [k for k in wf.parameters if k not in groups]

    kws = dict(
        tstep=0.1,
        vmc_kwargs={"nsteps_per_block": 5},
        hdf_file=hdf_file,
    )
    wf, df = optimize(wf, initial_guess(mol, 400), updaters, max_iterations=6, **kws)

    df = pd.DataFrame(df)
    assert len(df) == 12
    assert list(df["sub_iteration"].values) == [0, 1] * 6
    assert list(df["iteration"].values) == [i for i in range(6) for _ in range(2)]
    assert mf.energy_tot() > df["energy"].values[-4:].mean()
    for k in groups:
        assert not np.allclose(start[k], np.array(wf.parameters[k])), k
    # parameters outside every transform are held fixed
    for k in frozen:
        assert np.allclose(start[k], np.array(wf.parameters[k])), k

    # restarting continues after the last recorded sub-iteration
    wf2, _ = generate_wf(mol, mf)
    updaters2 = [
        MinSRUpdate(EnergyAccumulator(mol), LinearTransform(wf2.parameters, {k: to_opt[k]}))
        for k in groups
    ]
    wf2, df2 = optimize(wf2, initial_guess(mol, 400), updaters2, max_iterations=8, **kws)
    assert [d["iteration"] for d in df2] == [6, 6, 7, 7]
    assert [d["sub_iteration"] for d in df2] == [0, 1, 0, 1]
    with h5py.File(hdf_file, "r") as hdf:
        assert hdf["energy"].shape == (16,)


@pytest.mark.slow
def test_minsr_matches_sr_step(H2_ccecp_uhf):
    """On the same sampled data, the minSR direction equals the explicit-S
    direction built from the averages the SR accumulator uses."""
    from pyqmc.observables.stochastic_reconfiguration import sr_averages, sr_direction

    mol, mf = H2_ccecp_uhf
    mol.output, mol.stdout = None, None
    wf, to_opt = generate_wf(mol, mf)
    pgrad = gradient_generator(mol, wf, to_opt)
    updater = MinSRUpdate(pgrad.enacc, pgrad.transform, eps=pgrad.eps)
    data, _ = updater.sample(wf, initial_guess(mol, 200))

    v_minsr, _ = updater.solve(data)
    v_sr, _ = sr_direction(sr_averages({"total": data["total"]}, data["dppsi"]), pgrad.eps)
    assert np.allclose(v_sr, v_minsr, atol=1e-8), np.abs(v_sr - v_minsr).max()


if __name__ == "__main__":
    test_minsr()
