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
    generate_wf,
    gradient_generator,
    initial_guess,
    minsr_optimization,
)
from pyqmc.observables.accumulators import LinearTransform


@pytest.mark.slow
def test_minsr(H2_ccecp_uhf):
    """Optimize a Slater-Jastrow wave function with minSR and check that it's
    better than Hartree-Fock."""
    mol, mf = H2_ccecp_uhf
    mol.output, mol.stdout = None, None

    np.random.seed(0)
    wf, to_opt = generate_wf(mol, mf)
    nconf = 1000
    wf, df = minsr_optimization(
        wf,
        initial_guess(mol, nconf),
        LinearTransform(wf.parameters, to_opt),
        EnergyAccumulator(mol),
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

    np.random.seed(0)
    wf, to_opt = generate_wf(mol, mf)
    # one transform per Jastrow parameter group
    groups = ["wf2acoeff", "wf2bcoeff"]
    transforms = [
        LinearTransform(wf.parameters, {k: to_opt[k]}) for k in groups
    ]
    start = {k: np.array(v) for k, v in wf.parameters.items()}
    frozen = [k for k in wf.parameters if k not in groups]

    kws = dict(
        tstep=0.1,
        vmcoptions={"nsteps_per_block": 5},
        hdf_file=hdf_file,
    )
    wf, df = minsr_optimization(
        wf, initial_guess(mol, 400), transforms, EnergyAccumulator(mol),
        max_iterations=6, **kws
    )

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
    transforms2 = [LinearTransform(wf2.parameters, {k: to_opt[k]}) for k in groups]
    wf2, df2 = minsr_optimization(
        wf2, initial_guess(mol, 400), transforms2, EnergyAccumulator(mol),
        max_iterations=8, **kws
    )
    assert [d["iteration"] for d in df2] == [6, 6, 7, 7]
    assert [d["sub_iteration"] for d in df2] == [0, 1, 0, 1]
    with h5py.File(hdf_file, "r") as hdf:
        assert hdf["energy"].shape == (16,)


@pytest.mark.slow
def test_minsr_matches_sr_step(H2_ccecp_uhf):
    """One minSR step reproduces the stochastic reconfiguration step computed
    from the same samples, without ever forming the S matrix."""
    from pyqmc.method.minsr import minsr_update, sample_minsr_data

    mol, mf = H2_ccecp_uhf
    mol.output, mol.stdout = None, None
    wf, to_opt = generate_wf(mol, mf)
    pgrad = gradient_generator(mol, wf, to_opt)
    coords = initial_guess(mol, 200)

    data, coords = sample_minsr_data(
        wf, coords, pgrad.transform, pgrad.enacc, pgrad.nodal_cutoff
    )
    dppsi, eloc = data["dppsi"], data["total"]
    tstep = 0.1

    dp_minsr, _ = minsr_update(dppsi, eloc, tstep, eps=pgrad.eps)
    # pgrad is only built here to compare against; minsr itself never needs one
    sr_averages = {
        "total": np.mean(eloc),
        "dppsi": np.mean(dppsi, axis=0),
        "dpH": np.mean(eloc[:, np.newaxis] * dppsi, axis=0),
        "dpidpj": np.einsum("ij,ik->jk", dppsi, dppsi) / dppsi.shape[0],
    }
    dp_sr = pgrad.delta_p([tstep], sr_averages)[0][0]
    assert np.allclose(dp_sr, dp_minsr, atol=1e-8), np.abs(dp_sr - dp_minsr).max()


if __name__ == "__main__":
    test_minsr()
