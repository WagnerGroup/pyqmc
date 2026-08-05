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

from pyqmc.api import generate_wf, initial_guess, gradient_generator, minsr_optimization


@pytest.mark.slow
def test_minsr(H2_ccecp_uhf):
    """Optimize a Slater-Jastrow wave function with minSR and check that it's
    better than Hartree-Fock."""
    mol, mf = H2_ccecp_uhf
    mol.output, mol.stdout = None, None

    wf, to_opt = generate_wf(mol, mf)
    nconf = 1000
    wf, df = minsr_optimization(
        wf,
        initial_guess(mol, nconf),
        gradient_generator(mol, wf, to_opt),
        tstep=0.1,
        max_iterations=20,
        verbose=True,
    )

    df = pd.DataFrame(df)
    mfen = mf.energy_tot()
    enfinal = df["energy"].values[-1]
    assert mfen > enfinal


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
