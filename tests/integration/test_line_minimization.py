import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
from pyqmc.api import generate_wf, line_minimization, initial_guess, gradient_generator
import pytest


@pytest.mark.slow
def test_linemin(H2_ccecp_uhf):
    """Optimize a Slater-Jastrow wave function and check that it's better than Hartree-Fock"""
    mol, mf = H2_ccecp_uhf
    mol.output, mol.stdout = None, None

    wf, to_opt = generate_wf(mol, mf)
    nconf = 100
    wf, dfgrad = line_minimization(
        wf, initial_guess(mol, nconf), gradient_generator(mol, wf, to_opt)
    )

    dfgrad = pd.DataFrame(dfgrad)
    mfen = mf.energy_tot()
    enfinal = dfgrad["energy"].values[-1]
    enfinal_err = dfgrad["energy_error"].values[-1]
    assert mfen > enfinal


if __name__ == "__main__":
    test_linemin()
