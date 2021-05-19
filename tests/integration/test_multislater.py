import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from pyscf import lib, gto, scf, mcscf
import pyqmc.testwf as testwf
import pyqmc.api as pyq
from pyqmc.accumulators import EnergyAccumulator
from pyqmc.slater import Slater


def test():
    """
    Tests that the multi-slater wave function value, gradient and
    parameter gradient evaluations are working correctly. Also
    checks that VMC energy matches energy calculated in PySCF
    """
    mol = gto.M(atom="Li 0. 0. 0.; H 0. 0. 1.5", basis="cc-pvtz", unit="bohr", spin=0)
    epsilon = 1e-4
    delta = 1e-5
    nsteps = 200
    warmup = 10
    for mf in [
        scf.UHF(mol).run()
    ]:  # [scf.RHF(mol).run(), scf.ROHF(mol).run(), scf.UHF(mol).run()]:
        # Test same number of elecs
        mc = mcscf.CASCI(mf, ncas=4, nelecas=(1, 1))
        mc.kernel()
        wf = Slater(mol, mf, mc)

        nconf = 10

        nelec = np.sum(mol.nelec)
        epos = pyq.initial_guess(mol, nconf)

        for k, item in testwf.test_updateinternals(wf, epos).items():
            assert item < epsilon
        assert testwf.test_wf_gradient(wf, epos, delta=delta) < epsilon
        assert testwf.test_wf_laplacian(wf, epos, delta=delta) < epsilon
        assert testwf.test_wf_pgradient(wf, epos, delta=delta) < epsilon

        # Test same number of elecs
        mc = mcscf.CASCI(mf, ncas=4, nelecas=(1, 1))
        mc.kernel()
        wf = pyq.generate_wf(mol, mf, mc=mc)[0]

        nelec = np.sum(mol.nelec)
        epos = pyq.initial_guess(mol, nconf)

        for k, item in testwf.test_updateinternals(wf, epos).items():
            assert item < epsilon
        assert testwf.test_wf_gradient(wf, epos, delta=delta) < epsilon
        assert testwf.test_wf_laplacian(wf, epos, delta=delta) < epsilon
        assert testwf.test_wf_pgradient(wf, epos, delta=delta) < epsilon

        # Test different number of elecs
        mc = mcscf.CASCI(mf, ncas=4, nelecas=(2, 0))
        mc.kernel()
        wf = Slater(mol, mf, mc=mc)

        nelec = np.sum(mol.nelec)
        epos = pyq.initial_guess(mol, nconf)

        for k, item in testwf.test_updateinternals(wf, epos).items():
            assert item < epsilon
        assert testwf.test_wf_gradient(wf, epos, delta=delta) < epsilon
        assert testwf.test_wf_laplacian(wf, epos, delta=delta) < epsilon
        assert testwf.test_wf_pgradient(wf, epos, delta=delta) < epsilon

        # Quick VMC test
        nconf = 1000
        coords = pyq.initial_guess(mol, nconf)
        df, coords = pyq.vmc(
            wf, coords, nsteps=nsteps, accumulators={"energy": EnergyAccumulator(mol)}
        )

        df = pd.DataFrame(df)
        df = pyq.avg_reblock(df["energytotal"][warmup:], 20)
        en = df.mean()
        err = df.sem()
        assert en - mc.e_tot < 5 * err


if __name__ == "__main__":
    test()
