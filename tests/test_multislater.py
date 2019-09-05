import os

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np
import pandas as pd
from pyscf import lib, gto, scf, mcscf
import pyqmc.testwf as testwf
from pyqmc.mc import vmc, initial_guess
from pyqmc.accumulators import EnergyAccumulator
from pyqmc.multislater import MultiSlater
from pyqmc.coord import OpenConfigs
import pyqmc

def test():
    """ 
    Tests that the multi-slater wave function value, gradient and 
    parameter gradient evaluations are working correctly. Also 
    checks that VMC energy matches energy calculated in PySCF
    """
    mol = gto.M(atom="Li 0. 0. 0.; H 0. 0. 1.5", basis="cc-pvtz", unit="bohr", spin=0)
    for mf in [scf.RHF(mol).run(), scf.ROHF(mol).run(), scf.UHF(mol).run()]:
        #Test same number of elecs
        mc = mcscf.CASCI(mf,ncas=4,nelecas=(1,1))
        mc.kernel()
        wf = MultiSlater(mol, mf, mc)

        epsilon = 1e-5
        nconf = 10
        nelec = np.sum(mol.nelec)
        epos = OpenConfigs(np.random.randn(nconf, nelec, 3))
      
        for k, item in testwf.test_updateinternals(wf, epos).items():
            assert item < epsilon
        assert testwf.test_wf_gradient(wf, epos, delta=1e-5)[0] < epsilon
        assert testwf.test_wf_laplacian(wf, epos, delta=1e-5)[0] < epsilon
        assert testwf.test_wf_pgradient(wf, epos, delta=1e-5)[0] < epsilon

        #Test same number of elecs
        mc = mcscf.CASCI(mf,ncas=4,nelecas=(1,1))
        mc.kernel()
        wf = pyqmc.default_msj(mol, mf, mc)[0]

        epsilon = 1e-5
        nconf = 10
        nelec = np.sum(mol.nelec)
        epos = OpenConfigs(np.random.randn(nconf, nelec, 3))
      
        for k, item in testwf.test_updateinternals(wf, epos).items():
            assert item < epsilon
        assert testwf.test_wf_gradient(wf, epos, delta=1e-5)[0] < epsilon
        assert testwf.test_wf_laplacian(wf, epos, delta=1e-5)[0] < epsilon
        assert testwf.test_wf_pgradient(wf, epos, delta=1e-5)[0] < epsilon

        #Test different number of elecs
        mc = mcscf.CASCI(mf,ncas=4,nelecas=(2,0))
        mc.kernel()
        wf = MultiSlater(mol, mf, mc)

        epsilon = 1e-5
        nconf = 10
        nelec = np.sum(mol.nelec)
        epos = OpenConfigs(np.random.randn(nconf, nelec, 3))
      
        for k, item in testwf.test_updateinternals(wf, epos).items():
            assert item < epsilon
        assert testwf.test_wf_gradient(wf, epos, delta=1e-5)[0] < epsilon
        assert testwf.test_wf_laplacian(wf, epos, delta=1e-5)[0] < epsilon
        assert testwf.test_wf_pgradient(wf, epos, delta=1e-5)[0] < epsilon
  
if __name__ == "__main__":
    test()
