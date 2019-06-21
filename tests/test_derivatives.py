# This must be done BEFORE importing numpy or anything else. 
# Therefore it must be in your main script.
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import sys
import numpy as np
import pyqmc.testwf as testwf
import pytest


def test_wfs():
    """
    Ensure that the wave function objects are consistent in several situations.
    """
    
    from pyscf import lib, gto, scf
    from pyqmc.slateruhf import PySCFSlaterUHF
    from pyqmc.jastrowspin import JastrowSpin
    from pyqmc.multiplywf import MultiplyWF
    mol = gto.M(atom='Li 0. 0. 0.; H 0. 0. 1.5', basis='cc-pvtz',unit='bohr')
    mf = scf.RHF(mol).run()
    mf_rohf = scf.ROHF(mol).run()
    mf_uhf = scf.UHF(mol).run()
    epsilon=1e-5
    nconf=10
    epos=np.random.randn(nconf,4,3)
    for wf in [JastrowSpin(mol),
               MultiplyWF(PySCFSlaterUHF(mol,mf),JastrowSpin(mol)), 
               PySCFSlaterUHF(mol,mf_uhf),PySCFSlaterUHF(mol,mf), 
               PySCFSlaterUHF(mol,mf_rohf)]:
        for k in wf.parameters:
            wf.parameters[k]=np.random.rand(*wf.parameters[k].shape)
        assert testwf.test_wf_gradient(wf, epos, delta=1e-5)[0] < epsilon 
        assert testwf.test_wf_laplacian(wf, epos, delta=1e-5)[0] < epsilon 
        assert testwf.test_wf_pgradient(wf, epos, delta=1e-5)[0] < epsilon
        
        for k,item in testwf.test_updateinternals(wf,epos).items():
            assert item < epsilon


def test_func3d():
    """
    Ensure that the 3-dimensional functions correctly compute their gradient and laplacian
    """
    from pyqmc.func3d import PadeFunction,GaussianFunction,test_func3d_gradient,test_func3d_laplacian
    test_functions = {'Pade':PadeFunction(0.2), 'Gaussian':GaussianFunction(0.4)}
    delta=1e-6
    epsilon=1e-5
    
    for name, func in test_functions.items():
        assert test_func3d_gradient(func,delta=delta)[0] < epsilon
        assert test_func3d_laplacian(func,delta=delta)[0] < epsilon
        
    




if __name__=="__main__":
    test_wfs()
    test_func3d()

