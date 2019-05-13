import numpy as np
import testwf
import pytest


def test_slater():
    from pyscf import lib, gto, scf
    from slater import PySCFSlaterRHF
    from jastrow import Jastrow2B
    mol = gto.M(atom='Li 0. 0. 0.; H 0. 0. 1.5', basis='cc-pvtz',unit='bohr')
    mf = scf.RHF(mol).run()
    wf=PySCFSlaterRHF(10,mol,mf)
    #wf=Jastrow2B(10,mol)
    epsilon=1e-5
    epos=np.random.randn(10,4,3)
    assert testwf.test_wf_gradient(wf, epos, delta=1e-5)[0] < epsilon 
    assert testwf.test_wf_laplacian(wf, epos, delta=1e-5)[0] < epsilon 
    assert testwf.test_wf_gradient(wf, epos, delta=1e-5)[0] < epsilon 
    assert testwf.test_wf_laplacian(wf, epos, delta=1e-5)[0] < epsilon 

