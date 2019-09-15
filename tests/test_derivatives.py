# This must be done BEFORE importing numpy or anything else.
# Therefore it must be in your main script.
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import numpy as np
import pyqmc.testwf as testwf


def test_wfs():
    """
    Ensure that the wave function objects are consistent in several situations.
    """

    from pyscf import lib, gto, scf
    from pyqmc.slateruhf import PySCFSlaterUHF
    from pyqmc.jastrowspin import JastrowSpin
    from pyqmc.multiplywf import MultiplyWF
    from pyqmc.coord import OpenConfigs
    import pyqmc

    mol = gto.M(atom="Li 0. 0. 0.; H 0. 0. 1.5", basis="cc-pvtz", unit="bohr")
    mf = scf.RHF(mol).run()
    mf_rohf = scf.ROHF(mol).run()
    mf_uhf = scf.UHF(mol).run()
    epsilon = 1e-5
    nconf = 10
    epos = pyqmc.initial_guess(mol, nconf) 
    for wf in [
        JastrowSpin(mol),
        MultiplyWF(PySCFSlaterUHF(mol, mf), JastrowSpin(mol)),
        PySCFSlaterUHF(mol, mf_uhf),
        PySCFSlaterUHF(mol, mf),
        PySCFSlaterUHF(mol, mf_rohf),
    ]:
        for k in wf.parameters:
            if k != 'mo_coeff':
                wf.parameters[k] = np.random.rand(*wf.parameters[k].shape)
        for fname, func in zip(['gradient', 'laplacian', 'pgradient'],
                         [testwf.test_wf_gradient, testwf.test_wf_laplacian, testwf.test_wf_pgradient]):
            err = []
            for delta in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
                err.append(func(wf, epos, delta)[0])
            print(fname, min(err))
            assert(min(err) < epsilon)
                
        
        for k, item in testwf.test_updateinternals(wf, epos).items():
            print(k,item)
            assert item < epsilon


def test_func3d():
    """
    Ensure that the 3-dimensional functions correctly compute their gradient and laplacian
    """
    from pyqmc.func3d import (
        PadeFunction,
        PolyPadeFunction,
        GaussianFunction,
        CutoffCuspFunction,
        test_func3d_gradient,
        test_func3d_laplacian,
    )

    test_functions = {
        "Pade": PadeFunction(0.2),
        "PolyPade": PolyPadeFunction(2.0, 1.5),
        "CutoffCusp": CutoffCuspFunction(2.0, 1.5),
        "Gaussian": GaussianFunction(0.4),
    }
    delta = 1e-6
    epsilon = 1e-5

    for name, func in test_functions.items():
        grad = test_func3d_gradient(func, delta=delta)[0]
        lap =  test_func3d_laplacian(func, delta=delta)[0]
        print(name, grad, lap)
        assert  grad < epsilon
        assert lap < epsilon


if __name__ == "__main__":
    test_wfs()
    test_func3d()
