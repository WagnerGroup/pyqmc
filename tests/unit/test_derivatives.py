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
    from pyqmc.multiplynwf import MultiplyNWF
    from pyqmc.coord import OpenConfigs
    from pyqmc.manybody_jastrow import J3
    import pyqmc

    mol = gto.M(atom="Li 0. 0. 0.; H 0. 0. 1.5", basis="sto-3g", unit="bohr")
    mf = scf.RHF(mol).run()
    mf_rohf = scf.ROHF(mol).run()
    mf_uhf = scf.UHF(mol).run()
    epsilon = 1e-4
    nconf = 10
    epos = pyqmc.initial_guess(mol, nconf)
    for wf in [
        JastrowSpin(mol),
        J3(mol),
        MultiplyWF(PySCFSlaterUHF(mol, mf), JastrowSpin(mol)),
        MultiplyNWF([PySCFSlaterUHF(mol, mf), JastrowSpin(mol), J3(mol)]),
        PySCFSlaterUHF(mol, mf_uhf),
        PySCFSlaterUHF(mol, mf),
        PySCFSlaterUHF(mol, mf_rohf),
    ]:
        for k in wf.parameters:
            if k != "mo_coeff":
                wf.parameters[k] = np.random.rand(*wf.parameters[k].shape)
        for k, item in testwf.test_updateinternals(wf, epos).items():
            print(k, item)
            assert item < epsilon

        testwf.test_mask(wf, 0, epos)

        for fname, func in zip(
            ["gradient", "laplacian", "pgradient"],
            [
                testwf.test_wf_gradient,
                testwf.test_wf_laplacian,
                testwf.test_wf_pgradient,
            ],
        ):
            err = []
            for delta in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
                err.append(func(wf, epos, delta)[0])
            print(fname, min(err))
            assert min(err) < epsilon, "epsilon {0}".format(epsilon)


def test_pbc_wfs():
    """
    Ensure that the wave function objects are consistent in several situations.
    """

    from pyscf.pbc import lib, gto, scf
    from pyqmc.slaterpbc import PySCFSlaterPBC, get_supercell
    from pyqmc.jastrowspin import JastrowSpin
    from pyqmc.multiplywf import MultiplyWF
    from pyqmc.coord import OpenConfigs
    import pyqmc

    mol = gto.M(
        atom="H 0. 0. 0.; H 1. 1. 1.", basis="sto-3g", unit="bohr", a=np.eye(3) * 4
    )
    mf = scf.KRKS(mol).run()
    # mf_rohf = scf.KROKS(mol).run()
    # mf_uhf = scf.KUKS(mol).run()
    epsilon = 1e-5
    nconf = 10
    supercell = get_supercell(mol, S=np.eye(3))
    epos = pyqmc.initial_guess(supercell, nconf)
    for wf in [
        MultiplyWF(PySCFSlaterPBC(supercell, mf), JastrowSpin(mol)),
        PySCFSlaterPBC(supercell, mf),
        # PySCFSlaterPBC(supercell, mf_uhf),
        # PySCFSlaterPBC(supercell, mf_rohf),
    ]:
        for k in wf.parameters:
            if "mo_coeff" not in k:
                wf.parameters[k] = np.random.rand(*wf.parameters[k].shape)
        for fname, func in zip(
            ["gradient", "laplacian", "pgradient"],
            [
                testwf.test_wf_gradient,
                testwf.test_wf_laplacian,
                testwf.test_wf_pgradient,
            ],
        ):
            err = []
            for delta in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
                err.append(func(wf, epos, delta)[0])
            print(fname, min(err))
            assert min(err) < epsilon

        for k, item in testwf.test_updateinternals(wf, epos).items():
            print(k, item)
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
        test_func3d_gradient_laplacian,
        test_func3d_pgradient,
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
        lap = test_func3d_laplacian(func, delta=delta)[0]
        andg, andl = test_func3d_gradient_laplacian(func)
        pgrad = test_func3d_pgradient(func, delta=1e-9)[0]
        print(name, grad, lap, "both:", andg, andl)
        print(name, pgrad)
        assert grad < epsilon
        assert lap < epsilon
        assert andg < epsilon
        assert andl < epsilon, andl
        for k, v in pgrad.items():
            assert v < epsilon, (name, k, v)

    # Check CutoffCusp does not diverge at r/rcut = 1
    rcut = 1.5
    f = CutoffCuspFunction(2.0, rcut)
    gamma = 2.0
    rc = 1.5
    basis = CutoffCuspFunction(gamma, rc)
    rvec = np.array([0, 0, rc])[np.newaxis, :]
    r = np.linalg.norm(rvec)[np.newaxis]

    v = basis.value(rvec, r)
    g = basis.gradient(rvec, r)
    l = basis.laplacian(rvec, r)
    g_both, l_both = basis.gradient_laplacian(rvec, r)

    assert abs(v).sum() == 0
    assert abs(g).sum() == 0
    assert abs(l).sum() == 0
    assert abs(g_both).sum() == 0
    assert abs(l_both).sum() == 0


if __name__ == "__main__":
    test_wfs()
    test_pbc_wfs()
    test_func3d()
