import numpy as np
import pyqmc.testwf as testwf
from pyqmc.gpu import cp, asnumpy
from pyqmc.slater import Slater
from pyqmc.multiplywf import MultiplyWF
from pyqmc.manybody_jastrow import J3
from pyqmc.wftools import generate_jastrow
import pyqmc.api as pyq


def test_obc_wfs(LiH_sto3g_rhf, epsilon=1e-5, nconf=10):
    """
    Ensure that the wave function objects are consistent in several situations.
    """

    mol, mf = LiH_sto3g_rhf
    epos = pyq.initial_guess(mol, nconf)
    for wf in [
        generate_jastrow(mol)[0],
        J3(mol),
        MultiplyWF(Slater(mol, mf), generate_jastrow(mol)[0]),
        MultiplyWF(Slater(mol, mf), generate_jastrow(mol)[0], J3(mol)),
        Slater(mol, mf),
    ]:
        for k in wf.parameters:
            if k != "mo_coeff":
                wf.parameters[k] = cp.asarray(np.random.rand(*wf.parameters[k].shape))
        for k, item in testwf.test_updateinternals(wf, epos).items():
            print(k, item)
            assert item < epsilon

        testwf.test_mask(wf, 0, epos)

        _, epos = pyq.vmc(wf, epos, nblocks=1, nsteps=2, tstep=1)  # move off node

        for fname, func in zip(
            ["gradient", "laplacian", "pgradient"],
            [
                testwf.test_wf_gradient,
                testwf.test_wf_laplacian,
                testwf.test_wf_pgradient,
            ],
        ):
            err = [func(wf, epos, delta) for delta in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]]
            print(type(wf), fname, min(err))
            assert min(err) < epsilon, "epsilon {0}".format(epsilon)

        for fname, func in zip(
            ["gradient_value", "gradient_laplacian"],
            [
                testwf.test_wf_gradient_value,
                testwf.test_wf_gradient_laplacian,
            ],
        ):
            d = func(wf, epos)
            print(type(wf), fname, min(err))
            for k, v in d.items():
                assert v < 1e-10, (k, v)



def test_pbc_wfs(H_pbc_sto3g_krks, epsilon=1e-5, nconf=10):
    """
    Ensure that the wave function objects are consistent in several situations.
    """
    mol, mf = H_pbc_sto3g_krks

    supercell = pyq.get_supercell(mol, S=(np.ones((3, 3)) - 2 * np.eye(3)))
    epos = pyq.initial_guess(supercell, nconf)
    for wf in [
        MultiplyWF(Slater(supercell, mf), generate_jastrow(supercell)[0]),
        Slater(supercell, mf),
    ]:
        for k in wf.parameters:
            if "mo_coeff" not in k and k != "det_coeff":
                wf.parameters[k] = cp.asarray(np.random.rand(*wf.parameters[k].shape))

        _, epos = pyq.vmc(wf, epos, nblocks=1, nsteps=2, tstep=1)  # move off node

        for fname, func in zip(
            ["gradient", "laplacian", "pgradient"],
            [
                testwf.test_wf_gradient,
                testwf.test_wf_laplacian,
                testwf.test_wf_pgradient,
            ],
        ):
            err = [func(wf, epos, delta) for delta in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]]
            print(type(wf), fname, min(err))
            assert min(err) < epsilon

        for k, item in testwf.test_updateinternals(wf, epos).items():
            print(k, item)
            assert item < epsilon

        for fname, func in zip(
            ["gradient_value", "gradient_laplacian"],
            [
                testwf.test_wf_gradient_value,
                testwf.test_wf_gradient_laplacian,
            ],
        ):
            d = func(wf, epos)
            print(type(wf), fname, min(err))
            for k, v in d.items():
                assert v < 1e-10, (k, v)

