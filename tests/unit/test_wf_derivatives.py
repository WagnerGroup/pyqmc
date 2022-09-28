import numpy as np
import pyqmc.testwf as testwf
from pyqmc.gpu import cp, asnumpy
from pyqmc.slater import Slater
from pyqmc.multiplywf import MultiplyWF
from pyqmc.addwf import AddWF
from pyqmc.j3 import J3
from pyqmc.wftools import generate_jastrow
import pyqmc.api as pyq


def run_tests(wf, epos, epsilon):

    _, epos = pyq.vmc(wf, epos, nblocks=1, nsteps=2, tstep=1)  # move off node

    for k, item in testwf.test_updateinternals(wf, epos).items():
        print(k, item)
        assert item < epsilon

    testwf.test_mask(wf, 0, epos)

    for fname, func in zip(
        ["gradient", "laplacian", "pgradient"],
        [testwf.test_wf_gradient, testwf.test_wf_laplacian, testwf.test_wf_pgradient,],
    ):
        err = [func(wf, epos, delta) for delta in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]]
        assert min(err) < epsilon, "epsilon {0}".format(epsilon)

    for fname, func in zip(
        ["gradient_value", "gradient_laplacian"],
        [testwf.test_wf_gradient_value, testwf.test_wf_gradient_laplacian,],
    ):
        d = func(wf, epos)
        for k, v in d.items():
            assert v < 1e-10, (k, v)


def test_obc_wfs(LiH_sto3g_rhf, epsilon=1e-5, nconf=10):
    """
    Ensure that the wave function objects are consistent in several situations.
    """

    mol, mf = LiH_sto3g_rhf
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

        epos = pyq.initial_guess(mol, nconf)
        run_tests(wf, epos, epsilon)


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
        run_tests(wf, epos, epsilon)


def test_pbc_wfs_triplet(h_noncubic_sto3g_triplet, epsilon=1e-5, nconf=10):
    """
    Ensure that the wave function objects are consistent in several situations.
    """
    mol, mf = h_noncubic_sto3g_triplet

    # supercell = pyq.get_supercell(mol, S=(np.ones((3, 3)) - 2 * np.eye(3)))
    supercell = pyq.get_supercell(mol, S=np.identity(3, dtype=int))
    epos = pyq.initial_guess(supercell, nconf)
    for wf in [
        MultiplyWF(Slater(supercell, mf), generate_jastrow(supercell)[0]),
        Slater(supercell, mf),
    ]:
        for k in wf.parameters:
            if "mo_coeff" not in k and k != "det_coeff":
                wf.parameters[k] = cp.asarray(np.random.rand(*wf.parameters[k].shape))

        _, epos = pyq.vmc(wf, epos, nblocks=1, nsteps=2, tstep=1)  # move off node
        run_tests(wf, epos, epsilon)


def test_hci_wf(H2_ccecp_hci, epsilon=1e-5):
    mol, mf, cisolver = H2_ccecp_hci
    configs = pyq.initial_guess(mol, 10)
    wf = Slater(mol, mf, cisolver, tol=0.0)
    run_tests(wf, configs, epsilon)


def test_rohf(C_ccecp_rohf, epsilon=1e-5):
    mol, mf = C_ccecp_rohf
    configs = pyq.initial_guess(mol, 10)
    wf = Slater(mol, mf)
    run_tests(wf, configs, epsilon)


def test_casci_s0(H2_ccecp_casci_s0, epsilon=1e-5):
    mol, mf, cisolver = H2_ccecp_casci_s0
    configs = pyq.initial_guess(mol, 10)
    wf = Slater(mol, mf, cisolver, tol=0.0)
    run_tests(wf, configs, epsilon)


def test_casci_s2(H2_ccecp_casci_s2, epsilon=1e-5):
    mol, mf, cisolver = H2_ccecp_casci_s2
    configs = pyq.initial_guess(mol, 10)
    wf = Slater(mol, mf, cisolver, tol=0.0)
    run_tests(wf, configs, epsilon)


def test_manual_slater(H2_ccecp_rhf, epsilon=1e-5):
    mol, mf = H2_ccecp_rhf

    determinants = [(1.0, [[0], [0]]), (-0.2, [[1], [1]])]
    wf = Slater(mol, mf, determinants=determinants)
    configs = pyq.initial_guess(mol, 10)
    run_tests(wf, configs, epsilon)


def test_manual_pbcs_fail(H_pbc_sto3g_krks, epsilon=1e-5, nconf=10):
    """
    This test makes sure that the number of k-points must match the number of k-points
    in the mf object.
    """
    mol, mf = H_pbc_sto3g_krks
    supercell = np.identity(3, dtype=int)
    supercell[0, 0] = 2
    mol = pyq.get_supercell(mol, supercell)
    try:
        determinants = [
            (1.0, [[0, 1], [0, 1]], [[0, 1], [0, 1]]),  # first determinant
            (-0.2, [[0, 2], [0, 1]], [[0, 2], [0, 1]]),  # second determinant
        ]
        wf = Slater(mol, mf, determinants=determinants)
        raise Exception("Should have failed here")
    except:
        pass


def test_manual_pbcs_correct(H_pbc_sto3g_kuks, epsilon=1e-5, nconf=10):
    """
    This test makes sure that the number of k-points must match the number of k-points
    in the mf object.
    """
    from pyqmc.determinant_tools import create_pbc_determinant

    mol, mf = H_pbc_sto3g_kuks
    supercell = np.identity(3, dtype=int)
    supercell[0, 0] = 2
    mol = pyq.get_supercell(mol, supercell)

    determinants = [
        (1.0, create_pbc_determinant(mol, mf, [])),
        (-0.2, create_pbc_determinant(mol, mf, [(0, 0, 0, 0, 1)])),
    ]
    wf = Slater(mol, mf, determinants=determinants)
    configs = pyq.initial_guess(mol, 10)
    run_tests(wf, configs, epsilon)


def test_superpose_wf(
    H2_casci, coeffs=[1 / np.sqrt(2), 1 / np.sqrt(2)], epsilon=1e-5, nconf=10
):
    """
    This test makes sure that the superposewf passes all the wftests, when adding two casci wave functions.
    """

    mol, mf, mc = H2_casci
    ci0, ci1 = mc.ci[0], mc.ci[1]

    mc.ci = ci0
    wf0 = Slater(mol, mf, mc, tol=0.0)

    mc.ci = ci1
    wf1 = Slater(mol, mf, mc, tol=0.0)

    wfs = [wf0, wf1]
    wf = AddWF(coeffs, wfs)
    configs = pyq.initial_guess(mol, nconf)
    run_tests(wf, configs, epsilon)
