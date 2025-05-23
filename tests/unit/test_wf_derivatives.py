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

import numpy as np
import pyqmc.wf.testwf as testwf
from pyqmc.gpu import cp, asnumpy
from pyqmc.wf.slater import Slater
from pyqmc.wf.multiplywf import MultiplyWF
from pyqmc.wf.addwf import AddWF
from pyqmc.wf.geminaljastrow import GeminalJastrow
from pyqmc.wf.three_body_jastrow import ThreeBodyJastrow
from pyqmc.wftools import default_jastrow_basis
from pyqmc.wftools import generate_jastrow
from pyqmc.wftools import generate_gps_jastrow
from pyqmc.wf.gps2 import GPSJastrow
import pyqmc.api as pyq

import jax
from pyqmc.wf.jax.slater import JAXSlater
from pyqmc.wf.jax.jastrowspin import JAXJastrowSpin
cpu=True
if cpu:
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update("jax_enable_x64", True)
else:
    pass 


def run_tests(wf, epos, epsilon):
    _, epos = pyq.vmc(wf, epos, nblocks=1, nsteps=2, tstep=1)  # move off node

    for k, item in testwf.test_updateinternals(wf, epos).items():
        print(k, item)
        assert item < epsilon

    testwf.test_mask(wf, 0, epos.electron(0))
    try:
        testwf.test_testvalue_many(wf, epos)
    except AttributeError as err:
        print(err)

    for fname, func in zip(
        ["gradient", "laplacian", "pgradient"],
        [
            testwf.test_wf_gradient,
            testwf.test_wf_laplacian,
            testwf.test_wf_pgradient,
        ],
    ):
        err = [func(wf, epos, delta) for delta in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]]
        assert min(err) < epsilon, f"{fname} error {err}"

    for fname, func in zip(
        ["gradient_value", "gradient_laplacian"],
        [
            testwf.test_wf_gradient_value,
            testwf.test_wf_gradient_laplacian,
        ],
    ):
        d = func(wf, epos)
        for k, v in d.items():
            assert v < 1e-10, (wf, fname, k, v)


def test_obc_wfs(LiH_sto3g_rhf, epsilon=1e-5, nconf=11):
    """
    Ensure that the wave function objects are consistent in several situations.
    """

    mol, mf = LiH_sto3g_rhf
    a_basis, b_basis = default_jastrow_basis(mol)
    for wf in [
        generate_gps_jastrow(mol)[0],
        MultiplyWF(Slater(mol, mf), generate_gps_jastrow(mol)[0]),
        generate_jastrow(mol)[0],
        GeminalJastrow(mol),
        ThreeBodyJastrow(mol, a_basis, b_basis),
        MultiplyWF(Slater(mol, mf), generate_jastrow(mol)[0]),
        MultiplyWF(Slater(mol, mf), generate_jastrow(mol)[0], GeminalJastrow(mol)),
        MultiplyWF(
            Slater(mol, mf),
            generate_jastrow(mol)[0],
            ThreeBodyJastrow(mol, a_basis, b_basis),
        ),
        Slater(mol, mf),
        GPSJastrow(mol, pyq.initial_guess(mol, 2).configs.reshape(-1, 2, 3)),
        JAXSlater(mol, mf),
        JAXJastrowSpin(mol),
    ]:
        for k in wf.parameters.keys():
            if k != "mo_coeff":
                wf.parameters[k] = cp.asarray(np.random.rand(*wf.parameters[k].shape))

        epos = pyq.initial_guess(mol, nconf)
        print(type(wf))
        _, epos = pyq.vmc(wf, epos, nblocks=1, nsteps=2, tstep=1)  # move off node
        run_tests(wf, epos, epsilon)


def test_pbc_wfs(H_pbc_sto3g_krks, epsilon=1e-5, nconf=10):
    """
    Ensure that the wave function objects are consistent in several situations.
    """
    mol, mf = H_pbc_sto3g_krks

    supercell = pyq.get_supercell(mol, S=(np.ones((3, 3)) - 2 * np.eye(3)))
    epos = pyq.initial_guess(supercell, nconf)
    for wf in [
        MultiplyWF(Slater(supercell, mf, eval_gto_precision=1e-6), generate_gps_jastrow(supercell)[0]),
        MultiplyWF(Slater(supercell, mf, eval_gto_precision=1e-6), generate_jastrow(supercell)[0]),
        MultiplyWF(
            Slater(supercell, mf, eval_gto_precision=1e-6),
            generate_jastrow(supercell)[0],
            GeminalJastrow(supercell, eval_gto_precision=1e-6),
        ),
        Slater(supercell, mf, eval_gto_precision=1e-6),
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
        MultiplyWF(Slater(supercell, mf, eval_gto_precision=1e-6), generate_jastrow(supercell)[0]),
        Slater(supercell, mf, eval_gto_precision=1e-6),
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
    from pyqmc.pyscftools import single_determinant_from_mf

    mol, mf = H_pbc_sto3g_kuks
    supercell = np.identity(3, dtype=int)
    supercell[0, 0] = 2
    mol = pyq.get_supercell(mol, supercell)

    determinants = [
        single_determinant_from_mf(mf, 1.0)[0],
        single_determinant_from_mf(mf, -0.2)[0],
    ]
    for s, ka, a, ki, i in [(0, 0, 0, 0, 1)]:
        determinants[1][1][s][ka].remove(a)
        determinants[1][1][s][ki].append(i)

    print(determinants[0])
    wf = Slater(mol, mf, determinants=determinants, eval_gto_precision=1e-6)
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
