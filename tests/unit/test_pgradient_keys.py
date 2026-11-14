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

"""pgradient(keys): only the requested gradients need to be computed."""
import copy

import numpy as np
import pytest

import pyqmc.api as pyq
from pyqmc.observables.accumulators import LinearTransform
from pyqmc.wf.addwf import AddWF
from pyqmc.wf.multiplywf import MultiplyWF
from pyqmc.wf.slater import Slater


@pytest.fixture(scope="module")
def casci_slater(H2_casci):
    mol, mf, mc = H2_casci
    mc = copy.copy(mc)
    mc.ci = mc.ci[0]
    return mol, lambda: Slater(mol, mf, mc=mc, tol=1e-20)


def _count_aos(slater):
    """Count the AO evaluations of all electrons (the ones pgradient does for mo_coeff)."""
    calls = []
    aos = slater.orbitals.aos

    def counting(eval_str, configs, mask=None):
        calls.append(configs.configs.shape)
        return aos(eval_str, configs, mask)

    slater.orbitals.aos = counting
    return calls


def test_slater_skips_orbitals(casci_slater):
    mol, make = casci_slater
    wf = make()
    wf.recompute(pyq.initial_guess(mol, 10))
    calls = _count_aos(wf)
    pgrad = wf.pgradient(["det_coeff"])
    assert set(pgrad) == {"det_coeff"}
    assert len(calls) == 0
    full = wf.pgradient()
    assert len(calls) == 2  # one per spin
    assert np.allclose(full["det_coeff"], pgrad["det_coeff"])
    beta = wf.pgradient(["mo_coeff_beta"])
    assert set(beta) == {"det_coeff", "mo_coeff_beta"}
    assert calls[-1][1] == mol.nelec[1]  # only the beta electrons were evaluated
    assert np.allclose(beta["mo_coeff_beta"], full["mo_coeff_beta"])


def test_nested_multiplywf_routing(casci_slater):
    mol, make = casci_slater
    slater = make()
    wf = MultiplyWF(MultiplyWF(slater, pyq.generate_jastrow(mol)[0]), pyq.generate_jastrow(mol)[0])
    wf.recompute(pyq.initial_guess(mol, 10))
    calls = _count_aos(slater)
    pgrad = wf.pgradient(["wf1wf1det_coeff", "wf2bcoeff"])
    assert len(calls) == 0
    assert "wf1wf1det_coeff" in set(pgrad.keys())
    assert "wf1wf1mo_coeff_alpha" not in set(pgrad.keys())
    wf.pgradient(["wf1wf1mo_coeff_alpha"])
    assert len(calls) == 1 and calls[0][1] == mol.nelec[0]


def test_addwf_routing(casci_slater):
    mol, make = casci_slater
    s1, s2 = make(), make()
    wf = AddWF(np.array([0.7, 0.3]), [s1, s2])
    wf.recompute(pyq.initial_guess(mol, 10))
    calls1, calls2 = _count_aos(s1), _count_aos(s2)
    pgrad = wf.pgradient(["wf1det_coeff", "wf2mo_coeff_alpha"])
    assert len(calls1) == 0 and len(calls2) == 1  # alpha only
    full = wf.pgradient()
    assert np.allclose(pgrad["wf1det_coeff"], full["wf1det_coeff"])
    assert np.allclose(pgrad["wf2mo_coeff_alpha"], full["wf2mo_coeff_alpha"])


@pytest.mark.parametrize("orbitals", [False, True])
def test_serialized_gradient_unchanged(casci_slater, orbitals):
    mol, make = casci_slater
    slater = make()
    wf = MultiplyWF(slater, pyq.generate_jastrow(mol)[0])
    to_opt = {
        "wf1det_coeff": np.ones(wf.parameters["wf1det_coeff"].shape, dtype=bool),
        "wf2bcoeff": np.ones(wf.parameters["wf2bcoeff"].shape, dtype=bool),
    }
    if orbitals:
        to_opt["wf1mo_coeff_alpha"] = np.ones(wf.parameters["wf1mo_coeff_alpha"].shape, dtype=bool)
    transform = LinearTransform(wf.parameters, to_opt)
    wf.recompute(pyq.initial_guess(mol, 10))
    calls = _count_aos(slater)
    dp = transform.serialize_pgradient(wf)
    assert len(calls) == int(orbitals)  # alpha only
    assert np.allclose(dp, transform.serialize_gradients(wf.pgradient()))
