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
from pyqmc.observables.accumulators import (
    EnergyAccumulator,
    LinearTransform,
    SqAccumulator,
    SymmetryAccumulator,
)
from pyqmc.observables.obdm import OBDMAccumulator
from pyqmc.observables.tbdm import TBDMAccumulator
import pyqmc.api as pyq
import copy


def test_transform(LiH_sto3g_rhf):
    """Tests that the shapes are ok"""
    mol, mf = LiH_sto3g_rhf
    wf, to_opt = pyq.generate_wf(mol, mf)
    transform = LinearTransform(wf.parameters)
    x = transform.serialize_parameters(wf.parameters)
    nconfig = 10
    configs = pyq.initial_guess(mol, nconfig)
    wf.recompute(configs)
    pgrad = wf.pgradient()
    gradtrans = transform.serialize_gradients(pgrad)
    assert gradtrans.shape[1] == len(x)
    assert gradtrans.shape[0] == nconfig


def test_info_functions_mol(LiH_sto3g_rhf):
    mol, mf = LiH_sto3g_rhf
    wf, to_opt = pyq.generate_wf(mol, mf)
    reflection_yz = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    symmetry_operators = {"reflection_yz": reflection_yz}
    accumulators = {
        "pgrad": pyq.gradient_generator(mol, wf, to_opt),
        "obdm": OBDMAccumulator(mol, orb_coeff=mf.mo_coeff),
        "tbdm_updown": TBDMAccumulator(mol, np.asarray([mf.mo_coeff] * 2), (0, 1)),
        "symmetry": SymmetryAccumulator(symmetry_operators=symmetry_operators),
    }
    info_functions(mol, wf, accumulators)


def test_info_functions_pbc(H_pbc_sto3g_krks):
    from pyqmc.pbc.supercell import get_supercell

    mol, mf = H_pbc_sto3g_krks
    kinds = [0, 1]
    dm_orbs = [mf.mo_coeff[i][:, :2] for i in kinds]
    wf, to_opt = pyq.generate_wf(mol, mf)
    accumulators = {
        "pgrad": pyq.gradient_generator(mol, wf, to_opt, ewald_gmax=10),
        "obdm": OBDMAccumulator(mol, dm_orbs, kpts=mf.kpts[kinds]),
        "Sq": SqAccumulator(mol),
    }
    info_functions(mol, wf, accumulators)


def info_functions(mol, wf, accumulators):
    accumulators["energy"] = accumulators["pgrad"].enacc
    configs = pyq.initial_guess(mol, 100)
    wf.recompute(configs)
    for k, acc in accumulators.items():
        shapes = acc.shapes()
        keys = acc.keys()
        assert shapes.keys() == keys, "keys: {0}\nshapes: {1}".format(keys, shapes)
        avg = acc.avg(configs, wf)
        assert avg.keys() == keys, (k, avg.keys(), keys)
        for ka in keys:
            assert shapes[ka] == avg[ka].shape, "{0} {1}".format(ka, avg[ka].shape)


if __name__ == "__main__":
    test_info_functions_mol()
    test_info_functions_pbc()
