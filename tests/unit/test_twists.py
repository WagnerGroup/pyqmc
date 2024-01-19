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
import pyqmc.twists
import pyqmc.supercell


def run_tests(cell, mf, S, n):
    cell = pyqmc.supercell.get_supercell(cell, S)
    twists = pyqmc.twists.create_supercell_twists(cell, mf)
    print(twists)
    assert (
        twists["twists"].shape[0] == n
    ), f"Found {twists['twists'].shape[0]} available twists but should have found {n}"

    assert twists["counts"][0] == cell.scale
    assert twists["primitive_ks"][0].shape[0] == cell.scale


def test_H_pbc_sto3g_krks(H_pbc_sto3g_krks):
    cell, mf = H_pbc_sto3g_krks
    run_tests(cell, mf, 1 * np.eye(3), 8)
    run_tests(cell, mf, 2 * np.eye(3), 1)


def test_h_noncubic_sto3g_triplet(h_noncubic_sto3g_triplet):
    cell, mf = h_noncubic_sto3g_triplet
    run_tests(cell, mf, 1 * np.eye(3), 1)
