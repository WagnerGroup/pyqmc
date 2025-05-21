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
from pyqmc.pbc.pbc import enforce_pbc


def test_enforce_pbcs():
    # TEST 1: Check if any electron in new config
    #         is out of the simulation box for set
    #         of non-orthogonal lattice vectors. We
    #         do a controlled comparison between
    #         initial configs and final ones.
    nconf = 7
    lattvecs = np.array(
        [[1.2, 0, 0], [0.6, 1.2 * np.sqrt(3) / 2, 0], [0, 0, 0.8]]
    )  # Triangular lattice
    trans = (
        np.array(
            [
                [0.1, 0.1, 0.1],
                [1.3, 0, 0.2],
                [0.9, 1.8 * np.sqrt(3) / 2, 0],
                [0, 0, 1.1],
                [2.34, 1.35099963, 0],
                [0.48, 1.24707658, 0],
                [-2.52, 2.28630707, -0.32],
            ]
        )
        + 1e-14
    )
    check_final = (
        np.array(
            [
                [0.1, 0.1, 0.1],
                [0.1, 0, 0.2],
                [0.3, 0.6 * np.sqrt(3) / 2, 0.0],
                [0, 0, 0.3],
                [0.54, 0.31176915, 0],
                [1.08, 0.2078461, 0],
                [1.08, 0.2078461, 0.48],
            ]
        )
        + 1e-14
    )
    check_wrap = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [-1, 1, 0], [-4, 2, -1]]
    )
    final_trans, final_wrap = enforce_pbc(lattvecs, trans)
    # Checks output configs
    relative_tol = 1e-8
    absolute_tol = 1e-8
    test1a = np.all(
        np.isclose(final_trans, check_final, rtol=relative_tol, atol=absolute_tol)
    )
    # Checks wraparound matrix
    test1b = np.all(
        np.isclose(final_wrap, check_wrap, rtol=relative_tol, atol=absolute_tol)
    )
    test1 = test1a * test1b
    assert test1


def test_non_orthogonal():
    # TEST 2: Check if any electron in new config
    #         is out of the simulation box for set
    #         of non-orthogonal lattice vectors.
    nconf = 50
    lattvecs = np.array(
        [[1.2, 0, 0], [0.6, 1.2 * np.sqrt(3) / 2, 0], [0, 0, 0.8]]
    )  # Triangular lattice
    recpvecs = np.linalg.inv(lattvecs)
    # Old config
    epos = np.random.random((nconf, 3))
    epos = np.einsum("ij,jk->ik", epos, lattvecs)
    # New config
    step = 0.5
    trans = epos + step * (np.random.random((nconf, 3)) - 0.5 * np.ones((nconf, 3)))
    final_trans, wrap = enforce_pbc(lattvecs, trans)
    # Configs in lattice vectors basis
    ff = np.einsum("ij,jk->ik", final_trans, recpvecs)
    test2 = np.all(ff < 1) & np.all(ff >= 0)

    assert test2


if __name__ == "__main__":
    test_enforce_pbcs()
