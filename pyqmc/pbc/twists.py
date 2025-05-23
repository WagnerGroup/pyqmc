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
import pyqmc.pbc.supercell
import pyqmc.pbc.pbc


def get_twist(cell, S, frac_twist):
    """Given the twist in unit of supercell reciprocal primitive lattice vectors and the supercell, return the twist in Bohr^-1"""
    supercell = pyqmc.pbc.supercell.get_supercell(cell, S)
    frac_twist = np.mod(frac_twist + 0.5, 1.0) - 0.5
    twist = np.dot(np.linalg.inv(supercell.a), frac_twist) * 2 * np.pi
    return twist


def get_frac_twist(cell, S, twist):
    """Given the twist in Bohr^-1 and the supercell, return the twist in supercell reciprocal primitive lattice vectors"""
    supercell = pyqmc.pbc.supercell.get_supercell(cell, S)
    frac_twist = np.dot(supercell.a, twist) / (2 * np.pi)
    frac_twist = np.mod(frac_twist + 0.5, 1.0) - 0.5
    return frac_twist


def create_supercell_twists(supercell, mf, tol=12):
    """

    Args:
    supercell: a supercell object made from pyqmc.supercell.make_supercell
    mf: a mf object with KRKS
    tol: number of decimals to round the k-points. Default should be ok for most meshes

    Returns:
    dictionary:
       twists: a list of ks's
       counts: number of k-points in the primitive cell that correspond to the twist
       kinds: indices of the k-points that correspond to the twist
    """
    kpts = mf.kpts
    super_reciprocal_vectors = supercell.reciprocal_vectors()
    frac = kpts @ np.linalg.inv(super_reciprocal_vectors)
    frac = np.around(frac, tol) % 1
    super_kpts = frac @ super_reciprocal_vectors

    twists, indices, counts = np.unique(
        np.round(super_kpts, tol), axis=0, return_counts=True, return_inverse=True
    )
    kinds = []
    for i in range(twists.shape[0]):
        k = np.argwhere(indices == i)
        kinds.append(k[:, 0])

    return {"twists": twists, "counts": counts, "primitive_ks": kinds}
