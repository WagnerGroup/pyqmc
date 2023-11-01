import numpy as np
import pyqmc.supercell
import pyqmc.pbc


def get_twist(cell, S, frac_twist):
    """Given the twist in unit of supercell reciprocal primitive lattice vectors and the supercell, return the twist in Bohr^-1"""
    supercell = pyqmc.supercell.get_supercell(cell, S)
    frac_twist = np.mod(frac_twist + 0.5, 1.0) - 0.5
    twist = np.dot(np.linalg.inv(supercell.a), frac_twist) * 2 * np.pi
    return twist


def get_frac_twist(cell, S, twist):
    """Given the twist in Bohr^-1 and the supercell, return the twist in supercell reciprocal primitive lattice vectors"""
    supercell = pyqmc.supercell.get_supercell(cell, S)
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
    super_kpts, wraparound = pyqmc.pbc.enforce_pbc(super_reciprocal_vectors, kpts)
    twists, indices, counts = np.unique(
        np.round(super_kpts, tol), axis=0, return_counts=True, return_inverse=True
    )
    kinds = []
    for i in range(twists.shape[0]):
        k = np.argwhere(indices == i)
        kinds.append(k[:, 0])

    return {"twists": twists, "counts": counts, "primitive_ks": kinds}
