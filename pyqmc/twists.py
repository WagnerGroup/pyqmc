import numpy as np
import pyqmc.supercell
import pyqmc.api


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


def check_equivalent(kpts, ind, tol=1e-6):
    """Given the index of a kpt in kpts, return the indices of its equivalent k-points in kpts"""
    equiv_ind = []
    for i in range(len(kpts)):
        kdiffs = np.mod(kpts[i] - kpts[ind] + 0.5, 1.0) - 0.5
        if np.linalg.norm(kdiffs) < tol:
            equiv_ind.append(i)
    return equiv_ind


def get_qmc_kpts(cell, S, frac_twist):
    """Given the cell, supercell transformation matrix and fractional twist, return the required k-points by a QMC calculation on that supercell"""
    twist = get_twist(cell, S, frac_twist)
    supercell = pyqmc.supercell.get_supercell(cell, S)
    qmc_kpts = pyqmc.supercell.get_supercell_kpts(supercell) + twist
    return qmc_kpts


def get_k_indices(cell, mf_kpts, kpts, tol=1e-6):
    """Given a list of kpts, return inds such that mf_kpts[inds] is a list of kpts equivalent to the input list"""
    kdiffs = mf_kpts[np.newaxis] - kpts[:, np.newaxis]
    frac_kdiffs = np.dot(kdiffs, cell.lattice_vectors().T) / (2 * np.pi)
    kdiffs = np.mod(frac_kdiffs + 0.5, 1.0) - 0.5
    return np.nonzero(np.linalg.norm(kdiffs, axis=-1) < tol)[1]


def available_twists(cell, mf, S):
    """Given the primitive cell and mf object from a DFT calculation and a supercell transformation matrix, return all valid twists (fractional) for a QMC calculation"""
    ind = 0
    unique_frac_kpts = np.array([get_frac_twist(cell, S, kpt) for kpt in mf.kpts])
    while ind < len(unique_frac_kpts):
        equiv_ind = check_equivalent(unique_frac_kpts, ind)
        if len(equiv_ind) > 1:
            unique_frac_kpts = np.delete(unique_frac_kpts, equiv_ind[1:], 0)
        ind += 1
    avail_twists = []
    for frac_twist in unique_frac_kpts:
        qmc_kpts = get_qmc_kpts(cell, S, frac_twist)
        if len(get_k_indices(cell, mf.kpts, qmc_kpts)) == qmc_kpts.shape[0]:
            avail_twists.append(frac_twist)
    return np.array(avail_twists)
