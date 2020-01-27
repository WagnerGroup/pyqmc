import numpy as np
import scipy
import scipy.spatial
import pyqmc.eval_ecp as eval_ecp
from pyqmc.distance import RawDistance


def ee_energy(configs):
    ne = configs.configs.shape[1]
    if ne == 1:
        return np.zeros(configs.configs.shape[0])
    ee = np.zeros(configs.configs.shape[0])
    ee, ij = configs.dist.dist_matrix(configs.configs)
    ee = np.linalg.norm(ee, axis=2)
    return np.sum(1.0 / ee, axis=1)


def ei_energy(mol, configs):
    ei = 0.0
    for c, coord in zip(mol.atom_charges(), mol.atom_coords()):
        delta = configs.configs - coord[np.newaxis, np.newaxis, :]
        deltar = np.sqrt(np.sum(delta ** 2, axis=2))
        ei += -c * np.sum(1.0 / deltar, axis=1)
    return ei


def ii_energy(mol):
    ei = 0.0
    d = RawDistance()
    rij, ij = d.dist_matrix(mol.atom_coords()[np.newaxis, :, :])
    if len(ij) == 0:
        return np.array([0.0])
    rij = np.linalg.norm(rij, axis=2)[0, :]
    iitot = 0
    c = mol.atom_charges()
    for (i, j), r in zip(ij, rij):
        iitot += c[i] * c[j] / r
    return iitot


def get_ecp(mol, configs, wf, threshold):
    return eval_ecp.ecp(mol, configs, wf, threshold)


def kinetic(configs, wf):
    nconf, nelec, ndim = configs.configs.shape
    ke = np.zeros(nconf)
    for e in range(nelec):
        ke += -0.5 * np.real(wf.laplacian(e, configs.electron(e)))
    return ke


def energy(mol, configs, wf, threshold):
    """Compute the local energy of a set of configurations.
    
    Args:
      mol: A pyscf-like 'Mole' object. nelec, atom_charges(), atom_coords(), and ._ecp are used.

      configs: a nconfiguration x nelectron x 3 numpy array
       
      wf: A Wavefunction-like object. Functions used include recompute(), lapacian(), and testvalue()

    Returns: 
      a dictionary with energy components ke, ee, ei, and total
      """
    ee = ee_energy(configs)
    ei = ei_energy(mol, configs)
    ecp_val = get_ecp(mol, configs, wf, threshold)
    ii = ii_energy(mol)
    ke = kinetic(configs, wf)
    # print(ke,ee,ei,ii)
    return {
        "ke": ke,
        "ee": ee,
        "ei": ei,
        "ecp": ecp_val,
        "total": ke + ee + ei + ecp_val + ii,
    }
