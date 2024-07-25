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

# what should the structure for orbital evaluation look like?
import numpy as np
import scipy.special
from numba import njit, jit
import pyqmc.spherical_harmonics as hsh
import pyqmc.distance


@njit(cache=False, fastmath=True)
def eval_spherical(max_l, rvec):
    """
    evaluate spherical harmonics up to angular momentum max_l
    max_l: (int)
    rvec: (nvec, 3) points to evaluate at
    """
    out = np.zeros(((max_l + 1)**2, rvec.shape[0]))

    hsh.HARDCODED_SPH_MACRO(max_l, rvec[:, 0], rvec[:, 1], rvec[:, 2], rvec[:, 0]**2, rvec[:, 1]**2, rvec[:, 2]**2, out)
    return out
    
@njit(cache=False, fastmath=True)
def eval_spherical_grad(max_l, rvec):
    """
    evaluate spherical harmonic gradients up to angular momentum max_l
    max_l: (int)
    rvec: (nvec, 3) points to evaluate at
    """
    out = np.zeros((4, (max_l + 1)**2, rvec.shape[0]))
    a, b, c, d = out

    hsh.HARDCODED_SPH_DERIVATIVE_MACRO(max_l, rvec[:, 0], rvec[:, 1], rvec[:, 2], rvec[:, 0]**2, rvec[:, 1]**2, rvec[:, 2]**2, a, b, c, d)
    out = np.transpose(out, (1, 0, 2))
    return out
    

@njit
def mol_eval_gto(all_rvec, basis_ls, basis_arrays, max_l, splits, l_splits):
    """
    all_rvec: (natom, nelec, 3) atom-electron distances
    basis_ls: (ncontractions,) l value for every Gaussian contraction (concatenated together)
    basis_arrays: (ngaussians, 2) contraction coefficients for all Gaussian contractions (concatenated together)
    max_l: (natom,) max angular momentum l for each atom
    splits: (ncontractions+1,) indexing for basis_arrays
    l_splits: (natom+1,) indexing for basis_ls
    """
    nbas_tot = np.sum(2 * basis_ls + 1)
    ao = np.zeros((nbas_tot, all_rvec.shape[1]))
    sel = 0
    split = 0

    for a, rvec in enumerate(all_rvec):
        r2 = np.zeros(rvec.shape[0])
        for e, v in enumerate(rvec):
            for i in range(3):
                r2[e] += v[i]**2
        #r2 = np.sum(rvec**2, axis=-1)
        spherical = eval_spherical(max_l[a], rvec)
        # this loops over all basis functions for the atom
        b_ind = 0
        for l in basis_ls[l_splits[a]:l_splits[a+1]]:
            bas = basis_arrays[splits[split]:splits[split+1]]
            nbas = (2 * l + 1)
            rad = radial_gto(r2, bas)
            for b in range(nbas):
                ao[sel+b_ind] = spherical[l*l+b] * rad
                b_ind += 1
            #ao[sel:sel+nbas] += spherical[l**2:(l+1)**2] * rad
            split += 1
        sel += b_ind
    return np.transpose(ao)

@njit
def mol_eval_gto_grad(all_rvec, basis_ls, basis_arrays, max_l, splits, l_splits):
    """
    all_rvec: (natom, nelec, 3) atom-electron distances
    basis_ls: (ncontractions,) l value for every Gaussian contraction (concatenated together)
    basis_arrays: (ngaussians, 2) contraction coefficients for all Gaussian contractions (concatenated together)
    max_l: (natom,) max angular momentum l for each atom
    splits: (ncontractions+1,) indexing for basis_arrays
    l_splits: (natom+1,) indexing for basis_ls
    """
    nbas_tot = np.sum(2 * basis_ls + 1)
    ao = np.zeros((nbas_tot, 4, all_rvec.shape[1]))
    sel = 0
    split = 0

    for a, rvec in enumerate(all_rvec):
        r2 = np.zeros(rvec.shape[0])
        for e, v in enumerate(rvec):
            for i in range(3):
                r2[e] += v[i]**2
        #r2 = np.sum(rvec**2, axis=-1)
        spherical = eval_spherical_grad(max_l[a], rvec)
        # this loops over all basis functions for the atom
        b_ind = 0
        for l in basis_ls[l_splits[a]:l_splits[a+1]]:
            bas = basis_arrays[splits[split]:splits[split+1]]
            nbas = (2 * l + 1)
            rad = radial_gto_grad(r2, rvec, bas)
            for b in range(nbas):
                for e in range(rad.shape[1]):
                    ao[sel+b_ind, 0, e] = spherical[l*l+b, 0, e] * rad[0, e]
                for i in range(1, 4):
                    for e in range(rad.shape[1]):
                        ao[sel+b_ind, i, e] = spherical[l*l+b, i, e] * rad[0, e] + spherical[l*l+b, 0, e] * rad[i, e]
                b_ind += 1
            split += 1
        sel += b_ind
    return np.transpose(ao, (1, 2, 0))


@njit
def mol_eval_gto_lap(all_rvec, basis_ls, basis_arrays, max_l, splits, l_splits):
    """
    all_rvec: (natom, nelec, 3) atom-electron distances
    basis_ls: (ncontractions,) l value for every Gaussian contraction (concatenated together)
    basis_arrays: (ngaussians, 2) contraction coefficients for all Gaussian contractions (concatenated together)
    max_l: (natom,) max angular momentum l for each atom
    splits: (ncontractions+1,) indexing for basis_arrays
    l_splits: (natom+1,) indexing for basis_ls
    """
    nbas_tot = np.sum(2 * basis_ls + 1)
    ao = np.zeros((nbas_tot, 5, all_rvec.shape[1]))
    sel = 0
    split = 0

    for a, rvec in enumerate(all_rvec):
        r2 = np.zeros(rvec.shape[0])
        for e, v in enumerate(rvec):
            for i in range(3):
                r2[e] += v[i]**2
        #r2 = np.sum(rvec**2, axis=-1)
        spherical = eval_spherical_grad(max_l[a], rvec)
        # this loops over all basis functions for the atom
        b_ind = 0
        for l in basis_ls[l_splits[a]:l_splits[a+1]]:
            bas = basis_arrays[splits[split]:splits[split+1]]
            rad = radial_gto_lap(r2, rvec, bas)
            for b in range(2*l+1):
                ao[sel+b_ind, 0] = spherical[l*l+b, 0] * rad[0]
                ao[sel+b_ind, 4] = spherical[l*l+b, 0] * rad[4]
                for i in range(1, 4):
                    ao[sel+b_ind, i] = spherical[l*l+b, i] * rad[0] + spherical[l*l+b, 0] * rad[i]
                    ao[sel+b_ind, 4] += 2 * spherical[l*l+b, i] * rad[i]
                b_ind += 1
            split += 1
        sel += b_ind
    return np.transpose(ao, (1, 2, 0))


@njit("float64[:](float64[:], float64[:, :])", fastmath=True)
def radial_gto(r2, coeffs):
    """
    Evaluate gaussian contraction (vectorized for molecules)
    r: (n, )
    coeffs: (ncontract, 2)
    l: int
    returns (n, )"""
    out = np.zeros_like(r2)
    for c in coeffs:
        out += np.exp(-r2 * c[0]) * c[1]
    return out


@njit("float64[:, :](float64[:], float64[:, :], float64[:, :])", fastmath=True)
def radial_gto_grad(r2, rvec, coeffs):
    """
    Evaluate gaussian contraction gradient (vectorized for molecules)
    r2: (n, )
    rvec: (n, 3)
    coeffs: (ncontract, 2)
    l: int
    returns (4, n, )"""
    out = np.zeros((4, r2.shape[0]))
    for c in coeffs:
        for a in range(r2.shape[0]):
            tmp = np.exp(-r2[a] * c[0]) * c[1]
            out[0, a] += tmp
            for i in range(3):
                out[i+1, a] +=  -tmp * 2 * c[0] * rvec[a, i]
    return out

@njit("float64[:, :](float64[:], float64[:, :], float64[:, :])", fastmath=True)
def radial_gto_lap(r2, rvec, coeffs):
    """
    Evaluate gaussian contraction laplacian (vectorized for molecules)
    r: (n, )
    coeffs: (ncontract, 2)
    l: int
    returns (5, n, )"""
    out = np.zeros((5, r2.shape[0]))
    for c in coeffs:
        tmp = np.exp(-r2 * c[0]) * c[1]
        out[0] += tmp
        tmpx2xc = tmp * 2 * c[0]
        for i in range(3):
            out[i+1] +=  -tmpx2xc * rvec[:, i]
        out[4] +=  tmpx2xc * (2*c[0] *r2 - 3)
    return out

 
@njit("float64(float64, float64[:, :])", fastmath=True)
def single_radial_gto(r2, coeffs):
    """
    r2: float
    coeffs: (ncontract, 2)
    l: int
    returns: float"""
    out = 0.
    for c in coeffs:
        out += np.exp(-r2 * c[0]) * c[1]
    return out


@njit("float64[:](float64, float64[:], float64[:, :])", fastmath=True)
def single_radial_gto_grad(r2, rvec, coeffs):
    """
    Value (0) and three gradient components (1, 2, 3)
    r2: float
    rvec: (3,)
    coeffs: (ncontract, 2)
    l: int
    returns: (4,)"""
    out = np.zeros(4)
    for c in coeffs:
        tmp = np.exp(-r2 * c[0]) * c[1]
        out[0] += tmp
        for i in range(3):
            out[i+1] +=  -tmp * c[0] * 2 * rvec[i]
    return out


@njit("float64[:](float64, float64[:], float64[:, :])", fastmath=True)
def single_radial_gto_lap(r2, rvec, coeffs):
    """
    Evaluate gaussian contraction laplacian (vectorized for molecules)
    r: (n, )
    coeffs: (ncontract, 2)
    l: int
    returns (5, n, )"""
    out = np.zeros(5)
    for c in coeffs:
        tmp = np.exp(-r2 * c[0]) * c[1]
        out[0] += tmp
        for i in range(3):
            out[i+1] +=  -tmp*2*c[0] * rvec[i]
        out[4] +=  tmp*2*c[0] * (2*c[0] *r2 - 3)
    return out

 
def normalize_basis_coeffs(basis):
    """
    https://pyscf.org/pyscf_api_docs/pyscf.gto.html#pyscf.gto.mole.gto_norm
    Compute normalization constants for each Gaussian coefficient
    z = l+1
    N^2 = gamma(2z) / gamma(z) * np.pi**.5 / (2**(2z+1) * (2a)**(z+.5))
    H. B. Schlegel and M. J. Frisch, Int. J. Quant. Chem., 54(1995), 83-87.

    https://en.wikipedia.org/wiki/Gamma_function#Properties
    gamma(2z) / gamma(z) = gamma(z + 1/2) / (2**(1-2*z) * np.pi**.5)

    plug in:
    N^2 = gamma(z + 1/2) / (2**(2z+1) * (2a)**(z+.5) * 2**(1-2z))
        = gamma(z + 1/2) / (4 * (2a)**(z+1/2))
    m = z + .5 = l + 1.5
    N^2 = gamma(m) / (4 * (2*a)**m)
    """
     
    basis_coeffs = []
    for l, coeffs in basis:
        m = l + 1.5
        b = 2 * coeffs[:, 0]
        b_g_int = 2 * b ** m / scipy.special.gamma(m)
        s1 = np.sqrt(b_g_int)
        cs = coeffs[:, 1] * s1

        ee = coeffs[:, 0] + coeffs[:, 0, np.newaxis]
        g_int = scipy.special.gamma(m) / (2 * ee ** m )
        s1 = 1. / np.sqrt(np.einsum('p,pq,q->', cs, g_int, cs))
        basis_coeffs.append(cs * s1)
    return basis_coeffs
            

@njit
def mol_cutoffs(basis_ls, basis_arrays, splits, l_splits, expcutoff=20):
    """
    compute cutoffs for Gaussian exponent. not used for molecules now.
    """
    natom = len(l_splits) - 1

    split = 0
    atom_cutoff = np.zeros(natom)
    l_cutoff = np.zeros(len(basis_ls))
    for a in range(natom):
        for i, l in enumerate(basis_ls[l_splits[a]:l_splits[a+1]]):
            bas = basis_arrays[splits[split]:splits[split+1]]
            log_c = np.log(np.abs(bas[:, 1]))
            if l==0:
                l_cutoff[l_splits[a]+i] = np.amax((expcutoff + log_c) / bas[:, 0])
            else:
                r2sup = .5 * l / np.amin(bas[:, 0])
                lconst = .5 * np.log(r2sup) * l 
                l_cutoff[l_splits[a]+i] = np.amax((expcutoff + log_c + lconst) / bas[:, 0])
            atom_cutoff[a] = max(atom_cutoff[a], l_cutoff[l_splits[a]+i])
            split += 1
    return atom_cutoff, l_cutoff


class AtomicOrbitalEvaluator:
    def __init__(self, mol, expcutoff=15):
        # Need atom coords, basis set
        self.atom_coords = mol.atom_coords()
        natom = len(self.atom_coords)
        self.atom_names = [mol.atom_pure_symbol(i) for i in range(natom)]
        list2arr = lambda v: [[x[0], np.array(x[1:])] for x in v]

        basis_ls = {}
        basis_arrays = {}
        _splits = {}
        self.basis_coeffs = {}
        for k, v in mol._basis.items():
            array_basis = list2arr(v)# for k, v in mol._basis.items()}
            ls, bases = list(zip(*array_basis))
            basis_ls[k] = np.asarray(ls)
            basis_coeffs = normalize_basis_coeffs(array_basis)
            self.basis_coeffs[k]=basis_coeffs
            for bas, norm in zip(bases, basis_coeffs):
                bas[:, 1] = norm
            _splits[k] = np.array([len(b) for b in bases])
            basis_arrays[k] = np.concatenate(bases, axis=0)
        max_l = {k: np.amax(zip(*v).__next__()) for k, v in mol._basis.items()}

        
        self.basis_ls = np.concatenate([basis_ls[atom] for atom in self.atom_names])
        self.basis_arrays = np.concatenate([basis_arrays[atom] for atom in self.atom_names])
        splits = np.concatenate([[0]] + [_splits[atom] for atom in self.atom_names])
        self.splits = np.cumsum(splits)
        self.max_l = np.asarray([max_l[atom] for atom in self.atom_names])
        #self.nbas_atom = np.asarray([np.sum(2 * ls + 1) for ls in self.basis_ls])
        self.l_splits = np.cumsum([0] + [len(basis_ls[atom]) for atom in self.atom_names])
        #self.nbas = np.sum(self.nbas_atom)

        self.dtype = float

        self.dist = pyqmc.distance.RawDistance()
        self.gto_func = dict(
            GTOval_sph=mol_eval_gto,
            GTOval_sph_deriv1=mol_eval_gto_grad,
            GTOval_sph_deriv2=mol_eval_gto_lap,
        )

        self.atom_cutoff, self.l_cutoff = mol_cutoffs(
            self.basis_ls, 
            self.basis_arrays, 
            self.splits, 
            self.l_splits, 
            expcutoff=expcutoff,
        )

    def eval_gto(self, eval_str, configs):
        """
        configs is (n, 3)
        """
        # (natom, nconf, 3)
        rvec = self.dist.pairwise(self.atom_coords[np.newaxis], configs[np.newaxis])[0]
        return self.gto_func[eval_str](
            rvec,
            self.basis_ls, 
            self.basis_arrays,
            self.max_l,
            self.splits,
            self.l_splits,
        )

