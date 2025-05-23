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
from numba import njit
import pyqmc.wf.numba.spherical_harmonics as hsh
import pyqmc.wf.numba.gto as gto
import pyscf.pbc.gto.cell
import pyqmc.wf.orbitals as pyqorb
import pyqmc.configurations.distance


"""
Wrappers for hsh.SPHn: evaluate spherical harmonics through l=2
v: (3,) vector to evaluate
out: ((n+1)**2,) output array
"""
@njit(cache=True, fastmath=True)
def sph0(v, out):
    hsh.SPH0(v[0], v[1], v[2], v[0]**2, v[1]**2, v[2]**2, out)

@njit(cache=True, fastmath=True)
def sph1(v, out):
    hsh.SPH1(v[0], v[1], v[2], v[0]**2, v[1]**2, v[2]**2, out)

@njit(cache=True, fastmath=True)
def sph2(v, out):
    hsh.SPH2(v[0], v[1], v[2], v[0]**2, v[1]**2, v[2]**2, out)

@njit(cache=True, fastmath=True)
def sph3(v, out):
    hsh.SPH3(v[0], v[1], v[2], v[0]**2, v[1]**2, v[2]**2, out)

@njit(cache=True, fastmath=True)
def sph4(v, out):
    hsh.SPH4(v[0], v[1], v[2], v[0]**2, v[1]**2, v[2]**2, out)

@njit(cache=True, fastmath=True)
def sph5(v, out):
    hsh.SPH5(v[0], v[1], v[2], v[0]**2, v[1]**2, v[2]**2, out)

@njit(cache=True, fastmath=True)
def sph0_grad(v, out):
    a, b, c, d = out
    hsh.SPH0_GRAD(v[0], v[1], v[2], v[0]**2, v[1]**2, v[2]**2, a, b, c, d)

@njit(cache=True, fastmath=True)
def sph1_grad(v, out):
    a, b, c, d = out
    hsh.SPH1_GRAD(v[0], v[1], v[2], v[0]**2, v[1]**2, v[2]**2, a, b, c, d)

@njit(cache=True, fastmath=True)
def sph2_grad(v, out):
    a, b, c, d = out
    hsh.SPH2_GRAD(v[0], v[1], v[2], v[0]**2, v[1]**2, v[2]**2, a, b, c, d)

@njit(cache=True, fastmath=True)
def sph3_grad(v, out):
    a, b, c, d = out
    hsh.SPH3_GRAD(v[0], v[1], v[2], v[0]**2, v[1]**2, v[2]**2, a, b, c, d)

@njit(cache=True, fastmath=True)
def sph4_grad(v, out):
    a, b, c, d = out
    hsh.SPH4_GRAD(v[0], v[1], v[2], v[0]**2, v[1]**2, v[2]**2, a, b, c, d)

@njit(cache=True, fastmath=True)
def sph5_grad(v, out):
    a, b, c, d = out
    hsh.SPH5_GRAD(v[0], v[1], v[2], v[0]**2, v[1]**2, v[2]**2, a, b, c, d)


@njit(fastmath=True, cache=False)
def _pbc_eval_gto(all_rvec, basis_ls, basis_arrays, max_l, splits, l_splits, Ls, num_Ls, r2_l_cutoff, r2_cutoff, phases):
    """
    all_rvec: (natom, nelec, 3) atom-electron distances
    basis_ls: (ncontractions,) l value for every Gaussian contraction (concatenated together)
    basis_arrays: (ngaussians, 2) contraction coefficients for all Gaussian contractions (concatenated together)
    max_l: (natom,) max angular momentum l for each atom
    splits: (ncontractions+1,) indexing for basis_arrays
    l_splits: (natom+1,) indexing for basis_ls
    Ls: (nL, 3) list of (sorted) lattice points to sum over
    num_Ls: (natom,) number of Ls to check for each atom
    r2_l_cutoff: (ncontractions,) distance cutoff for each contraction
    r2_cutoff: (natoms,) distance cutoff for each atom
    """
    natom, nelec = all_rvec.shape[:2]
    nbas_tot = np.sum(2 * basis_ls + 1)
    ao = np.zeros((phases.shape[1], all_rvec.shape[1], nbas_tot), dtype=phases.dtype)

    atom_start = np.zeros(natom+1, dtype=np.int32)
    basis_ = []
    bstart = 0
    for a in range(natom):
        basis_ls_a = basis_ls[l_splits[a]:l_splits[a+1]]
        tmp_bas = []
        atom_start[a+1] = atom_start[a]
        for l_ind, l in enumerate(basis_ls_a):
            split = bstart + l_ind
            tmp_bas.append(basis_arrays[splits[split]:splits[split+1]])
            atom_start[a+1] += 2 * l + 1
        basis_.append(tmp_bas)
        bstart += len(basis_ls_a)

    for a, rvec in enumerate(all_rvec):
        basis_ls_a = basis_ls[l_splits[a]:l_splits[a+1]]
        _single_atom(
            ao, 
            rvec, 
            basis_ls_a, 
            basis_[a], 
            l_splits[a], 
            Ls[:num_Ls[a]], 
            r2_l_cutoff, 
            r2_cutoff[a], 
            atom_start[a], 
            max_l[a],
            phases[:num_Ls[a]],
        )
    return ao


@njit(fastmath=True, cache=False)
def _single_atom(ao, rvec, basis_ls_a, basis_a, l_split_a, Ls_a, r2_l_cutoff, cut, astart, max_l, phases):
    """
    Calculate basis functions for one atom

    ao: (1, nelec, nao) output array
    rvec: (nelec, 3) atom-electron distances
    basis_ls_a: (ncontractions,) l value for every Gaussian contraction in this atom's basis
    basis_arrays: (ngaussians, 2) contraction coefficients for all Gaussian contractions in this atom's basis
    l_split_a: (int) starting index for r_l_cutoff
    Ls_a: (nL, 3) list of (sorted) lattice points to check for this atom
    r2_l_cutoff: (ncontractions,) distance cutoff for each contraction
    cut: (float) distance cutoff for this atom
    astart: (int) starting index for ao
    max_l: (int) max angular momentum l for this atom
    """
    if max_l == 0: sph_func = sph0#hsh.SPH2
    elif max_l == 1: sph_func = sph1#hsh.SPH2
    elif max_l == 2: sph_func = sph2#hsh.SPH2
    elif max_l == 3: sph_func = sph3#hsh.SPH3
    elif max_l == 4: sph_func = sph4#hsh.SPH4
    else: sph_func = sph5#hsh.SPH5

    rvec_L = np.zeros(3)
    spherical = np.zeros((max_l+1)**2)
    nbas = np.sum(basis_ls_a * 2 + 1)
    for e, v in enumerate(rvec):
        for j, L in enumerate(Ls_a):
            r2 = 0
            for i in range(3):
                rvec_L[i] = v[i] - L[i]
                r2 += rvec_L[i]**2
            if r2 > cut: continue

            phases_j = phases[j]
            sph_func(rvec_L, spherical)

            # this loops over all basis functions for the atom
            bstart=astart
            for l_ind, l in enumerate(basis_ls_a):#[l_splits[a]:l_splits[a+1]]):
                if r2 < r2_l_cutoff[l_split_a+l_ind]: 
                    rad = gto.single_radial_gto(r2, basis_a[l_ind])
                    for k, phase in enumerate(phases_j):
                        for b in range(2*l+1):
                            ao[k, e, bstart+b] += spherical[l*l+b] * rad * phase
                bstart += 2*l+1


@njit(fastmath=True, cache=False)
def _pbc_eval_gto_grad(all_rvec, basis_ls, basis_arrays, max_l, splits, l_splits, Ls, num_Ls, r2_l_cutoff, r2_cutoff, phases):
    """
    all_rvec: (natom, nelec, 3) atom-electron distances
    basis_ls: (ncontractions,) l value for every Gaussian contraction (concatenated together)
    basis_arrays: (ngaussians, 2) contraction coefficients for all Gaussian contractions (concatenated together)
    max_l: (natom,) max angular momentum l for each atom
    splits: (ncontractions+1,) indexing for basis_arrays
    l_splits: (natom+1,) indexing for basis_ls
    Ls: (nL, 3) list of (sorted) lattice points to sum over
    num_Ls: (natom,) number of Ls to check for each atom
    r2_l_cutoff: (ncontractions,) distance cutoff for each contraction
    r2_cutoff: (natoms,) distance cutoff for each atom
    """
    natom, nelec = all_rvec.shape[:2]
    nbas_tot = np.sum(2 * basis_ls + 1)
    ao = np.zeros((phases.shape[1], all_rvec.shape[1], nbas_tot, 4), dtype=phases.dtype)

    atom_start = np.zeros(natom+1, dtype=np.int32)
    basis_ = []
    bstart = 0
    for a in range(natom):
        basis_ls_a = basis_ls[l_splits[a]:l_splits[a+1]]
        tmp_bas = []
        atom_start[a+1] = atom_start[a]
        for l_ind, l in enumerate(basis_ls_a):
            split = bstart + l_ind
            tmp_bas.append(basis_arrays[splits[split]:splits[split+1]])
            atom_start[a+1] += 2 * l + 1
        basis_.append(tmp_bas)
        bstart += len(basis_ls_a)

    for a, rvec in enumerate(all_rvec):
        basis_ls_a = basis_ls[l_splits[a]:l_splits[a+1]]
        _single_atom_grad(
            ao, 
            rvec, 
            basis_ls_a, 
            basis_[a], 
            l_splits[a], 
            Ls[:num_Ls[a]], 
            r2_l_cutoff, 
            r2_cutoff[a], 
            atom_start[a], 
            max_l[a],
            phases[:num_Ls[a]],
        )
    return np.transpose(ao, (0, 3, 1, 2))


@njit(fastmath=True, cache=False)
def _single_atom_grad(ao, rvec, basis_ls_a, basis_a, l_split_a, Ls_a, r2_l_cutoff, cut, astart, max_l, phases):
    """
    Calculate basis functions for one atom

    ao: (1, nelec, nao) output array
    rvec: (nelec, 3) atom-electron distances
    basis_ls_a: (ncontractions,) l value for every Gaussian contraction in this atom's basis
    basis_arrays: (ngaussians, 2) contraction coefficients for all Gaussian contractions in this atom's basis
    l_split_a: (int) starting index for r_l_cutoff
    Ls_a: (nL, 3) list of (sorted) lattice points to check for this atom
    r2_l_cutoff: (ncontractions,) distance cutoff for each contraction
    cut: (float) distance cutoff for this atom
    astart: (int) starting index for ao
    max_l: (int) max angular momentum l for this atom
    """
    if max_l == 0: sph_func = sph0_grad#hsh.SPH2
    elif max_l == 1: sph_func = sph1_grad#hsh.SPH2
    elif max_l == 2: sph_func = sph2_grad#hsh.SPH2
    elif max_l == 3: sph_func = sph3_grad#hsh.SPH3
    elif max_l == 4: sph_func = sph4_grad#hsh.SPH4
    else: sph_func = sph5_grad#hsh.SPH5

    rvec_L = np.zeros(3)
    spherical = np.zeros((4, (max_l+1)**2))
    nbas = np.sum(basis_ls_a * 2 + 1)
    rad = np.zeros(4)
    for e, v in enumerate(rvec):
        for j, L in enumerate(Ls_a):
            r2 = 0
            for i in range(3):
                rvec_L[i] = v[i] - L[i]
                r2 += rvec_L[i]**2
            if r2 > cut: continue

            phases_j = phases[j]
            sph_func(rvec_L, spherical)

            # this loops over all basis functions for the atom
            b_ind = 0
            for l_ind, l in enumerate(basis_ls_a):#[l_splits[a]:l_splits[a+1]]):
                if r2 > r2_l_cutoff[l_split_a+l_ind]: 
                    b_ind += 2*l+1
                    continue
                gto.single_radial_gto_grad(r2, rvec_L, basis_a[l_ind], rad)
                for b in range(2*l+1):
                    for k, phase in enumerate(phases_j):
                        ao[k, e, astart+b_ind, 0] += spherical[0, l*l+b] * rad[0] * phase
                        for i in range(1, 4):
                            ao[k, e, astart+b_ind, i] += (spherical[i, l*l+b] * rad[0] + spherical[0, l*l+b] * rad[i]) * phase
                    b_ind += 1


@njit(fastmath=True, cache=False)
def _pbc_eval_gto_lap(all_rvec, basis_ls, basis_arrays, max_l, splits, l_splits, Ls, num_Ls, r2_l_cutoff, r2_cutoff, phases):
    """
    all_rvec: (natom, nelec, 3) atom-electron distances
    basis_ls: (ncontractions,) l value for every Gaussian contraction (concatenated together)
    basis_arrays: (ngaussians, 2) contraction coefficients for all Gaussian contractions (concatenated together)
    max_l: (natom,) max angular momentum l for each atom
    splits: (ncontractions+1,) indexing for basis_arrays
    l_splits: (natom+1,) indexing for basis_ls
    Ls: (nL, 3) list of (sorted) lattice points to sum over
    num_Ls: (natom,) number of Ls to check for each atom
    r2_l_cutoff: (ncontractions,) distance cutoff for each contraction
    r2_cutoff: (natoms,) distance cutoff for each atom
    """
    natom, nelec = all_rvec.shape[:2]
    nbas_tot = np.sum(2 * basis_ls + 1)
    ao = np.zeros((phases.shape[1], all_rvec.shape[1], nbas_tot, 5), dtype=phases.dtype)

    atom_start = np.zeros(natom+1, dtype=np.int32)
    basis_ = []
    bstart = 0
    for a in range(natom):
        basis_ls_a = basis_ls[l_splits[a]:l_splits[a+1]]
        tmp_bas = []
        atom_start[a+1] = atom_start[a]
        for l_ind, l in enumerate(basis_ls_a):
            split = bstart + l_ind
            tmp_bas.append(basis_arrays[splits[split]:splits[split+1]])
            atom_start[a+1] += 2 * l + 1
        basis_.append(tmp_bas)
        bstart += len(basis_ls_a)

    for a, rvec in enumerate(all_rvec):
        basis_ls_a = basis_ls[l_splits[a]:l_splits[a+1]]
        _single_atom_lap(
            ao, 
            rvec, 
            basis_ls_a, 
            basis_[a], 
            l_splits[a], 
            Ls[:num_Ls[a]], 
            r2_l_cutoff, 
            r2_cutoff[a], 
            atom_start[a], 
            max_l[a],
            phases[:num_Ls[a]],
        )
    return np.transpose(ao, (0, 3, 1, 2))


@njit(fastmath=True, cache=False)
def _single_atom_lap(ao, rvec, basis_ls_a, basis_a, l_split_a, Ls_a, r2_l_cutoff, cut, astart, max_l, phases):
    """
    Calculate basis functions for one atom

    ao: (1, nelec, nao) output array
    rvec: (nelec, 3) atom-electron distances
    basis_ls_a: (ncontractions,) l value for every Gaussian contraction in this atom's basis
    basis_arrays: (ngaussians, 2) contraction coefficients for all Gaussian contractions in this atom's basis
    l_split_a: (int) starting index for r_l_cutoff
    Ls_a: (nL, 3) list of (sorted) lattice points to check for this atom
    r2_l_cutoff: (ncontractions,) distance cutoff for each contraction
    cut: (float) distance cutoff for this atom
    astart: (int) starting index for ao
    max_l: (int) max angular momentum l for this atom
    """
    if max_l == 0: sph_func = sph0_grad#hsh.SPH2
    elif max_l == 1: sph_func = sph1_grad#hsh.SPH2
    elif max_l == 2: sph_func = sph2_grad#hsh.SPH2
    elif max_l == 3: sph_func = sph3_grad#hsh.SPH3
    elif max_l == 4: sph_func = sph4_grad#hsh.SPH4
    else: sph_func = sph5_grad#hsh.SPH5

    rvec_L = np.zeros(3)
    spherical = np.zeros((4, (max_l+1)**2))
    nbas = np.sum(basis_ls_a * 2 + 1)
    rad = np.zeros(5)
    for e, v in enumerate(rvec):
        for j, L in enumerate(Ls_a):
            r2 = 0
            for i in range(3):
                rvec_L[i] = v[i] - L[i]
                r2 += rvec_L[i]**2
            if r2 > cut: continue

            phases_j = phases[j]
            sph_func(rvec_L, spherical)

            # this loops over all basis functions for the atom
            b_ind = 0
            for l_ind, l in enumerate(basis_ls_a):#[l_splits[a]:l_splits[a+1]]):
                if r2 > r2_l_cutoff[l_split_a+l_ind]: 
                    b_ind += 2*l+1
                    continue
                gto.single_radial_gto_lap(r2, rvec_L, basis_a[l_ind], rad)
                for b in range(2*l+1):
                    for k, phase in enumerate(phases_j):
                        ao[k, e, astart+b_ind, 0] += spherical[0, l*l+b] * rad[0] * phase
                        tmp_lap = spherical[0, l*l+b] * rad[4] 
                        for i in range(1, 4):
                            ao[k, e, astart+b_ind, i] += (spherical[i, l*l+b] * rad[0] + spherical[0, l*l+b] * rad[i]) * phase
                            tmp_lap += 2 * spherical[i, l*l+b] * rad[i]
                        ao[k, e, astart+b_ind, 4] += tmp_lap * phase
                    b_ind += 1


@njit(cache=True)
def max_distance_in_cell(lvecs):
    """
    calculate the maximum possible distance in the cell to estimate how many Ls are needed
    """
    combos = np.array([[1., 1., 1.],
                       [-1., 1., 1.],
                       [1., -1., 1.],
                       [1., 1., -1.]])
    vecs = combos @ lvecs
    distances = np.sum(vecs**2, axis=-1)#np.linalg.norm(vecs, axis=-1)
    i = np.argmax(distances)
    return vecs[i] / 2


@njit(cache=True)
def calc_num_Ls(rvec, Ls, basis_arrays, basis_ls, splits, l_splits, expcutoff):
    """
    Not used
    Calculate number of Ls to sum over for a particular rvec
    """
    res = np.ones(len(splits), dtype=np.int64)
    r2 = np.zeros(rvec.shape[1])
    split = 0
    for a, v in enumerate(rvec):
        for i, l in enumerate(basis_ls[l_splits[a]:l_splits[a+1]]):
            bas = basis_arrays[splits[split]:splits[split+1]]
            for j in np.arange(len(Ls))[::-1]:
                min_exp = expcutoff + 1
                r2[:] = np.sum((v - Ls[j])**2, axis=-1)
                for b in bas:
                    logc = np.log(np.abs(b[1]))
                    # assume r**l < 148**l
                    min_exp = min(min_exp, np.amin( b[0] * r2 - logc))# - 5 * l) 
                if min_exp < expcutoff:
                    res[split] = max(res[split], j+1)
                    break
            split += 1
    return res 


@njit(cache=True)
def max_Ls(Ls, lvecs, basis_ls, basis_arrays, splits, l_splits, expcutoff=20):
    """
    Ls: (nL, 3) list of (sorted) lattice points to check
    lvecs: (3, 3) lattice vectors
    basis_ls: (ncontractions,) l value for every Gaussian contraction (concatenated together)
    basis_arrays: (ngaussians, 2) contraction coefficients for all Gaussian contractions (concatenated together)
    splits: (ncontractions+1,) indexing for basis_arrays
    l_splits: (natom+1,) indexing for basis_ls
    expcutoff: (float) value of exponent to cut off Gaussian evaluation
    """
    natom = len(l_splits) - 1
    Lmax = np.zeros(len(splits)-1, dtype=np.int32)
    Lmax_a = np.zeros(natom, dtype=np.int32)
    min_exp = np.zeros(len(Ls))
    v = max_distance_in_cell(lvecs)

    r2 = np.sum((v - Ls)**2, axis=-1)
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
            for b in range(len(Ls)):
                min_exp[b] = np.amin(bas[:, 0] * r2[b] - log_c - 0.5 * np.log(r2[b]) * l)
            where = np.where(min_exp < expcutoff)[0]
            Lmax[split] = where.max() + 1 if len(where)>0 else 1
            Lmax_a[a] = max(Lmax_a[a], Lmax[split])
            split += 1
    return Lmax_a, atom_cutoff, l_cutoff


class PeriodicAtomicOrbitalEvaluator(gto.AtomicOrbitalEvaluator):
    def __init__(self, cell, kpts=None, eval_gto_precision=None):
        super().__init__(cell)
        if kpts is None: kpts = np.zeros((1, 3))
        self.kpts = kpts
        eval_gto_precision = 1e-2 if eval_gto_precision is None else eval_gto_precision
        self.rcut = _estimate_rcut(cell, eval_gto_precision)#.max()
        Ls = cell.get_lattice_Ls(rcut=self.rcut.max(), dimension=3)
        self.Ls = Ls[np.argsort(np.linalg.norm(Ls, axis=1))]
        expcutoff = -3.5*np.log(eval_gto_precision) # this number is a guess
        #print("expcutoff", expcutoff, np.exp(-expcutoff))
        self.num_Ls, self.atom_cutoff, self.l_cutoff = max_Ls(
            self.Ls, 
            cell.lattice_vectors(), 
            self.basis_ls, 
            self.basis_arrays, 
            self.splits, 
            self.l_splits, 
            expcutoff=expcutoff,
        )
        #print("num_Ls", self.num_Ls)
        #print("atom_cutoff", self.atom_cutoff)
        #print("l_cutoff", self.l_cutoff)
        self.Lmax = self.num_Ls.max()

        phases = np.exp(1j * self.Ls @ kpts.T)
        self.phases = np.real_if_close(phases)
        self.dtype =  self.phases.dtype
        self.get_wrapphase = pyqorb.get_wrapphase_complex if self.dtype == complex else pyqorb.get_wrapphase_real

        self.dist = pyqmc.configurations.distance.RawDistance() # don't use minimal image
        self._gto_func = dict(
            GTOval_sph=_pbc_eval_gto,
            GTOval_sph_deriv1=_pbc_eval_gto_grad,
            GTOval_sph_deriv2=_pbc_eval_gto_lap,
        )


    def eval_gto(self, eval_str, configs):
        eval_str = eval_str.replace("PBC", "")
        rvec = self.dist.pairwise(self.atom_coords[np.newaxis], configs[np.newaxis])[0]
        return self._gto_func[eval_str](
            rvec,
            self.basis_ls, 
            self.basis_arrays,
            self.max_l,
            self.splits,
            self.l_splits,
            self.Ls[:self.Lmax],
            self.num_Ls,
            self.l_cutoff,
            self.atom_cutoff,
            self.phases,
        )
        

# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#


# We modified _estimate_rcut slightly to take a new precision argument instead of using cell.precision
def _estimate_rcut(cell, eval_gto_precision):
    """
    Returns the cutoff raidus, above which each shell decays to a value less than the
    required precsion
    """
    vol = cell.vol
    weight_penalty = vol  # ~ V[r] * (vol/ngrids) * ngrids
    init_rcut = pyscf.pbc.gto.cell.estimate_rcut(cell, precision=eval_gto_precision)
    precision = eval_gto_precision / max(weight_penalty, 1)
    rcut = []
    for ib in range(cell.nbas):
        l = cell.bas_angular(ib)
        es = cell.bas_exp(ib)
        cs = abs(cell._libcint_ctr_coeff(ib)).max(axis=1)
        norm_ang = ((2 * l + 1) / (4 * np.pi)) ** 0.5
        fac = 2 * np.pi / vol * cs * norm_ang / es / precision
        r = init_rcut
        for _ in range(2):
            r = (np.log(fac * r ** (l + 1) + 1.0) / es) ** 0.5
        rcut.append(r.max())
    return np.array(rcut)
