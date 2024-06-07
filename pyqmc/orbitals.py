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
import pyqmc.gpu as gpu
import pyscf.pbc.gto.eval_gto
import pyscf.pbc.gto.cell
import pyqmc.pbc

"""
The evaluators have the concept of a 'set' of atomic orbitals, that may apply to 
different sets of molecular orbitals

For example, for the PBC evaluator, each k-point is a set, since each molecular 
orbital is only a sum over the k-point of its type.

In the future, this could apply to orbitals of a given point group symmetry, for example.
"""


def get_wrapphase_real(x):
    return (-1) ** np.round(x / np.pi)


def get_wrapphase_complex(x):
    return np.exp(1j * x)


def get_complex_phase(x):
    return x / np.abs(x)


class MoleculeOrbitalEvaluator:
    def __init__(self, mol, mo_coeff):
        self.parameters = {
            "mo_coeff_alpha": gpu.cp.asarray(mo_coeff[0]),
            "mo_coeff_beta": gpu.cp.asarray(mo_coeff[1]),
        }
        self.parm_names = ["mo_coeff_alpha", "mo_coeff_beta"]
        iscomplex = bool(sum(map(gpu.cp.iscomplexobj, self.parameters.values())))
        self.ao_dtype = True
        self.mo_dtype = complex if iscomplex else float

        self._mol = mol

    def nmo(self):
        return [
            self.parameters["mo_coeff_alpha"].shape[-1],
            self.parameters["mo_coeff_beta"].shape[-1],
        ]

    def aos(self, eval_str, configs, mask=None):
        """"""
        mycoords = configs.configs if mask is None else configs.configs[mask]
        mycoords = mycoords.reshape((-1, mycoords.shape[-1]))
        aos = gpu.cp.asarray([self._mol.eval_gto(eval_str, mycoords)])
        if len(aos.shape) == 4:  # if derivatives are included
            return aos.reshape((1, aos.shape[1], *mycoords.shape[:-1], aos.shape[-1]))
        else:
            return aos.reshape((1, *mycoords.shape[:-1], aos.shape[-1]))

    def mos(self, ao, spin):
        return ao[0].dot(self.parameters[self.parm_names[spin]])

    def pgradient(self, ao, spin):
        nelec = [self.parameters[self.parm_names[spin]].shape[1]]
        return (gpu.cp.array(nelec), ao)


class PBCOrbitalEvaluatorKpoints:
    """
    Evaluate orbitals from a PBC object.
    cell is expected to be one made with make_supercell().
    mo_coeff should be in [spin][k][ao,mo] order
    kpts should be a list of the k-points corresponding to mo_coeff

    """

    def __init__(self, cell, mo_coeff=None, kpts=None, eval_gto_precision=None):
        """
        :parameter cell: PyQMC supercell object (from get_supercell)
        :parameter mo_coeff: (2, nk, nao, nelec) array. MO coefficients for all kpts of primitive cell. If None, this object can't evaluate mos(), but can still evaluate aos().
        :parameter kpts: list of kpts to evaluate AOs
        :eval_gto_precision: desired value of orbital at rcut, used for determining rcut for periodic system. Default value = 0.01
        """
        self._cell = cell.original_cell
        self.S = cell.S
        self.Lprim = self._cell.lattice_vectors()

        self._kpts = [0, 0, 0] if kpts is None else kpts
        # If gamma-point only, AOs are real-valued
        isgamma = np.abs(self._kpts).sum() < 1e-9
        if mo_coeff is not None:
            nelec_per_kpt = [np.asarray([m.shape[1] for m in mo]) for mo in mo_coeff]
            self.param_split = [np.cumsum(nelec_per_kpt[spin]) for spin in [0, 1]]
            self.parm_names = ["mo_coeff_alpha", "mo_coeff_beta"]
            self.parameters = {
                "mo_coeff_alpha": gpu.cp.asarray(np.concatenate(mo_coeff[0], axis=1)),
                "mo_coeff_beta": gpu.cp.asarray(np.concatenate(mo_coeff[1], axis=1)),
            }
            iscomplex = (not isgamma) or bool(
                sum(map(gpu.cp.iscomplexobj, self.parameters.values()))
            )
        else:
            iscomplex = not isgamma

        self.ao_dtype = float if isgamma else complex
        self.mo_dtype = complex if iscomplex else float

        eval_gto_precision = 1e-2 if eval_gto_precision is None else eval_gto_precision
        self.rcut = _estimate_rcut(self._cell, eval_gto_precision)
        Ls = self._cell.get_lattice_Ls(rcut=self.rcut.max(), dimension=3)
        self.Ls = Ls[np.argsort(np.linalg.norm(Ls, axis=1))]

    def nmo(self):
        return [
            self.parameters["mo_coeff_alpha"].shape[-1],
            self.parameters["mo_coeff_beta"].shape[-1],
        ]

    def aos(self, eval_str, configs, mask=None):
        """
        Returns an ndarray in order [k,..., orbital] of the ao's if value is requested

        if a derivative is requested, will instead return [k,d,...,orbital].

        The ... is the total length of mycoords. You'll need to reshape if you want the original shape
        """
        mycoords = configs.configs if mask is None else configs.configs[mask]
        mycoords = mycoords.reshape((-1, mycoords.shape[-1]))
        primcoords, primwrap = pyqmc.pbc.enforce_pbc(self.Lprim, mycoords)
        ao = gpu.cp.asarray(
            pyscf.pbc.gto.eval_gto.eval_gto(
                self._cell,
                "PBC" + eval_str,
                primcoords,
                kpts=self._kpts,
                rcut=self.rcut,
                Ls=self.Ls,
            )
        )
        if self.ao_dtype == complex:
            wrap = configs.wrap if mask is None else configs.wrap[mask]
            wrap = np.dot(wrap, self.S)
            wrap = wrap.reshape((-1, wrap.shape[-1])) + primwrap
            kdotR = np.linalg.multi_dot(
                (self._kpts, self._cell.lattice_vectors().T, wrap.T)
            )
            # k, coordinate
            wrap_phase = get_wrapphase_complex(kdotR)
            ao = gpu.cp.einsum("k...,k...a->k...a", wrap_phase, ao)
        if len(ao.shape) == 4:  # if derivatives are included
            return ao.reshape(
                (ao.shape[0], ao.shape[1], *mycoords.shape[:-1], ao.shape[-1])
            )
        else:
            return ao.reshape((ao.shape[0], *mycoords.shape[:-1], ao.shape[-1]))

    def mos(self, ao, spin):
        """ao should be [k,[d],...,ao].
        Returns a concatenated list of all molecular orbitals in form [..., mo]

        In the derivative case, returns [d,..., mo]
        """
        p = np.split(
            self.parameters[self.parm_names[spin]],
            self.param_split[spin],
            axis=-1,
        )
        ps = [0] + list(self.param_split[spin])
        nelec = self.parameters[self.parm_names[spin]].shape[1]
        out = gpu.cp.zeros([nelec, *ao[0].shape[:-1]], dtype=self.mo_dtype)
        for i, ak, mok in zip(range(len(ao)), ao, p[:-1]):
            gpu.cp.einsum(
                "...a,an->n...", ak, mok, out=out[ps[i] : ps[i + 1]], optimize="greedy"
            )
        return out.transpose([*np.arange(1, len(out.shape)), 0])

    def pgradient(self, ao, spin):
        """
        returns:
        N sets of atomic orbitals
        split: which molecular orbitals correspond to which set

        You can construct the determinant by doing, for example:
        split, aos = pgradient(self.aos)
        mos = np.split(range(nmo),split)
        for ao, mo in zip(aos,mos):
            for i in mo:
                pgrad[:,:,i] = self._testcol(i,spin,ao)

        """
        return self.param_split[spin], ao


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
