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
import pyqmc.gto
import pyqmc.pbcgto
import pyqmc.distance
import pyqmc.coord

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

        self._mol = mol
        self.evaluator = pyqmc.gto.AtomicOrbitalEvaluator(mol)
        self.ao_dtype = self.evaluator.dtype
        self.mo_dtype = complex if iscomplex else self.ao_dtype

    def nmo(self):
        return [
            self.parameters["mo_coeff_alpha"].shape[-1],
            self.parameters["mo_coeff_beta"].shape[-1],
        ]

    def aos(self, eval_str, configs, mask=None):
        """"""
        mycoords = configs.configs if mask is None else configs.configs[mask]
        mycoords = mycoords.reshape((-1, mycoords.shape[-1]))
        aos = gpu.cp.asarray([self.evaluator.eval_gto(eval_str, mycoords)])
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

        self._kpts = np.zeros((1, 3)) if kpts is None else np.asarray(kpts).reshape((-1, 3))
        self.isgamma = np.abs(self._kpts).sum() < 1e-9

        eval_gto_precision = 1e-2 if eval_gto_precision is None else eval_gto_precision
        self.evaluator = pyqmc.pbcgto.PeriodicAtomicOrbitalEvaluator(cell.original_cell, kpts=self._kpts, eval_gto_precision=eval_gto_precision)

        if mo_coeff is not None:
            nelec_per_kpt = [np.asarray([m.shape[1] for m in mo]) for mo in mo_coeff]
            self.param_split = [np.cumsum(nelec_per_kpt[spin]) for spin in [0, 1]]
            self.parm_names = ["mo_coeff_alpha", "mo_coeff_beta"]
            self.parameters = {
                "mo_coeff_alpha": gpu.cp.asarray(np.concatenate(mo_coeff[0], axis=1)),
                "mo_coeff_beta": gpu.cp.asarray(np.concatenate(mo_coeff[1], axis=1)),
            }
            iscomplex = bool(
                sum(map(gpu.cp.iscomplexobj, self.parameters.values()))
            )
        else:
            iscomplex = False

        self.ao_dtype = self.evaluator.dtype
        self.mo_dtype = complex if iscomplex else self.ao_dtype
        self.get_wrapphase = get_wrapphase_complex if iscomplex else get_wrapphase_real

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
        ao = gpu.cp.asarray(self.evaluator.eval_gto(eval_str, primcoords))
        if self.isgamma == False:
            wrap = configs.wrap if mask is None else configs.wrap[mask]
            wrap = np.dot(wrap, self.S)
            wrap = wrap.reshape((-1, wrap.shape[-1])) + primwrap
            kdotR = np.linalg.multi_dot(
                (self._kpts, self._cell.lattice_vectors().T, wrap.T)
            )
            # k, coordinate
            wrap_phase = self.get_wrapphase(kdotR)
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

