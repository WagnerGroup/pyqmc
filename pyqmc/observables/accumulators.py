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
import pyqmc.observables.energy as energy
import pyqmc.observables.ewald as ewald
import pyqmc.observables.eval_ecp as eval_ecp
from pyqmc.observables.stochastic_reconfiguration import StochasticReconfiguration
import copy
import time
import pyqmc.observables.jax_ecp as ecp_accumulator

def gradient_generator(mol, wf, to_opt=None, nodal_cutoff=1e-3, eps=1e-1, inverse_strategy="regularized_inverse", **ewald_kwargs):
    return StochasticReconfiguration(
        EnergyAccumulator(mol, **ewald_kwargs),
        LinearTransform(wf.parameters, to_opt),
        nodal_cutoff=nodal_cutoff,
        eps=eps,
        inverse_strategy=inverse_strategy,
    )


class EnergyAccumulator:
    """Returns local energy of each configuration in a dictionary."""

    def __init__(self, mol, threshold=10, naip=None, use_old_ecp=True, **kwargs):
        self.mol = mol
        self.threshold = threshold
        self.naip = naip
        if hasattr(mol, "a"):
            self.coulomb = ewald.Ewald(mol, **kwargs)
        else:
            self.coulomb = energy.OpenCoulomb(mol, **kwargs)
        self.use_old_ecp = use_old_ecp
        if not use_old_ecp:
            self.ecp = ecp_accumulator.ECPAccumulator(mol, naip=naip)


    def __call__(self, configs, wf):
        ee, ei, ii = self.coulomb.energy(configs)
        if self.use_old_ecp:
            ecp_val = eval_ecp.ecp(self.mol, configs, wf, self.threshold, self.naip)
        else:
            ecp_val = self.ecp(configs, wf)
        ke, grad2 = energy.kinetic(configs, wf)

        return {
            "ke": np.asarray(ke),
            "ee": ee,
            "ei": ei,
            "ecp": ecp_val,
            "grad2": grad2,
            "total": np.asarray(ke + ee + ei + ecp_val + ii),
        }

    def avg(self, configs, wf):
        return {k: np.mean(it, axis=0) for k, it in self(configs, wf).items()}

    def nonlocal_tmoves(self, configs, wf, e, tau):
        if self.use_old_ecp:
            return eval_ecp.compute_tmoves(self.mol, configs, wf, e, self.threshold, tau)
        else:
            return self.ecp.nonlocal_tmoves(configs, wf, e, tau)

    def has_nonlocal_moves(self):
        return self.mol._ecp != {}

    def keys(self):
        return set(["ke", "ee", "ei", "ecp", "total", "grad2"])

    def shapes(self):
        return {"ke": (), "ee": (), "ei": (), "ecp": (), "total": (), "grad2": ()}


class LinearTransform:
    """
    Linearize a dictionary of wf parameters.

    :parameter dict parameters: the wave function parameters
    :parameter dict to_opt: is a dictionary with the keys to optimize, and its values are boolean arrays indicating which specific elements to optimize

    Note:

    * to_opt[k] can't be boolean scalar; it has to be an array with the same dimension as parameters[k].
    * to_opt doesn't have to have all the keys of parameters, but all keys of to_opt must be keys of parameters.
    * If you change wf.parameters, then all serializations will be out of date.
    For example, if you serialize, then change wf.paramaters, then deserialize, the deserialization will be incorrect.
    """

    def __init__(self, parameters, to_opt=None):
        parameters = {k: gpu.asnumpy(v) for k, v in parameters.items()}
        if to_opt is None:
            to_opt = {k: np.ones(p.shape, dtype=bool) for k, p in parameters.items()}
        self.to_opt = {k: o for k, o in to_opt.items() if np.any(o)}

        self.shapes = {k: parameters[k].shape for k in self.to_opt}
        self.slices = {k: np.prod(s) for k, s in self.shapes.items()}
        self.dtypes = {k: parameters[k].dtype for k in self.to_opt}
        self.complex = {k: d == complex for k, d in self.dtypes.items()}
        self.nimag = {k: to_opt[k].sum() if c else 0 for k, c in self.complex.items()}
        if any(self.nimag.values()):
            self.complex_inds = np.concatenate(
             [np.ones(to_opt[k].sum(), dtype=bool) * c for k, c in self.complex.items()]
          )
        else:
            self.complex_inds = np.asarray([], dtype=bool)
        self.nparams = np.sum([v.sum() for v in self.to_opt.values()])

    def serialize_parameters(self, parameters):
        """Convert the dictionary to a linear list of gradients"""
        if len(self.to_opt) == 0:
            return np.zeros((0))
        params = np.concatenate(
            [gpu.asnumpy(parameters[k])[opt] for k, opt in self.to_opt.items()]
        )
        return np.concatenate((params.real, params[self.complex_inds].imag))

    def serialize_gradients(self, pgrad):
        """Convert the dictionary to a linear list of gradients, mask allows for certain fixed parameters"""
        grads = []
        for k, opt in self.to_opt.items():
            mask = ~np.repeat(opt[np.newaxis, :], pgrad[k].shape[0], axis=0)
            #print(k, "mask", mask.shape, pgrad[k].shape, self.to_opt[k].shape)
            mask_grads = np.ma.array(pgrad[k], mask=mask).reshape(pgrad[k].shape[0], -1)
            grads.append(np.ma.compress_cols(mask_grads))
        if len(grads) == 0:
            return np.zeros((0))
        grads = np.concatenate(grads, axis=1)
        return np.concatenate((grads, grads[:, self.complex_inds] * 1j), axis=1)

    def deserialize(self, wf, parameters):
        """Convert serialized parameters to dictionary.
        Inputs:
        wf object (this is needed to fill in the frozen parameters)
        serialized parameters

        output: dictionary with the unserialized parameters.
        """
        n = 0
        m = self.nparams
        d = {}
        frozen_parms = {
            k: gpu.asnumpy(wf.parameters[k])[~opt] for k, opt in self.to_opt.items()
        }

        for k, opt in self.to_opt.items():
            opt_ = opt.flatten()
            n_p = np.sum(opt_)

            flat_parms = np.zeros(self.slices[k], dtype=self.dtypes[k])
            flat_parms[~opt_] = frozen_parms[k]
            flat_parms[opt_] = np.real(parameters[n : n + n_p])
            if self.complex[k]:
                m_p = self.nimag[k]
                flat_parms[opt_] += parameters[m : m + m_p] * 1j
                m += m_p
            d[k] = gpu.cp.asarray(flat_parms.reshape(self.shapes[k]))
            n += n_p
        return d


PGradTransform = StochasticReconfiguration


class SqAccumulator:
    r"""
    Accumulates structure factor

    .. math:: S(\vec{q}) = \langle \rho_{\vec{q}} \rho_{-\vec{q}} \rangle
                         = \langle \left| \sum_{j=1}^{N_e} e^{i\vec{q}\cdot\vec{r}_j} \right|^2 \rangle
    .. math:: S_{\rm spin}(\vec{q}) = \langle \left| \sum_{j=1}^{N_e} s_j e^{i\vec{q}\cdot\vec{r}_j} \right|^2 \rangle

    """

    def __init__(self, cell, nq=4, qlist=None):
        """
        Inputs:
            cell: pyscf Cell object
            nq: int. If qlist is nonzero, use a uniform grid of shape (nq, nq, nq)
            qlist: (n, 3) array-like. If qlist is provided, nq is ignored
        """
        if qlist is not None:
            self.qlist = qlist
        else:
            recvec = np.linalg.inv(cell.lattice_vectors()).T
            self.qlist = ewald.generate_positive_gpoints(nq, recvec)
        nup = cell.nelec[0]
        self.nelec = sum(cell.nelec)
        self.spins = np.ones((2, self.nelec))
        self.spins[1, nup:] = -1

    def __call__(self, configs, wf):
        exp_iqr = np.exp(1j * np.inner(configs.configs, self.qlist))
        sum_exp_iqr = np.einsum("ijk,sj->sik", exp_iqr, self.spins)
        Sq = (sum_exp_iqr.real**2 + sum_exp_iqr.imag**2) / self.nelec
        return {"Sq": Sq[0], "spinSq": Sq[1]}

    def avg(self, configs, wf):
        exp_iqr = np.exp(1j * np.inner(configs.configs, self.qlist))
        sum_exp_iqr = np.einsum("ijk,sj->sik", exp_iqr, self.spins)
        Sq = (sum_exp_iqr.real**2 + sum_exp_iqr.imag**2).mean(axis=1) / self.nelec
        return {"Sq": Sq[0], "spinSq": Sq[1]}

    def keys(self):
        return set(["Sq", "spinSq"])

    def shapes(self):
        return {"Sq": (len(self.qlist),), "spinSq": (len(self.qlist),)}


class SymmetryAccumulator:
    """
    Evaluates S * Psi(R) / Psi(R) for each many-body symmetry operator S given in a dictionary
    Makes use of the equivariance property S * Psi(R) = Psi(S * R) by transforming all electron coordinates R and recomputing the wf
    When defining a SymmetryAccumulator object, pass in a dictionary of symmetry operator names and their respective 3x3 unitary matrices
    For example, to evaluate a rotation of angle theta about the z-axis and mirror reflection about the yz plane, use the code

    rotation_z = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    reflection_yz = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    symmetry_operators = {"rotation_z": rotation_z, "reflection_yz": reflection_yz}
    acc = {"symmetry": SymmetryAccumulator(symmetry_operators=symmetry_operators)}
    """

    def __init__(self, symmetry_operators):
        """
        Inputs:
            symmetry_operators: dictionary of symmetry operator names and their respective unitary transformation matrices of shape (3,3)
        """
        self.symmetry_operators = symmetry_operators

    def __call__(self, configs, wf):
        symmetry_observables = {}
        original_wf_value = wf.value()
        configs_copy = copy.deepcopy(configs)
        for S_name, S_matrix in self.symmetry_operators.items():
            configs_copy.configs = np.einsum("ijk,kl->ijl", configs.configs, S_matrix)
            transformed_wf_value = wf.recompute(configs_copy)
            symmetry_observables[S_name] = (
                transformed_wf_value[0] / original_wf_value[0]
            ) * np.exp(transformed_wf_value[1] - original_wf_value[1])
        wf.recompute(configs)
        return symmetry_observables

    def avg(self, configs, wf):
        return {k: np.mean(it, axis=0) for k, it in self(configs, wf).items()}

    def keys(self):
        return self.shapes().keys()

    def shapes(self):
        return {S: () for S in self.symmetry_operators.keys()}
