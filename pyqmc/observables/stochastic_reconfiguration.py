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
import h5py


def _weighted_average(arr, weights, wsum):
    """Equivalent to np.average(arr, weights=weights, axis=0).

    np.average forms a full-size (weights * arr) temporary before reducing it,
    which dominates the cost for the (nconf, nparams) arrays here. tensordot
    contracts axis 0 directly through BLAS instead.
    """
    return np.tensordot(weights, arr, axes=(0, 0)) / wsum


def nodal_regularization(grad2, nodal_cutoff=1e-3):
    """
    Return true if a given configuration is within nodal_cutoff
    of the node
    Also return the regularization polynomial if true,
    f = a * r ** 2 + b * r ** 4 + c * r ** 6

    This uses the method from
    Shivesh Pathak and Lucas K. Wagner.
    “A Light Weight Regularization for Wave Function
    Parameter Gradients in Quantum Monte Carlo.”
    AIP Advances 10, no. 8 (August 6, 2020): 085213.
    https://doi.org/10.1063/5.0004008.
    """
    r = 1.0 / grad2
    mask = r < nodal_cutoff**2

    c = 7.0 / (nodal_cutoff**6)
    b = -15.0 / (nodal_cutoff**4)
    a = 9.0 / (nodal_cutoff**2)

    f = a * r + b * r**2 + c * r**3
    f[np.logical_not(mask)] = 1.0

    return mask, f


def local_derivatives(wf, configs, enacc, transform, nodal_cutoff=1e-3):
    """Local energy and regularized parameter derivatives of log(psi) for every
    configuration.

    Every SR solver (explicit S, minSR, CG) builds its equations from these, so
    they all regularize the same way: the derivatives are multiplied by the
    nodal regularization f, which enters the gradient once and S as f^2.

    :returns: (energy dictionary from enacc, (nconf, nparameters) regularized derivatives)
    """
    den = enacc(configs, wf)
    dp = transform.serialize_gradients(wf.pgradient())
    _, f = nodal_regularization(den["grad2"], nodal_cutoff)
    return den, dp * f[:, np.newaxis]


def sr_averages(den, dppsi, weights=None):
    """Averages that define the SR equations.

    :parameter den: energy dictionary, each entry (nconf,)
    :parameter dppsi: (nconf, nparameters) regularized derivatives
    :parameter weights: (nconf,) weights, or None for equal weights
    :returns: dictionary with the averaged energy terms and
        dppsi = <O>, dpH = <E O*>, dpidpj = <O_i* O_j>
    """
    nconf = dppsi.shape[0]
    weights = np.ones(nconf) / nconf if weights is None else weights / np.sum(weights)
    wsum = np.sum(weights)
    d = {k: _weighted_average(it, weights, wsum) for k, it in den.items()}
    if dppsi.shape[1] > 0:
        # dpH and dpidpj share this (nconf, nparams) temporary; build it once.
        w_dp = weights[:, np.newaxis] * dppsi
        d["dpH"] = den["total"] @ np.conj(w_dp)
        d["dppsi"] = _weighted_average(dppsi, weights, wsum)
        d["dpidpj"] = np.einsum("ij,ik->jk", np.conj(dppsi), w_dp, optimize=True)
    return d


def sr_direction(data, eps, inverse_strategy="regularized_inverse", g=None):
    """Solve the SR equations with the explicit (nparameters, nparameters) S matrix.

    :parameter data: output of :func:`sr_averages`, averaged over blocks
    :parameter float eps: regularization
    :parameter str inverse_strategy: 'regularized_inverse' or 'pseudo_inverse'
    :parameter g: (nparameters,) gradient term to add to the energy gradient, or None
    :returns: (v, report), where v = (S + eps)^-1 (f + g) and f is the energy gradient
    """
    pgrad = 2 * np.real(data["dpH"] - data["total"] * np.conj(data["dppsi"]))
    if g is not None:
        pgrad = pgrad + g
    Sij = np.real(
        data["dpidpj"] - np.einsum("i,j->ij", np.conj(data["dppsi"]), data["dppsi"])
    )
    if inverse_strategy == "pseudo_inverse":
        invSij = np.linalg.pinv(Sij, rcond=eps)
    elif inverse_strategy == "regularized_inverse":
        invSij = np.linalg.inv(Sij + eps * np.eye(Sij.shape[0]))
    else:
        raise ValueError(
            "Invalid inverse strategy. Valid options are pseudo_inverse and regularized_inverse."
        )
    v = np.einsum("ij,j->i", invSij, pgrad)
    report = {
        "pgrad": np.linalg.norm(pgrad),
        "SRdot": np.dot(pgrad, v) / (np.linalg.norm(v) * np.linalg.norm(pgrad)),
    }
    return v, report


class StochasticReconfiguration:
    """
    Accumulator for the averages that define the SR equations; see
    :func:`sr_averages`. :meth:`delta_p` solves them.
    """

    def __init__(
        self,
        enacc,
        transform,
        nodal_cutoff=1e-3,
        eps=1e-1,
        inverse_strategy="pseudo_inverse",
        verbose=False,
    ):
        """
        eps here is the regularization for SR.
        """
        self.enacc = enacc
        self.transform = transform
        self.nodal_cutoff = nodal_cutoff
        self.eps = eps
        self.inverse_strategy = inverse_strategy
        self.verbose = verbose

    def __call__(self, configs, wf):
        """Per-configuration values of the quantities avg() averages."""
        d, dppsi = local_derivatives(wf, configs, self.enacc, self.transform, self.nodal_cutoff)
        d["dpH"] = d["total"][:, np.newaxis] * np.conj(dppsi)
        d["dppsi"] = dppsi
        d["dpidpj"] = np.einsum("ij,ik->ijk", np.conj(dppsi), dppsi)
        return d

    def avg(self, configs, wf, weights=None):
        """(Weighted) average over configurations."""
        den, dppsi = local_derivatives(wf, configs, self.enacc, self.transform, self.nodal_cutoff)
        return sr_averages(den, dppsi, weights)

    def keys(self):
        return self.enacc.keys().union(["dpH", "dppsi", "dpidpj"])

    def shapes(self):
        nparms = np.sum([np.sum(opt) for opt in self.transform.to_opt.values()])
        d = {"dpH": (nparms,), "dppsi": (nparms,), "dpidpj": (nparms, nparms)}
        d.update(self.enacc.shapes())
        return d

    def update_state(self, hdf_file: h5py.File):
        """
        Update the state of the accumulator from a restart file.
        StochasticReconfiguration does not keep a state.

        hdf_file: h5py.File object
        """
        pass

    def delta_p(self, steps, data: dict, verbose=False):
        """Parameter changes -step * v for each step, with v from :func:`sr_direction`.

        data: averaged data from avg(), with the keys returned by keys().
        """
        v, report = sr_direction(data, self.eps, self.inverse_strategy)
        if verbose or self.verbose:
            print("Gradient norm: ", report["pgrad"])
            print("Dot product between gradient and SR step: ", report["SRdot"])
        return [-step * v for step in steps], report
