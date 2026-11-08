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


class StochasticReconfiguration:
    """
    This class works as an accumulator, but has an extra method
    that computes the change in parameters
    given the averages given by avg() and __call__.
    """

    def __init__(
        self,
        enacc,
        transform,
        nodal_cutoff=1e-3,
        eps=1e-1,
        inverse_strategy="pseudo_inverse",
        verbose = False
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
        pgrad = wf.pgradient()
        d = self.enacc(configs, wf)
        energy = d["total"]
        dp = self.transform.serialize_gradients(pgrad)

        node_cut, f = nodal_regularization(d["grad2"], self.nodal_cutoff)
        dp_regularized = dp * f[:, np.newaxis]

        d["dpH"] = np.einsum("i,ij->ij", energy, dp_regularized)
        d["dppsi"] = dp_regularized
        d["dpidpj"] = np.einsum("ij,ik->ijk", dp, dp_regularized)

        return d

    def avg(self, configs, wf, weights=None):
        """
        Compute (weightsd) average
        """

        nconf = configs.configs.shape[0]
        if weights is None:
            weights = np.ones(nconf)
        weights = weights / np.sum(weights)

        pgrad = wf.pgradient()
        den = self.enacc(configs, wf)
        energy = den["total"]
        dp = self.transform.serialize_gradients(pgrad)

        node_cut, f = nodal_regularization(den["grad2"])

        dp_regularized = dp * f[:, np.newaxis]

        wsum = np.sum(weights)
        d = {k: _weighted_average(it, weights, wsum) for k, it in den.items()}
        if self.transform.nparams > 0:
            # dpH and dpidpj share this (nconf, nparams) temporary; build it once.
            w_dp_regularized = weights[:, np.newaxis] * dp_regularized
            d["dpH"] = energy @ w_dp_regularized
            d["dppsi"] = _weighted_average(dp_regularized, weights, wsum)
            d["dpidpj"] = np.einsum(
                "ij,ik->jk", dp, w_dp_regularized, optimize=True
            )

        return d

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
        """
        steps: a list/numpy array of timesteps
        data: averaged data from avg() or __call__. Note that if you use VMC to compute this with
        an accumulator with a name, you'll need to remove that name from the keys.
        That is, the keys should be equal to the ones returned by keys().

        Compute the change in parameters given the data from a stochastic reconfiguration step.
        Return the change in parameters, and data that we may want to use for diagnostics.
        """

        pgrad = 2 * np.real(data["dpH"] - data["total"] * data["dppsi"])
        Sij = np.real(
            data["dpidpj"] - np.einsum("i,j->ij", data["dppsi"], data["dppsi"])
        )

        if self.inverse_strategy == "pseudo_inverse":
            invSij = np.linalg.pinv(Sij, rcond=self.eps)
        elif self.inverse_strategy == "regularized_inverse":
            invSij = np.linalg.inv(Sij + self.eps * np.eye(Sij.shape[0]))
        else:
            raise ValueError(
                "Invalid inverse strategy. Valid options are pseudo_inverse and regularized_inverse."
            )
        v = np.einsum("ij,j->i", invSij, pgrad)
        dp = [-step * v for step in steps]
        report = {
            "pgrad": np.linalg.norm(pgrad),
            "SRdot": np.dot(pgrad, v) / (np.linalg.norm(v) * np.linalg.norm(pgrad)),
        }

        if verbose or self.verbose:
            eigvals = np.linalg.eigvals(Sij)
            print("eigenvalues of Sij", eigvals)
            print("Gradient norm: ", np.linalg.norm(pgrad))
            print("Dot product between gradient and SR step: ", report["SRdot"])
            print("gradient", pgrad)
            print("v", v)
        return dp, report
