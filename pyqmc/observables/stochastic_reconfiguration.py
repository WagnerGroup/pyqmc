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
from pyqmc.observables.accumulators_multiwf import invert_list_of_dicts
import scipy.stats


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

    def __init__(self, enacc, transform, nodal_cutoff=1e-3, eps=1e-1, inverse_strategy="pseudo_inverse"):
        """
        eps here is the regularization for SR.
        """
        self.enacc = enacc
        self.transform = transform
        self.nodal_cutoff = nodal_cutoff
        self.eps = eps
        self.inverse_strategy = inverse_strategy

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

        d = {k: np.average(it, weights=weights, axis=0) for k, it in den.items()}
        if self.transform.nparams > 0:
            d["dpH"] = np.einsum("i,ij->j", energy, weights[:, np.newaxis] * dp_regularized)
            d["dppsi"] = np.average(dp_regularized, weights=weights, axis=0)
            d["dpidpj"] = np.einsum(
                "ij,ik->jk", dp, weights[:, np.newaxis] * dp_regularized, optimize=True
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
            raise ValueError("Invalid inverse strategy. Valid options are pseudo_inverse and regularized_inverse.")
        v = np.einsum("ij,j->i", invSij, pgrad)
        dp = [-step * v for step in steps]
        report = {
            "pgrad": np.linalg.norm(pgrad),
            "SRdot": np.dot(pgrad, v) / (np.linalg.norm(v) * np.linalg.norm(pgrad)),
        }

        if verbose:
            eigvals = np.linalg.eigvals(Sij)
            print("eigenvalues of Sij", eigvals)
            print("Gradient norm: ", np.linalg.norm(pgrad))
            print("Dot product between gradient and SR step: ", report["SRdot"])
            print("gradient", pgrad)
            print("v", v)
        return dp, report


class StochasticReconfigurationMultipleWF:
    """
    This class works as an accumulator, but has an extra method that computes the change in parameters
    given the averages given by avg()
    """

    def __init__(self, enacc, transforms, eps=1e-1):
        """
        eps here is the regularization for SR.
        Note that we don't need the nodal cutoff here, because we are sampling the sum of squares which
        doesn't have the nodal divergence.
        """
        self.enacc = enacc
        self.transforms = transforms
        self.eps = eps

    def avg(self, configs, wfs, weights=None):
        """
        Compute (weighted) average
        """

        energies = invert_list_of_dicts([self.enacc(configs, wf) for wf in wfs])
        dppsi = [
            transform.serialize_gradients(wf.pgradient())
            for transform, wf in zip(self.transforms, wfs)
        ]
        d = {}
        for k, en in energies.items():
            d[k] = np.einsum("jc,ijc->ij", np.asarray(en), weights / weights.shape[-1])

        nconfig = weights.shape[-1]
        for wfi, dp in enumerate(dppsi):
            if self.transforms[wfi].nparams == 0:

                continue
            d[("dp", wfi)] = (
                np.einsum("cp,jc->pj", dp, weights[wfi, :, :], optimize=True) / nconfig
            )

            d[("dpipj", wfi)] = (
                np.einsum("cp,cq,c->pq", dp, dp, weights[wfi, wfi, :], optimize=True)
                / nconfig
            )

            d[("dpH", wfi)] = (
                np.einsum("jc,cp,jc->pj", energies["total"], dp, weights[wfi, :, :])
                / nconfig
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

    def block_average(self, data, weights):
        """
        This is meant to be called to create correctly weighted average after a number of blocks have
        been performed.
        weights are block, wf, wf
        data is a dictionary, with each entry being a numpy array of shape (block, ...) (i.e., block is added to the front of what's returned from avg())
        """
        weight_avg = np.mean(weights, axis=0)

        N = np.abs(weight_avg.diagonal())
        Nij = np.sqrt(np.outer(N, N))

        avg = {}
        error = {}
        for k in ["total"]:
            it = data[k]
            avg[k] = np.mean(it, axis=0) / Nij
            error[k] = scipy.stats.sem(it, axis=0) / Nij

        
        nwf = weights.shape[1]
        for k in ["dp", "dpH"]:
            for w in range(nwf):
                if self.transforms[w].nparams == 0:
                    continue
                it = data[(k, w)]
                avg[(k, w)] = np.mean(it, axis=0) / Nij[w]
                error[(k, w)] = scipy.stats.sem(it, axis=0) / Nij[w]

        for k in ["dpipj"]:
            for w in range(nwf):
                if self.transforms[w].nparams == 0:
                    continue
                it = data[(k, w)]
                avg[(k, w)] = np.mean(it, axis=0) / Nij[w, w]
                error[(k, w)] = scipy.stats.sem(it, axis=0) / Nij[w, w]
        avg["overlap"] = weight_avg
        return avg, error

    def _collect_terms(self, avg, error):
        ret = {}
        nwf = avg["total"].shape[0]
        N = np.abs(avg["overlap"].diagonal())
        Nij = np.sqrt(np.outer(N, N))

        ret["norm"] = N
        ret["overlap"] = avg["overlap"] / Nij
        fac = np.ones((nwf, nwf)) + np.identity(nwf)
        for wfi in range(nwf):
            if self.transforms[wfi].nparams == 0:
                continue
            ret[("dp_energy", wfi)] = fac[wfi] * np.real(
                avg[("dpH", wfi)] - avg["total"][wfi] * avg[("dp", wfi)]
            )
            ret[("dp_norm", wfi)] = 2.0 * np.real(avg[("dp", wfi)][:, wfi])

            norm_part = (
                np.einsum("i,p->pi", avg["overlap"][wfi, :], ret[("dp_norm", wfi)])
                / N[wfi]
            )
            ret[("dp_overlap", wfi)] = (
                fac[wfi] * (avg[("dp", wfi)] - 0.5 * norm_part) / Nij[wfi]
            )
            ret[("dpipj", wfi)] = np.real(
                avg[("dpipj", wfi)]
                - np.einsum(
                    "i,j->ij", avg[("dp", wfi)][:, wfi], avg[("dp", wfi)][:, wfi]
                )
            )
        ret["energy"] = avg["total"]
        return ret

    def delta_p(
        self, steps: np.ndarray, data: dict, overlap_penalty: np.ndarray, verbose=False
    ):
        """
        steps: a list/numpy array of timesteps
        data: averaged data from avg() or __call__. Note that if you use VMC to compute this with
        an accumulator with a name, you'll need to remove that name from the keys.
        That is, the keys should be equal to the ones returned by keys().


        Compute the change in parameters given the data from a stochastic reconfiguration step.
        Return the change in parameters, and data that we may want to use for diagnostics.
        """
        data = self._collect_terms(data, None)
        nwf = data["energy"].shape[0]
        dp_all = []
        for wf in range(nwf):
            overlap_cost = 0.0
            for i in range(wf):
                overlap_cost += overlap_penalty[wf, i] * data["overlap"][wf, i]
            if verbose:
                print("wave function", wf)
                print("Overlap cost", overlap_cost)

            if self.transforms[wf].nparams == 0:
                dp_all.append([np.zeros((0)) for step in steps])
                continue

            Sij = np.real(data[("dpipj", wf)])
            ovlp = 0.0
            for i in range(wf):
                ovlp += (
                    2.0
                    * data[("dp_overlap", wf)][:, i]
                    * overlap_penalty[wf, i]
                    * data["overlap"][wf, i]
                )
            pgrad = data[("dp_energy", wf)][:, wf] + ovlp

            invSij = np.linalg.inv(Sij + self.eps * np.eye(Sij.shape[0]))
            v = np.einsum("ij,j->i", invSij, pgrad)
            dp = [-step * v for step in steps]
            dp_all.append(dp)
            report = {
                "pgrad": np.linalg.norm(pgrad),
                "SRdot": np.dot(pgrad, v) / (np.linalg.norm(v) * np.linalg.norm(pgrad)),
            }
            if verbose:
                print("overlap gradient norm", np.linalg.norm(ovlp))
                print("Gradient norm: ", np.linalg.norm(pgrad))
                print("Dot product between gradient and SR step: ", report["SRdot"])
        return dp_all, report
