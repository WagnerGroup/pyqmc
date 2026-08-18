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
"""
Ensemble (excited state) optimization with minSR.

:class:`MinSRWfbyWf` is a drop-in replacement for
:class:`pyqmc.method.ensemble_optimization.StochasticReconfigurationWfbyWf`: it
plugs into the same :func:`pyqmc.method.ensemble_optimization.optimize_ensemble`
driver and gets the same threading, warmups, and checkpointing::

    from pyqmc.method.ensemble_optimization import optimize_ensemble
    from pyqmc.method.ensemble_minsr import MinSRWfbyWf

    updater = [[MinSRWfbyWf(enacc, LinearTransform(wf.parameters, to_opt))]
               for wf in wfs]
    optimize_ensemble(wfs, configs, updater, hdf_file, tau=0.1,
                      vmc_kwargs={"nblocks": 1, "nsteps_per_block": 10})

The algorithm is unchanged -- each state sampled separately, an overlap penalty
against the lower states, a stochastic reconfiguration step. Only the solve is
different, and the (nparameters, nparameters) S matrix is never built.

The ensemble gradient has two pieces,

.. math:: f = f_{\\rm energy} + f_{\\rm overlap}

The energy piece comes from per-sample derivatives, so
:math:`f_{\\rm energy} = A^T b` and the overlap matrix is :math:`S = A^T A`,
which is exactly the minSR structure. The overlap penalty piece comes from the
separate overlap sampling and is just a vector, with no reason to lie in the row
space of A. :func:`pyqmc.method.minsr.sr_solve` applies the same regularized
inverse to both using only the (nsamples, nsamples) kernel.
"""

import h5py
import numpy as np

import pyqmc.method.sample_many
from pyqmc.method.ensemble_optimization import _block_sem
from pyqmc.method.minsr import (
    local_energy_error,
    real_design_matrix,
    sample_minsr_data,
    sr_solve,
)


def overlap_derivatives_worker(wfs, configs, transform):
    """Weighted parameter derivatives of the last wave function, summed over the
    configurations of one snapshot.

    :parameter wfs: the wave functions sharing the overlap distribution
    :parameter configs: (nconfig, nelec, 3) configurations distributed as sum_i |psi_i|^2
    :parameter transform: a LinearTransform for the last wave function's parameters
    :returns: ((nparameters, nwf, nwf) sum over configurations, number of configurations)
    """
    for wf in wfs:
        wf.recompute(configs)
    dp = transform.serialize_gradients(wfs[-1].pgradient())
    weights = pyqmc.method.sample_many.compute_weights(wfs)
    return np.einsum("cp,jkc->pjk", dp, weights, optimize=True), weights.shape[-1]


def sample_overlap_derivatives(wfs, configs, transform, client=None, npartitions=None):
    """Average of :func:`overlap_derivatives_worker` over all configurations,
    distributed over a client if one is given.

    The configurations are not propagated here, so they are not returned.

    :returns: (nparameters, nwf, nwf) weighted derivatives averaged over configurations
    """
    if client is None:
        total, nconfig = overlap_derivatives_worker(wfs, configs, transform)
        return total / nconfig
    runs = [
        client.submit(overlap_derivatives_worker, wfs, conf, transform)
        for conf in configs.split(npartitions)
    ]
    results = [r.result() for r in runs]
    return sum(r[0] for r in results) / sum(r[1] for r in results)


class MinSRWfbyWf:
    """Updater for one state of an ensemble, the minSR analogue of
    :class:`pyqmc.method.ensemble_optimization.StochasticReconfigurationWfbyWf`.

    It drives both samplings for its state -- `sample_energy` for the energy and
    `sample_overlap` for the overlap penalty -- and turns the result into a
    parameter change in `delta_p`.

    Unlike the SR version it is not an accumulator at all. Both samplings run
    with no accumulator and evaluate the parameter derivatives per configuration
    on a snapshot per block, rather than averaging them into dpH/dppsi/dpidpj or
    into wtdp at every Metropolis step.

    Note that the number of samples per state per iteration is
    nconfig * vmc_kwargs['nblocks'], and the kernel solved is square in that
    number, so the driver's default of 10 blocks is usually not what you want
    here: pass vmc_kwargs={"nblocks": 1, ...} unless you have the memory for it.

    :parameter enacc: an EnergyAccumulator-like object
    :parameter transform: a LinearTransform for this state's parameters
    :parameter float eps: regularization of the SR equations
    :parameter float nodal_cutoff: regularization distance for the nodal divergence of the derivatives
    """

    def __init__(self, enacc, transform, eps=1e-2, nodal_cutoff=1e-3):
        self.enacc = enacc
        self.transform = transform
        self.eps = eps
        self.nodal_cutoff = nodal_cutoff

    def sample_overlap(
        self,
        wfs,
        configs,
        client=None,
        npartitions=None,
        nblocks=10,
        nsteps_per_block=10,
        tstep=0.5,
    ):
        """Sample the overlap distribution and accumulate what delta_p needs from
        it, here the weighted derivatives that give the overlap gradient.

        The derivatives are evaluated on one decorrelated snapshot per block
        rather than at every Metropolis step. Both estimates of

        .. math:: \\partial_p \\langle \\psi_k | \\psi_i \\rangle
                  = \\langle O_p^* w_{ki} \\rangle_\\rho

        are unbiased averages of the same per-configuration quantity, so the
        update is unchanged; the snapshot version just calls `pgradient` nblocks
        times instead of nblocks * nsteps_per_block times, which is the whole
        point of minSR on the energy side too. The overlap matrix itself needs
        only `wf.value()` and is still accumulated at every step.

        :returns: (weighted, unweighted, configs), matching
            :func:`pyqmc.method.sample_many.sample_overlap`
        """
        wtdp = []
        unweighted = []
        for _ in range(nblocks):
            _, block_unweighted, configs = pyqmc.method.sample_many.sample_overlap(
                wfs,
                configs,
                None,
                nblocks=1,
                nsteps_per_block=nsteps_per_block,
                tstep=tstep,
                client=client,
                npartitions=npartitions,
            )
            unweighted.append(block_unweighted)
            wtdp.append(
                sample_overlap_derivatives(
                    wfs, configs, self.transform, client, npartitions
                )
            )
        weighted = {"wtdp": np.asarray(wtdp)}
        unweighted = {
            k: np.concatenate([u[k] for u in unweighted], axis=0) for k in unweighted[0]
        }
        return weighted, unweighted, configs

    def sample_energy(
        self, wf, configs, client=None, npartitions=None, verbose=True, **kwargs
    ):
        """Sample this state on its own, keeping the derivatives per
        configuration instead of averaging them into an S matrix.

        `verbose` is accepted for the interface and ignored; the sampling here
        has nothing per-block to report.

        :returns: (data, configs) with data in the form block_average expects
        """
        return sample_minsr_data(
            wf,
            configs,
            self.transform,
            self.enacc,
            self.nodal_cutoff,
            client=client,
            npartitions=npartitions,
            **kwargs,
        )

    def update_state(self, hdf_file: h5py.File):
        """This accumulator keeps no state."""
        pass

    def block_average(self, data_sample1, data, weights):
        """Average the sampled data, with the same signature and normalization as
        StochasticReconfigurationWfbyWf.block_average.

        `data_sample1` is the per-sample output of sample_energy rather than
        block-averaged SR data, so the derivative rows are passed through
        unaveraged for delta_p to build the design matrix from.
        """
        weight_avg = np.mean(weights, axis=0)
        N = np.abs(weight_avg.diagonal())
        Nij = np.sqrt(np.outer(N, N))
        wfi = Nij.shape[0] - 1

        avg = {}
        error = {}
        for k in ["wtdp"]:
            it = data[k]
            avg[k] = np.mean(it, axis=0) / Nij[wfi]
            error[k] = _block_sem(it) / Nij[wfi]
        avg["overlap"] = weight_avg

        eloc = data_sample1["total"]
        avg["dppsi_samples"] = data_sample1["dppsi"]
        avg["eloc_samples"] = eloc
        avg["total"] = np.mean(eloc).real
        # the local energies are kept per configuration, so the error bar comes
        # from their spread rather than from the scatter of the block means; one
        # block is enough, which is what these methods default to
        error["total"] = local_energy_error(eloc)
        return avg, error

    def _collect_terms(self, avg):
        """Normalized overlaps and their derivatives. This is the overlap part of
        StochasticReconfigurationWfbyWf._collect_terms; the energy part is
        replaced by the design matrix built in delta_p.
        """
        ret = {}
        nwf = avg["overlap"].shape[0]
        N = np.abs(avg["overlap"].diagonal())
        Nij = np.sqrt(np.outer(N, N))

        fac = np.ones((nwf, nwf)) + np.identity(nwf)
        wfi = nwf - 1
        ret["norm"] = N
        ret["overlap"] = avg["overlap"] / Nij
        ret["dp_norm"] = 2.0 * np.real(avg["wtdp"][:, wfi, wfi])
        norm_part = np.einsum("i,p->pi", avg["overlap"][wfi, :], ret["dp_norm"]) / N
        ret["dp_overlap"] = (
            fac[wfi] * (avg["wtdp"][:, wfi, :] - 0.5 * norm_part) / Nij[wfi]
        )
        ret["energy"] = avg["total"]
        return ret

    def delta_p(self, steps, data, overlap_penalty, verbose=False):
        """Compute the change in parameters for this state.

        Solves :math:`(S + \\epsilon I) v = f_{\\rm energy} + f_{\\rm overlap}`
        in sample space. The energy gradient convention matches
        StochasticReconfigurationWfbyWf, so `steps` and `overlap_penalty` mean
        the same thing as they do there.

        :parameter steps: list of timesteps
        :parameter dict data: output of block_average
        :parameter overlap_penalty: (nwf, nwf) penalty matrix
        :returns: (list of parameter changes, report dictionary)
        """
        terms = self._collect_terms(data)
        nwf = terms["overlap"].shape[0]
        wfi = nwf - 1

        overlap_cost = 0.0
        ovlp = 0.0
        for i in range(wfi):
            overlap_cost += overlap_penalty[wfi, i] * terms["overlap"][wfi, i]
            ovlp += (
                2.0
                * terms["dp_overlap"][:, i]
                * overlap_penalty[wfi, i]
                * terms["overlap"][wfi, i]
            )

        A, b = real_design_matrix(data["dppsi_samples"], data["eloc_samples"])
        g = None if wfi == 0 else np.real(ovlp)
        v = sr_solve(A, b, g=g, eps=self.eps)
        dp = [-step * v for step in steps]

        pgrad = A.T @ b if g is None else A.T @ b + g
        report = {
            "pgrad": np.linalg.norm(pgrad),
            "SRdot": np.dot(pgrad, v) / (np.linalg.norm(v) * np.linalg.norm(pgrad)),
            "overlap_cost": overlap_cost,
        }
        if verbose:
            print("Overlap cost", overlap_cost)
            print("overlap gradient norm", np.linalg.norm(ovlp))
            print("Gradient norm: ", report["pgrad"])
            print("Dot product between gradient and SR step: ", report["SRdot"])
        return dp, report
