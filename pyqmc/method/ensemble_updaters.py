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

r"""Per-state updaters for ensemble (excited state) optimization.

Each state i is optimized on

.. math:: E_i + \sum_{j<i} \lambda_{ij} |S_{ij}|^2,

where :math:`S_{ij}` is the normalized overlap with a lower state and
:math:`\lambda` is the overlap penalty. The SR step solves

.. math:: (S + \epsilon I)\, v = f_{\rm energy} + g_{\rm overlap}

where :math:`f_{\rm energy}` is the energy gradient and :math:`g_{\rm overlap}`
the gradient of the penalty. :class:`EnsembleUpdater` samples the overlap and
computes :math:`g_{\rm overlap}`; the energy sampling and the solve come from a
single-state updater (:class:`~pyqmc.method.optimize.SRUpdate`,
:class:`~pyqmc.method.optimize.MinSRUpdate`, or
:class:`~pyqmc.method.optimize.CGSRUpdate`), which takes
:math:`g_{\rm overlap}` as its extra gradient term. That term comes from separate
sampling, so it need not lie in the row space of the sampled derivatives; each
solver handles that exactly (see :func:`pyqmc.method.minsr.sr_solve`).
"""

import numpy as np
import scipy.stats

import pyqmc.method.sample_many
import pyqmc.rng
from pyqmc.method.optimize import CGSRUpdate, MinSRUpdate, SRUpdate


def _block_sem(blocks):
    """Standard error over the leading (block) axis.

    scipy.stats.sem warns and returns nan for a single block, which is a common
    setting for the per-sample methods; return nan quietly instead.
    """
    if blocks.shape[0] < 2:
        return np.full(blocks.shape[1:], np.nan)
    return scipy.stats.sem(blocks, axis=0)


class _OverlapDerivativeAccumulator:
    """Weighted parameter derivatives of the last wave function, accumulated at
    every step of the overlap sampling."""

    def __init__(self, transform):
        self.transform = transform

    def avg(self, configs, wfs, weights):
        dp = self.transform.serialize_gradients(wfs[-1].pgradient())
        nconfig = weights.shape[-1]
        return {"wtdp": np.einsum("cp,jkc->pjk", dp, weights, optimize=True) / nconfig}


def overlap_derivatives_worker(wfs, configs, transform):
    """Weighted parameter derivatives of the last wave function, summed over the
    configurations of one snapshot.

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


def _sample_overlap_snapshots(
    wfs, configs, transform, client=None, npartitions=None, nblocks=10, nsteps_per_block=10, tstep=0.5, rng=None
):
    """Sample the overlap distribution, evaluating the weighted derivatives on
    one snapshot per block rather than at every step.

    Both this and the every-step accumulation are unbiased averages of the same
    per-configuration quantity; this calls pgradient nblocks times instead of
    nblocks * nsteps_per_block times. The overlap matrix itself needs only
    wf.value() and is still accumulated at every step.

    :returns: (weighted, unweighted, configs), matching
        :func:`pyqmc.method.sample_many.sample_overlap`
    """
    rng = pyqmc.rng.get_rng(rng)
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
            rng=rng,
        )
        unweighted.append(block_unweighted)
        wtdp.append(sample_overlap_derivatives(wfs, configs, transform, client, npartitions))
    weighted = {"wtdp": np.asarray(wtdp)}
    unweighted = {k: np.concatenate([u[k] for u in unweighted], axis=0) for k in unweighted[0]}
    return weighted, unweighted, configs


class EnsembleUpdater:
    """Updater for one state of an ensemble.

    :parameter single: a single-state updater (SRUpdate, MinSRUpdate, or
        CGSRUpdate) that samples the energy and solves the SR equations

    With SRUpdate, the overlap derivatives are accumulated at every step of the
    overlap sampling, as SRUpdate does for the energy. With the per-sample
    updaters, they are evaluated on one snapshot per block.
    """

    def __init__(self, single):
        self.single = single

    @property
    def transform(self):
        return self.single.transform

    @property
    def enacc(self):
        return self.single.enacc

    @property
    def eps(self):
        return self.single.eps

    def sample_energy(self, wf, configs, client=None, npartitions=None, verbose=True, rng=None, **kwargs):
        """Sample this state on its own. :returns: (data, configs)"""
        return self.single.sample(
            wf, configs, client=client, npartitions=npartitions, rng=rng, verbose=verbose, **kwargs
        )

    def sample_overlap(self, wfs, configs, client=None, npartitions=None, **kwargs):
        """Sample the overlap distribution of wfs, the last of which is this
        state. :returns: (weighted, unweighted, configs)"""
        if isinstance(self.single, SRUpdate):
            return pyqmc.method.sample_many.sample_overlap(
                wfs,
                configs,
                _OverlapDerivativeAccumulator(self.transform),
                client=client,
                npartitions=npartitions,
                **kwargs,
            )
        return _sample_overlap_snapshots(
            wfs, configs, self.transform, client=client, npartitions=npartitions, **kwargs
        )

    def block_average(self, energy_data, overlap_weighted, overlap_weights):
        """Combine the two samplings.

        :parameter energy_data: output of sample_energy
        :parameter overlap_weighted: weighted output of sample_overlap
        :parameter overlap_weights: (nblocks, nwf, nwf) overlap from sample_overlap
        :returns: (avg, error)
        """
        weight_avg = np.mean(overlap_weights, axis=0)
        N = np.abs(weight_avg.diagonal())
        Nij = np.sqrt(np.outer(N, N))
        wfi = Nij.shape[0] - 1
        it = overlap_weighted["wtdp"]
        avg = {
            "wtdp": np.mean(it, axis=0) / Nij[wfi],
            "overlap": weight_avg,
            "energy": energy_data["energy"],
            "energy_data": energy_data,
        }
        error = {"wtdp": _block_sem(it) / Nij[wfi], "energy": energy_data["energy_error"]}
        return avg, error

    def delta_p(self, steps, data, overlap_penalty, verbose=False):
        """Parameter changes -step * v for each step.

        :parameter steps: list of step sizes
        :parameter data: output of block_average
        :parameter overlap_penalty: (nwf, nwf) penalty matrix
        :returns: (list of parameter changes, report dictionary)
        """
        nwf = data["overlap"].shape[0]
        wfi = nwf - 1
        N = np.abs(data["overlap"].diagonal())
        Nij = np.sqrt(np.outer(N, N))
        overlap = data["overlap"] / Nij
        # derivative of the normalized overlaps with respect to this state's parameters
        fac = np.ones((nwf, nwf)) + np.identity(nwf)
        dp_norm = 2.0 * np.real(data["wtdp"][:, wfi, wfi])
        norm_part = np.einsum("i,p->pi", data["overlap"][wfi, :], dp_norm) / N
        dp_overlap = fac[wfi] * (data["wtdp"][:, wfi, :] - 0.5 * norm_part) / Nij[wfi]

        overlap_cost = 0.0
        ovlp = 0.0
        for i in range(wfi):
            overlap_cost += overlap_penalty[wfi, i] * overlap[wfi, i]
            ovlp += 2.0 * dp_overlap[:, i] * overlap_penalty[wfi, i] * overlap[wfi, i]

        g = None if wfi == 0 else np.real(ovlp)
        v, report = self.single.solve(data["energy_data"], g=g)
        report["overlap_cost"] = overlap_cost
        if verbose:
            print("Overlap cost", overlap_cost)
            print("overlap gradient norm", np.linalg.norm(ovlp))
            print("Gradient norm: ", report["pgrad"])
            print("Dot product between gradient and SR step: ", report["SRdot"])
        return [-step * v for step in steps], report


#: algorithms optimize_ensemble can be asked for by name, and the regularization
#: each one wants by default
UPDATERS = {
    "sr": (SRUpdate, 1e-3),
    "minsr": (MinSRUpdate, 1e-2),
    "cgsr": (CGSRUpdate, 1e-2),
}


def make_updater(transform, enacc, method="sr", eps=None, nodal_cutoff=1e-3):
    """Build the per-state updater for one algorithm.

    :parameter transform: a LinearTransform for this state's parameters
    :parameter enacc: an EnergyAccumulator-like object
    :parameter str method: 'sr' (explicit S matrix), 'minsr' (sample-space
        solve), or 'cgsr' (conjugate gradient)
    :parameter float eps: regularization of the solve; defaults to 1e-3 for sr
        and 1e-2 for minsr and cgsr
    :parameter float nodal_cutoff: regularization distance for the nodal divergence of the derivatives
    """
    if method not in UPDATERS:
        raise ValueError(f"Unknown method {method!r}; choose one of {sorted(UPDATERS)}.")
    cls, default_eps = UPDATERS[method]
    return EnsembleUpdater(
        cls(enacc, transform, eps=default_eps if eps is None else eps, nodal_cutoff=nodal_cutoff)
    )


def build_updaters(transforms, enacc=None, method="sr", eps=None, nodal_cutoff=1e-3):
    """Turn parameter transforms into the nested list of updaters the driver uses.

    :parameter transforms: one transform per state, or a nested list indexed by
        state then sub-iteration. Objects that are already updaters (anything
        with delta_p) are passed through, so a hand-built updater still works.
    :parameter enacc: an EnergyAccumulator-like object shared by every state, or
        a list with one per state
    :returns: nested list of updaters indexed by state then sub-iteration
    """
    nested = [t if isinstance(t, (list, tuple)) else [t] for t in transforms]
    if all(hasattr(t, "delta_p") for state in nested for t in state):
        return [list(state) for state in nested]
    if enacc is None:
        raise ValueError(
            "enacc is required to build updaters from transforms; pass an "
            "EnergyAccumulator, or pass updater objects instead of transforms."
        )
    enaccs = enacc if isinstance(enacc, (list, tuple)) else [enacc] * len(nested)
    if len(enaccs) != len(nested):
        raise ValueError(f"got {len(enaccs)} energy accumulators for {len(nested)} states")
    return [
        [make_updater(t, e, method, eps, nodal_cutoff) for t in state]
        for state, e in zip(nested, enaccs)
    ]
