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
r"""
Ensemble (excited state) optimization with matrix-free CG stochastic reconfiguration.

:class:`CGSRWfbyWf` plugs into
:func:`pyqmc.method.ensemble_optimization.optimize_ensemble` the same way
:class:`pyqmc.method.ensemble_minsr.MinSRWfbyWf` does -- it is selected by
``method="cgsr"`` and gets the same threading, warmups, and checkpointing::

    optimize_ensemble(wfs, configs, transforms, hdf_file, enacc=enacc,
                      method="cgsr", tau=0.1,
                      vmc_kwargs={"nblocks": 1, "nsteps_per_block": 10})

The algorithm and the sampling are unchanged; it derives from
:class:`~pyqmc.method.ensemble_minsr.MinSRWfbyWf` and overrides only the solve.
Where minSR inverts the (nsamples, nsamples) kernel in closed form, this solves

.. math:: (S + \epsilon I)\, v = f_{\rm energy} + f_{\rm overlap}

by conjugate gradient, forming neither the (nparameters, nparameters) matrix nor
the (nsamples, nsamples) kernel. Use it when the sample count per state is large
enough that minSR's kernel is the problem; see :mod:`pyqmc.method.cgsr` for the
mechanism and for where the crossover sits.

One thing reads more simply here than in minSR. The ensemble gradient has two
pieces: the energy part comes from per-sample derivatives and so lies in the row
space of the sampled derivatives, while the overlap penalty comes from the
separate overlap sampling and generally does not.
:func:`pyqmc.method.minsr.sr_solve` therefore needs a Woodbury branch to apply
the sample-space inverse to it. CG works in parameter space, so the two pieces
are just added into one right hand side.

That is a simplification of the code, not of the mathematics: the component of
the penalty gradient orthogonal to every sampled row still comes back divided by
:math:`\epsilon`, because that is what :math:`(S+\epsilon I)^{-1}` does on the
null space of :math:`S`. Both solvers return the same vector, which
``tests/unit/test_ensemble_cgsr_update.py`` checks.

It does cost CG something, though only where that unresolvable component exists.
On synthetic data with two states, the penalty adds about a fifth to the
iteration count when there are fewer samples than parameters (60 to 72 at
eps=1e-2, 61 to 90 at eps=1e-3 for 50 samples and 200 parameters) and nothing
measurable when there are many more samples than parameters (21 to 22 for 200
samples and 50 parameters), where the sampled rows span everything anyway. As
always with this solver, the hardest case is samples comparable to parameters,
and shrinking eps costs iterations: 153 against 444 at 400 of each.
"""

import numpy as np

from pyqmc.method.cgsr import (
    SROperator,
    preconditioned_conjugate_gradient,
    sr_gradient,
)
from pyqmc.method.ensemble_minsr import MinSRWfbyWf


class CGSRWfbyWf(MinSRWfbyWf):
    r"""Updater for one state of an ensemble, the CG analogue of
    :class:`pyqmc.method.ensemble_minsr.MinSRWfbyWf`.

    Everything except the solve is inherited: the same per-configuration energy
    sampling, the same snapshot-per-block overlap derivatives, and the same block
    averaging.

    :func:`pyqmc.method.ensemble_optimization.make_updater` constructs updaters
    with a fixed signature, so the CG controls are plain attributes with
    defaults. To tune them, build the updater yourself and pass it to
    ``optimize_ensemble`` in place of a transform, which ``build_updaters``
    accepts::

        updaters = [[CGSRWfbyWf(enacc, transform, rtol=1e-8)] for transform in transforms]
        optimize_ensemble(wfs, configs, updaters, hdf_file, tau=0.1)

    :parameter enacc: an EnergyAccumulator-like object
    :parameter transform: a LinearTransform for this state's parameters
    :parameter float eps: regularization of the SR equations
    :parameter float nodal_cutoff: regularization distance for the nodal divergence of the derivatives
    :parameter float rtol: relative residual at which CG stops
    :parameter int max_cg_iterations: iteration cap for CG
    :parameter boolean precondition: use the Jacobi preconditioner
    :parameter boolean warm_start: start each solve from the previous iteration's
        solution for this state
    """

    def __init__(
        self,
        enacc,
        transform,
        eps=1e-2,
        nodal_cutoff=1e-3,
        rtol=1e-6,
        max_cg_iterations=None,
        precondition=True,
        warm_start=True,
    ):
        super().__init__(enacc, transform, eps=eps, nodal_cutoff=nodal_cutoff)
        self.rtol = rtol
        self.max_cg_iterations = max_cg_iterations
        self.precondition = precondition
        self.warm_start = warm_start
        self._previous_v = None

    def delta_p(self, steps, data, overlap_penalty, verbose=False):
        r"""Compute the change in parameters for this state.

        Solves :math:`(S+\epsilon I) v = f_{\rm energy} + f_{\rm overlap}` by
        conjugate gradient. The gradient convention matches
        :class:`pyqmc.method.ensemble_optimization.StochasticReconfigurationWfbyWf`
        -- in particular the energy gradient carries no factor of two here,
        unlike the standalone :func:`pyqmc.method.cgsr.cgsr_update` -- so `steps`
        and `overlap_penalty` mean the same thing they do there.

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

        dppsi = data["dppsi_samples"]
        eloc = data["eloc_samples"]
        # sr_gradient is 2 Re[<O* E> - <O*><E>]; the ensemble convention drops
        # the factor of two, matching StochasticReconfigurationWfbyWf
        pgrad = 0.5 * sr_gradient(dppsi, eloc)
        if wfi > 0:
            pgrad = pgrad + np.real(ovlp)

        operator = SROperator(dppsi, eps=self.eps)
        minv = (
            1.0 / (operator.diagonal() + self.eps) if self.precondition else None
        )
        v, info = preconditioned_conjugate_gradient(
            operator.matvec,
            pgrad,
            minv=minv,
            x0=self._previous_v if self.warm_start else None,
            rtol=self.rtol,
            maxiter=self.max_cg_iterations,
        )
        self._previous_v = v
        dp = [-step * v for step in steps]

        vnorm = np.linalg.norm(v)
        gnorm = np.linalg.norm(pgrad)
        report = {
            "pgrad": gnorm,
            "SRdot": np.dot(pgrad, v) / (vnorm * gnorm) if vnorm * gnorm > 0 else 0.0,
            "overlap_cost": overlap_cost,
            "cg_iterations": info["iterations"],
            "cg_residual": info["residual"],
            "cg_converged": info["converged"],
        }
        if verbose:
            print("Overlap cost", overlap_cost)
            print("overlap gradient norm", np.linalg.norm(ovlp))
            print("Gradient norm: ", report["pgrad"])
            print("Dot product between gradient and SR step: ", report["SRdot"])
            print(
                "CG iterations",
                info["iterations"],
                "relative residual",
                f"{info['residual']:.2e}",
            )
        return dp, report
