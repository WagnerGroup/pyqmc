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

import warnings

from pyqmc.method.optimize import LineSearch, SRUpdate, optimize


def line_minimization(
    wf,
    coords,
    pgrad_acc,
    steprange=0.2,
    max_iterations=30,
    warmup_options=None,
    correlated_reference_wfs=None,
    stderr_weight=3.0,
    vmcoptions=None,
    verbose=False,
    npts=40,
    hdf_file=None,
    client=None,
    npartitions=None,
    correlated_sampling=True,
    rng=None,
):
    """Deprecated: use :func:`pyqmc.method.optimize.optimize`.

    Equivalent to ``optimize`` with an :class:`~pyqmc.method.optimize.SRUpdate`
    built from `pgrad_acc` and a :class:`~pyqmc.method.optimize.LineSearch`
    with `steprange`, `npts`, `stderr_weight`, and `correlated_reference_wfs`.
    With ``correlated_sampling=False`` it takes a fixed step of `steprange`
    instead.

    :parameter pgrad_acc: a StochasticReconfiguration accumulator, such as the
        one from :func:`pyqmc.observables.accumulators.gradient_generator`, or a
        list of them for sub-iterations
    :return: optimized wave function, list of per-step data
    """
    warnings.warn(
        "line_minimization is deprecated; use pyqmc.method.optimize.optimize, "
        "which takes a fixed step by default.",
        DeprecationWarning,
        stacklevel=2,
    )
    accs = pgrad_acc if isinstance(pgrad_acc, (list, tuple)) else [pgrad_acc]
    updaters = [
        SRUpdate(
            acc.enacc,
            acc.transform,
            eps=acc.eps,
            nodal_cutoff=acc.nodal_cutoff,
            inverse_strategy=acc.inverse_strategy,
        )
        for acc in accs
    ]
    line_search = None
    if correlated_sampling:
        line_search = LineSearch(
            steprange=steprange,
            npts=npts,
            stderr_weight=stderr_weight,
            correlated_reference_wfs=(0, 1) if correlated_reference_wfs is None else correlated_reference_wfs,
        )
    return optimize(
        wf,
        coords,
        updaters,
        tstep=steprange,
        line_search=line_search,
        max_iterations=max_iterations,
        warmup_options=warmup_options,
        vmc_kwargs=vmcoptions,
        hdf_file=hdf_file,
        client=client,
        npartitions=npartitions,
        verbose=verbose,
        rng=rng,
    )
