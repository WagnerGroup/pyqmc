"""Choosing a stochastic-reconfiguration solver.

`recommend_optimizer` estimates -- and, when cheap enough, directly benchmarks
-- the peak memory and solve time of the three mathematically equivalent
stochastic-reconfiguration solvers: SRUpdate (SR), MinSRUpdate (minSR), and
CGSRUpdate (CG-SR). The ensemble updaters use the same solves. They solve the same linear system
and produce the same optimization step; they differ only in cost, so the choice
is made on memory and time at your (nparam, nconfig).
"""
import gc
import time
import tracemalloc

import numpy as np
import pandas as pd

from pyqmc.method.optimize import CGSRUpdate, MinSRUpdate, SRUpdate
from pyqmc.observables.stochastic_reconfiguration import sr_averages

from ._util import get_true_size

_UPDATERS = {
    "sr": SRUpdate,          # builds the (nparam, nparam) S matrix
    "minsr": MinSRUpdate,    # inverts the (nconfig, nconfig) kernel
    "cgsr": CGSRUpdate,      # matrix-free conjugate gradient
}

_BYTES_PER = 8  # float64


def _fake_sample_data(method, nconfig, nparam, seed=0):
    """A stand-in for one sampling's output, holding only what `method`'s solve
    reads -- and, crucially, only the large array that method keeps resident
    (the nparam^2 S-matrix for sr, the nconfig x nparam per-sample derivatives
    for minsr/cgsr).

    The parameters are real (LinearTransform serializes complex wf parameters
    into real components), so the fake derivatives are real too, which keeps S
    real symmetric PSD and the regularized inverse well posed.
    """
    rng = np.random.default_rng(seed)
    O = rng.standard_normal((nconfig, nparam))          # log-derivatives O_si
    eloc = -67.0 + 0.3 * rng.standard_normal(nconfig)   # local energies
    if method == "sr":
        # sr keeps the averaged matrices; the per-sample O is discarded upstream
        return sr_averages({"total": eloc}, O)
    return {"dppsi": O, "total": eloc}  # minsr, cgsr keep the per-sample derivatives


def _estimate_elements(method, nparam, nconfig):
    """Peak solver footprint in float64 elements. The constants are calibrated
    against the benchmark (peak/estimate ~ 1.0 across regimes):

      * sr    -- input dpidpj + Sij + (Sij+eps*I) + invSij: ~4 nparam^2.
      * minsr -- held derivatives + an explicit real design copy
                 (~2 nconfig*nparam) plus the sample kernel and its copy
                 (~2 nconfig^2). The held-derivatives term dominates in minsr's
                 own regime (nparam >> nconfig), so it must be counted.
      * cgsr  -- only the per-sample derivatives (nconfig*nparam); no matrix.
    """
    if method == "sr":
        return 4 * nparam * nparam
    if method == "minsr":
        return 2 * nconfig * nparam + 2 * nconfig * nconfig
    if method == "cgsr":
        return nconfig * nparam
    raise ValueError(f"unknown method {method}")


def _estimate_mb(method, nparam, nconfig):
    """Analytical estimate of the peak solver footprint, in MB."""
    return _estimate_elements(method, nparam, nconfig) * _BYTES_PER / 1024**2


def _benchmark(cls, method, nparam, nconfig, ntime):
    """Actually build fake data and run the solve, measuring memory and time."""
    data = _fake_sample_data(method, nconfig, nparam)
    held_mb = get_true_size(data) / 1024**2
    updater = cls(None, None)                  # solve ignores enacc/transform
    # each real optimization step resamples, so warm-starting cgsr from the
    # previous solution is not representative -- measure the cold solve.
    if hasattr(updater, "warm_start"):
        updater.warm_start = False
    updater.solve(data)                        # warmup (first-touch / BLAS init)

    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    _, report = updater.solve(data)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    solve_mb = max(peak - base, 0) / 1024**2

    best = np.inf
    for _ in range(ntime):
        t0 = time.perf_counter()
        updater.solve(data)
        best = min(best, time.perf_counter() - t0)
    return held_mb, solve_mb, best * 1e3, report.get("cg_iterations", np.nan)


def recommend_optimizer(to_opt, nconfig, max_memory=1.0, ntime=3):
    """Recommend a stochastic-reconfiguration solver by cost.

    SR, minSR, and CG-SR solve the *same* linear system and produce the same
    optimization step; they differ only in how much memory and time the solve
    costs at a given (nparam, nconfig). This tabulates that cost so the choice
    is made on resources.

    For each method the peak solver memory is first *estimated* analytically
    (see ``_estimate_elements``: ~4 nparam^2 for sr, 2 nconfig*nparam +
    2 nconfig^2 for minsr, nconfig*nparam for cgsr). If that estimate *exceeds*
    ``max_memory`` (in GB) the method's benchmark is skipped -- it would
    allocate more than the budget -- and only the estimate is reported.
    Otherwise it is safe to allocate, so a real benchmark of the solve runs on
    synthetic data of the right shape and measures:

      * ``held_MB``    -- resident input footprint (get_true_size).
      * ``solve_MB``   -- extra memory the solve allocates (tracemalloc peak).
      * ``delta_p_ms`` -- wall time of one solve (min over ``ntime`` calls).

    Parameters
    ----------
    to_opt : dict
        Optimization mask, ``{parameter_key: boolean ndarray}`` as returned by
        ``pyq.generate_wf``. The total number of True entries is the parameter
        count nparam.
    nconfig : int
        Number of configurations (samples) the optimizer will use.
    max_memory : float, optional
        Memory budget in GB (default 1.0). A method whose *estimate* exceeds it
        is reported from the estimate only, without allocating.
    ntime : int, optional
        Number of solve timings; the minimum is reported (default 3).

    Returns
    -------
    pandas.DataFrame
        One row per method, with columns ``method``, ``nparam``, ``nconfig``,
        ``est mem (MB)``, ``measured``, and (when measured) ``held_MB``,
        ``solve_MB``, ``delta_p_ms``, ``cg_iters``.
    """
    nparam = sum(int(np.sum(mask)) for mask in to_opt.values())

    max_memory_mb = max_memory * 1024
    rows = []
    for method, cls in _UPDATERS.items():
        est_mb = _estimate_mb(method, nparam, nconfig)
        row = {"method": method, "nparam": nparam, "nconfig": nconfig,
               "est mem (MB)": round(est_mb, 1)}
        if est_mb > max_memory_mb:
            # would exceed the budget -- don't allocate it; report the estimate
            row.update(measured=False, held_MB=np.nan, solve_MB=np.nan,
                       delta_p_ms=np.nan, cg_iters=np.nan)
        else:
            # safe to allocate -- run the real benchmark
            held_mb, solve_mb, t_ms, cgit = _benchmark(
                cls, method, nparam, nconfig, ntime)
            row.update(measured=True, held_MB=round(held_mb, 1),
                       solve_MB=round(solve_mb, 1), delta_p_ms=round(t_ms, 2),
                       cg_iters=cgit)
        rows.append(row)

    return pd.DataFrame(rows)
