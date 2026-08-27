"""Choosing how many configurations to run.

`benchmark_nconfig` times the core per-configuration wavefunction operations and
measures the wavefunction's memory footprint as nconfig grows, so you can pick
the smallest nconfig that is still computationally efficient (real work
dominating Python call overhead) and that stays within a memory budget. Both
are hardware-dependent, so run it on the machine you will compute on. It is not
specific to optimization -- the same sizing applies to a VMC or DMC run.
"""
import time

import pandas as pd
import pyqmc.api as pyq

from ._util import get_true_size


def benchmark_nconfig(mol, wf, nconfigs=None):
    """Benchmark a wavefunction's speed and memory as a function of nconfig.

    For each configuration count in `nconfigs`, a fresh walker ensemble is
    generated and the core operations a sampler calls every step are timed:
    ``recompute``, ``gradient_value``, ``gradient_laplacian``, and the local
    energy -- each reported *per configuration in milliseconds* (total time /
    nconfig). The wavefunction object's memory footprint is also recorded via
    ``get_true_size``; it grows with nconfig because the wavefunction caches
    per-configuration intermediates (orbital values, inverses) internally.

    How to read the result to choose nconfig:

      * Efficiency. The per-configuration times start high and *fall* as nconfig
        grows, then flatten. The knee is the smallest efficient nconfig: below
        it, Python and per-call overhead dominate the cost of each
        configuration; above it, you are paying for genuine vectorized work.
        The knee is hardware- and system-dependent, so run this on the machine
        you will actually compute on.
      * Memory. The ``wf data (MB)`` column bounds how large nconfig can grow
        before the wavefunction's own cache exhausts the memory budget.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule, used to build the local-energy accumulator and the initial
        walker guess.
    wf : pyqmc wavefunction
        Wavefunction to benchmark. Its parameters are not modified.
    nconfigs : sequence of int, optional
        Configuration counts to test. Defaults to [50, 100, 200, 400, 600].

    Returns
    -------
    pandas.DataFrame
        One row per nconfig with columns ``nconfig``, ``wf data (MB)``, and the
        per-configuration times in milliseconds: ``recompute (ms)``,
        ``gradient_value (ms)``, ``gradient_laplacian (ms)``, ``energy (ms)``.
    """
    if nconfigs is None:
        nconfigs = [50, 100, 200, 400, 600]

    energy = pyq.EnergyAccumulator(mol)
    rows = []
    for nconfig in nconfigs:
        configs = pyq.initial_guess(mol, nconfig)

        t0 = time.perf_counter()
        wf.recompute(configs)
        t_recompute = time.perf_counter() - t0

        t0 = time.perf_counter()
        wf.gradient_value(0, configs.electron(0))
        t_gradient_value = time.perf_counter() - t0

        t0 = time.perf_counter()
        wf.gradient_laplacian(0, configs.electron(0))
        t_gradient_laplacian = time.perf_counter() - t0

        t0 = time.perf_counter()
        energy(configs, wf)
        t_energy = time.perf_counter() - t0

        ms_per_config = 1e3 / nconfig
        rows.append({
            "nconfig": nconfig,
            "wf data (MB)": get_true_size(wf) / 1024**2,
            "recompute (ms)": t_recompute * ms_per_config,
            "gradient_value (ms)": t_gradient_value * ms_per_config,
            "gradient_laplacian (ms)": t_gradient_laplacian * ms_per_config,
            "energy (ms)": t_energy * ms_per_config,
        })
    return pd.DataFrame(rows)
