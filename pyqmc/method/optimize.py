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

r"""Energy optimization of a single wave function by stochastic reconfiguration.

Each iteration samples the wave function, solves the SR equations

.. math:: (S + \epsilon I)\, v = f

for a direction v, and changes the parameters by :math:`-\tau v`. Here f is the
energy gradient and :math:`S` is the overlap matrix of the parameter
derivatives. The updaters differ only in how they sample and solve:

- :class:`SRUpdate` accumulates S during VMC and inverts it. Memory goes as
  nparameters^2.
- :class:`MinSRUpdate` keeps the derivatives per sample and solves in the
  (nsamples, nsamples) sample space.
- :class:`CGSRUpdate` keeps the derivatives per sample and solves by conjugate
  gradient, forming neither matrix.

For the same eps they give the same direction. :func:`optimize` takes a fixed
step :math:`\tau` = `tstep` by default, or chooses it by correlated sampling
with a :class:`LineSearch`.

Typical usage::

    transform = LinearTransform(wf.parameters, to_opt)
    updater = MinSRUpdate(EnergyAccumulator(mol), transform)
    wf, df = optimize(wf, configs, updater, hdf_file="opt.hdf5")
"""

import copy
import logging
import os

import h5py
import numpy as np

import pyqmc.method.mc
import pyqmc.method.sample_many as sm
import pyqmc.rng
import pyqmc.wftools
from pyqmc.method.cgsr import cgsr_direction
from pyqmc.method.minsr import local_energy_error, minsr_direction, sample_minsr_data
from pyqmc.observables.stochastic_reconfiguration import (
    StochasticReconfiguration,
    sr_direction,
)


class SRUpdate:
    """Accumulates the SR averages during VMC and solves with the explicit
    (nparameters, nparameters) S matrix.

    :parameter enacc: an EnergyAccumulator-like object
    :parameter transform: a LinearTransform for the parameters to optimize
    :parameter float eps: regularization
    :parameter float nodal_cutoff: regularization distance for the nodal divergence of the derivatives
    :parameter str inverse_strategy: 'regularized_inverse' or 'pseudo_inverse'
    """

    def __init__(
        self, enacc, transform, eps=1e-2, nodal_cutoff=1e-3, inverse_strategy="regularized_inverse"
    ):
        self.enacc = enacc
        self.transform = transform
        self.eps = eps
        self.inverse_strategy = inverse_strategy
        self._accumulator = StochasticReconfiguration(
            enacc, transform, nodal_cutoff=nodal_cutoff, eps=eps, inverse_strategy=inverse_strategy
        )

    def attrs(self):
        return {"method": "sr", "eps": self.eps, "inverse_strategy": self.inverse_strategy}

    def sample(self, wf, coords, client=None, npartitions=None, rng=None, verbose=False, **vmc_kwargs):
        """Run VMC with the SR accumulator.

        :returns: (data, coords); data holds the block-averaged SR averages and
            'energy' and 'energy_error' from the block means
        """
        df, coords = pyqmc.method.mc.vmc(
            wf,
            coords,
            accumulators={"pgrad": self._accumulator},
            client=client,
            npartitions=npartitions,
            rng=rng,
            **{"verbose": verbose, **vmc_kwargs},
        )
        data = {k: np.mean(df["pgrad" + k], axis=0) for k in self._accumulator.keys()}
        total = df["pgradtotal"]
        data["energy"] = np.mean(total).real
        data["energy_error"] = (np.std(total, axis=0) / np.sqrt(total.shape[0])).real
        return data, coords

    def solve(self, data, g=None):
        """:returns: (v, report) with v = (S + eps)^-1 (f + g)"""
        return sr_direction(data, self.eps, self.inverse_strategy, g=g)


class MinSRUpdate:
    """Keeps the derivatives per sample and solves the SR equations in sample
    space (minSR), never forming S. See :mod:`pyqmc.method.minsr`.

    The number of samples is nconfig * nblocks, and the kernel is square in it,
    so sampling defaults to one block.

    :parameter enacc: an EnergyAccumulator-like object
    :parameter transform: a LinearTransform for the parameters to optimize
    :parameter float eps: regularization
    :parameter float nodal_cutoff: regularization distance for the nodal divergence of the derivatives
    :parameter str inverse_strategy: 'regularized_inverse' or 'pseudo_inverse'
    """

    def __init__(
        self, enacc, transform, eps=1e-2, nodal_cutoff=1e-3, inverse_strategy="regularized_inverse"
    ):
        self.enacc = enacc
        self.transform = transform
        self.eps = eps
        self.nodal_cutoff = nodal_cutoff
        self.inverse_strategy = inverse_strategy

    def attrs(self):
        return {"method": "minsr", "eps": self.eps, "inverse_strategy": self.inverse_strategy}

    def sample(self, wf, coords, client=None, npartitions=None, rng=None, verbose=False, **vmc_kwargs):
        """Sample per-configuration derivatives and local energies; see
        :func:`pyqmc.method.minsr.sample_minsr_data` for vmc_kwargs.

        :returns: (data, coords)
        """
        data, coords = sample_minsr_data(
            wf,
            coords,
            self.transform,
            self.enacc,
            self.nodal_cutoff,
            client=client,
            npartitions=npartitions,
            rng=rng,
            **vmc_kwargs,
        )
        data["energy"] = np.mean(data["total"]).real
        data["energy_error"] = local_energy_error(data["total"])
        return data, coords

    def solve(self, data, g=None):
        """:returns: (v, report) with v = (S + eps)^-1 (f + g)"""
        return minsr_direction(data["dppsi"], data["total"], self.eps, self.inverse_strategy, g=g)


class CGSRUpdate(MinSRUpdate):
    """Same sampling as :class:`MinSRUpdate`, solved by preconditioned conjugate
    gradient. See :mod:`pyqmc.method.cgsr`.

    :parameter float rtol: relative residual at which CG stops
    :parameter int max_cg_iterations: iteration cap for CG
    :parameter boolean precondition: use the Jacobi preconditioner
    :parameter boolean warm_start: start each solve from the previous solution
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

    def attrs(self):
        return {
            "method": "cgsr",
            "eps": self.eps,
            "rtol": self.rtol,
            "precondition": self.precondition,
            "warm_start": self.warm_start,
        }

    def solve(self, data, g=None):
        """:returns: (v, report) with v = (S + eps)^-1 (f + g)"""
        v, report = cgsr_direction(
            data["dppsi"],
            data["total"],
            eps=self.eps,
            rtol=self.rtol,
            max_cg_iterations=self.max_cg_iterations,
            x0=self._previous_v if self.warm_start else None,
            precondition=self.precondition,
            g=g,
        )
        self._previous_v = v
        return v, report


UPDATERS = {"sr": SRUpdate, "minsr": MinSRUpdate, "cgsr": CGSRUpdate}


def make_updater(transform, enacc, method="sr", **kwargs):
    """Build an updater by name: 'sr', 'minsr', or 'cgsr'. kwargs go to its constructor."""
    if method not in UPDATERS:
        raise ValueError(f"Unknown method {method!r}; choose one of {sorted(UPDATERS)}.")
    return UPDATERS[method](enacc, transform, **kwargs)


class LineSearch:
    """Choose the step along the SR direction by correlated sampling.

    The energy is evaluated by correlated sampling at npts steps between
    -steprange/(npts-2) and steprange, and the step that minimizes
    energy + stderr_weight * (standard deviation) is taken.

    :parameter float steprange: largest step tried. If the chosen step is always
        at the edge, increase it; if it is always much smaller, decrease it.
    :parameter int npts: number of steps tried
    :parameter float stderr_weight: weight of the standard deviation. Increase
        this if the optimization struggles to find downhill directions.
    :parameter correlated_reference_wfs: indices of the steps whose wave
        functions are mixed to sample the correlated distribution
    """

    def __init__(self, steprange=0.2, npts=40, stderr_weight=3.0, correlated_reference_wfs=(0, 1)):
        assert npts >= 3, f"LineSearch npts={npts}; need npts >= 3 for correlated sampling"
        self.steprange = steprange
        self.npts = npts
        self.stderr_weight = stderr_weight
        self.correlated_reference_wfs = list(correlated_reference_wfs)

    def attrs(self):
        return {
            "steprange": self.steprange,
            "npts": self.npts,
            "stderr_weight": self.stderr_weight,
            "correlated_reference_wfs": self.correlated_reference_wfs,
        }

    def __call__(self, wf, coords, x0, v, transform, enacc, client=None, npartitions=None, rng=None):
        """:returns: (step, report)"""
        steps = np.linspace(-self.steprange / (self.npts - 2), self.steprange, self.npts)
        params = [x0 + (-step * v) for step in steps]
        correlated_data = correlated_compute(
            wf,
            coords,
            params,
            transform,
            enacc,
            client=client,
            npartitions=npartitions,
            ref_wfs=self.correlated_reference_wfs,
            rng=rng,
        )
        w = correlated_data["weight"].copy()
        w = w / np.mean(w, axis=1, keepdims=True)
        en = np.real(np.mean(correlated_data["total"] * w, axis=1))
        en_std = np.std(correlated_data["total"], axis=1)
        yfit = en + self.stderr_weight * en_std
        est_min = steps[np.argmin(yfit)]
        report = {
            "steps": steps,
            "yfit": yfit,
            "est_min": est_min,
            "correlated_energy": en,
            "correlated_energy_std": en_std,
        }
        return est_min, report


def correlated_compute(
    wf, configs, params, transform, enacc, client=None, npartitions=None, ref_wfs=None, rng=None, **kws
):
    """Evaluate the energy on the same configurations for several parameter sets.

    :parameter params: (nsteps, nparameters) serialized parameters
    :returns: dictionary of (nsteps, nconfig) arrays, including 'weight'
    """
    if ref_wfs is None:
        ref_wfs = [0, 1]
    wfs = [copy.deepcopy(wf) for i in ref_wfs]
    for i in ref_wfs:
        set_wf_params(wfs[i], params[i], transform)
    rng = pyqmc.rng.get_rng(rng)
    # sample combined distribution
    _, _, configs = sm.sample_overlap(
        wfs, configs, None, client=client, npartitions=npartitions, rng=rng, **kws
    )
    if client is None:
        return correlated_compute_worker(wf, configs, params, transform, enacc, ref_wfs, rng=rng)
    config = configs.split(npartitions)
    children = rng.spawn(len(config))
    runs = [
        client.submit(
            correlated_compute_worker, wf, conf, params, transform, enacc, ref_wfs, rng=child
        )
        for conf, child in zip(config, children)
    ]
    allresults = [r.result() for r in runs]
    return {k: np.hstack([res[k] for res in allresults]) for k in allresults[0]}


@pyqmc.rng.worker
def correlated_compute_worker(wf, configs, params, transform, enacc, ref_wfs, *, rng=None):
    """Evaluate the energy for each parameter set on the same configurations.

    :parameter ref_wfs: reference wave functions; the configurations are assumed
        distributed as sum_i |psi_i|^2 over these
    :returns: dictionary of (nsteps, nconfig) arrays, including 'weight'
    """
    data = []
    # every parameter set sees the same ECP quadrature rotations
    current_state = rng.bit_generator.state
    psi = np.zeros((len(params), len(configs.configs)))
    for i, p in enumerate(params):
        rng.bit_generator.state = current_state
        set_wf_params(wf, p, transform)
        psi[i] = wf.recompute(configs)[1]  # recompute gives logdet
        data.append(enacc(configs, wf))

    data_ret = sm.invert_list_of_dicts(data)
    ref = np.amax(psi, axis=0)
    psirel = np.exp(2 * (psi - ref))
    rho = np.mean([psirel[i] for i in ref_wfs], axis=0)
    data_ret["weight"] = psirel / rho
    return data_ret


def set_wf_params(wf, params, transform):
    """Set the wave function parameters from a serialized parameter vector."""
    newparms = transform.deserialize(wf, params)
    for k in newparms:
        wf.parameters[k] = newparms[k]


def opt_hdf(hdf_file, data, attr, configs, parameters):
    import pyqmc.method.hdftools as hdftools

    if hdf_file is not None:
        with h5py.File(hdf_file, "a") as hdf:
            if "configs" not in hdf.keys():
                hdftools.setup_hdf(hdf, data, attr)
                configs.initialize_hdf(hdf)
                hdf.create_group("wf")
                for k, it in parameters.items():
                    hdf.create_dataset("wf/" + k, data=it)
            hdftools.append_hdf(hdf, data)
            configs.to_hdf(hdf)
            for k, it in parameters.items():
                hdf["wf/" + k][...] = it.copy()


def optimize(
    wf,
    coords,
    updater,
    tstep=0.02,
    line_search=None,
    max_norm=None,
    max_iterations=30,
    warmup_options=None,
    vmc_kwargs=None,
    hdf_file=None,
    client=None,
    npartitions=None,
    verbose=False,
    rng=None,
):
    """Optimize the energy by stochastic reconfiguration.

    `updater` may be a list of updaters, in which case each iteration runs one
    sub-iteration per updater, optimizing that updater's parameters while
    holding the rest fixed.

    If `hdf_file` exists, the optimization restarts from it: the parameters and
    configurations are read back and the iteration count continues from the
    last recorded sub-iteration.

    :parameter wf: initial wave function
    :parameter coords: initial configurations
    :parameter updater: an SRUpdate, MinSRUpdate, or CGSRUpdate, or a list of them
    :parameter float tstep: step size along the SR direction; unused with a line search
    :parameter line_search: None for a fixed step, True for a default
        :class:`LineSearch`, or a LineSearch
    :parameter float max_norm: if not None, the maximum 2-norm of a parameter change
    :parameter int max_iterations: total number of iterations, including any read from hdf_file
    :parameter dict warmup_options: kwargs for the initial vmc warmup
    :parameter dict vmc_kwargs: kwargs for the updater's sampling
    :parameter str hdf_file: hdf file to store the optimization
    :parameter client: an object with submit() functions that return futures
    :parameter int npartitions: the number of workers to submit at a time
    :parameter boolean verbose: print output if True
    :parameter rng: anything accepted by :func:`pyqmc.rng.get_rng`; None uses pyqmc's module generator
    :return: optimized wave function, list of per-step data
    """
    updaters = list(updater) if isinstance(updater, (list, tuple)) else [updater]
    if line_search is True:
        line_search = LineSearch()
    rng = pyqmc.rng.get_rng(rng)
    vmc_kwargs = {} if vmc_kwargs is None else dict(vmc_kwargs)
    warmup_options = (
        {"nblocks": 1, "nsteps_per_block": 100} if warmup_options is None else dict(warmup_options)
    )
    if "tstep" not in warmup_options and "tstep" in vmc_kwargs:
        warmup_options["tstep"] = vmc_kwargs["tstep"]

    iteration_offset = 0
    sub_iteration_offset = 0
    if hdf_file is not None and os.path.isfile(hdf_file):  # restarting -- read in data
        pyqmc.wftools.read_wf(wf, hdf_file)
        with h5py.File(hdf_file, "r") as hdf:
            if "iteration" in hdf:
                iteration_offset = np.max(hdf["iteration"][...])
            if "sub_iteration" in hdf:
                # resume at the sub-iteration after the last one recorded
                sub_iteration_offset = hdf["sub_iteration"][-1] + 1
            coords.load_hdf(hdf)
    else:  # not restarting -- VMC warm up period
        if verbose:
            print("starting warmup")
        _, coords = pyqmc.method.mc.vmc(
            wf,
            coords,
            accumulators={},
            client=client,
            npartitions=npartitions,
            rng=rng,
            **warmup_options,
        )
        if verbose:
            print("finished warmup", flush=True)
    if iteration_offset >= max_iterations:
        logging.warning(
            f"iteration_offset {iteration_offset} >= max_iterations {max_iterations}; no steps will be run."
        )

    attr = {"max_iterations": max_iterations, **updaters[0].attrs()}
    if line_search is None:
        attr["tstep"] = tstep
    else:
        attr.update(line_search.attrs())
    if max_norm is not None:
        attr["max_norm"] = max_norm

    df = []
    for it in range(iteration_offset, max_iterations):
        for sub_it in range(sub_iteration_offset, len(updaters)):
            if verbose:
                print("#############################\nStarting iteration", it, "sub iteration", sub_it)
            u = updaters[sub_it]
            x0 = u.transform.serialize_parameters(wf.parameters)

            data, coords = u.sample(
                wf, coords, client=client, npartitions=npartitions, rng=rng, verbose=verbose, **vmc_kwargs
            )
            if np.isnan(data["energy"]):
                raise ValueError(
                    "NaN in optimization. Try reducing the step size or increasing eps."
                )
            if verbose:
                print("Current energy", data["energy"], data["energy_error"])

            v, report = u.solve(data)
            step_data = dict(report)
            if line_search is None:
                tau = tstep
            else:
                tau, search_report = line_search(
                    wf, coords, x0, v, u.transform, u.enacc, client=client, npartitions=npartitions, rng=rng
                )
                step_data.update(search_report)
            dp = -tau * v
            norm = np.linalg.norm(dp)
            step_data["clipped"] = max_norm is not None and norm > max_norm
            if step_data["clipped"]:
                dp = dp * (max_norm / norm)
            step_data["step_norm"] = min(norm, max_norm) if max_norm is not None else norm
            step_data.update(
                tau=tau,
                energy=data["energy"],
                energy_error=data["energy_error"],
                iteration=it,
                sub_iteration=sub_it,
                nconfig=coords.configs.shape[0],
            )
            if verbose:
                print("Step size", tau, "step norm", step_data["step_norm"])

            set_wf_params(wf, x0 + dp, u.transform)
            opt_hdf(hdf_file, step_data, attr, coords, wf.parameters)
            df.append(step_data)
        sub_iteration_offset = 0
    return wf, df
