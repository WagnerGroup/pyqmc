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


import pyqmc.method.sample_many
import numpy as np
import h5py
from pyqmc.method import hdftools
import pyqmc.gpu as gpu
import os


def hdf_save(hdf_file, data, attr, wfs):
    if hdf_file is not None:
        with h5py.File(hdf_file, "a") as hdf:
            if "energy" not in hdf.keys():
                hdftools.setup_hdf(hdf, data, attr)
                for wfi, wf in enumerate(wfs):
                    for k, it in wf.parameters.items():
                        hdf.create_dataset(f"wf/{wfi}/" + k, data=it)

            hdftools.append_hdf(hdf, data)
            for wfi, wf in enumerate(wfs):
                for k, it in wf.parameters.items():
                    hdf[f"wf/{wfi}/" + k][...] = it.copy()


def set_wf_params(wfs, params, updater):
    for wf, p, transform in zip(wfs, params, updater.transforms):
        newparms = transform.deserialize(wf, p)
        for k in newparms.keys():
            wf.parameters[k] = newparms[k]


def renormalize(wfs, norms, pivot=0, N=1):
    """
    Renormalize the wave functions so that they have the same normalization as the pivot wave function.

    .. math::

    """
    for i, wf in enumerate(wfs):
        if i == pivot:
            continue
        renorm = np.sqrt(norms[pivot] / norms[i] * N)
        if "wf1det_coeff" in wfs[-1].parameters.keys():
            wf.parameters["wf1det_coeff"] *= renorm
        elif "det_coeff" in wfs[-1].parameters.keys():
            wf.parameters["det_coeff"] *= renorm
        else:
            raise NotImplementedError("need wf1det_coeff or det_coeff in parameters")


def optimize_ensemble(
    wfs,
    configs,
    updater,
    hdf_file,
    tau=1,
    max_iterations=100,
    overlap_penalty=None,
    npartitions=None,
    client=None,
    verbose=False,
    vmc_kwargs={},
):
    """Optimize a set of wave functions using ensemble VMC.


    Returns
    -------

    wfs : list of optimized wave functions
    """

    nwf = len(wfs)
    if overlap_penalty is None:
        overlap_penalty = np.ones((nwf, nwf)) * 0.5

    try:
        sub_iterations = len(updater)
    except TypeError:
        if verbose:
            print(
                "Was passed a single PGradAccumulator; using 1 sub_iteration. This is deprecated behavior."
            )
        sub_iterations = 1
        updater = [updater]

    iteration_offset = 0
    sub_iteration_offset = 0
    if hdf_file is not None and os.path.isfile(hdf_file):  # restarting -- read in data
        with h5py.File(hdf_file, "r") as hdf:
            if "wf" in hdf.keys():
                for wfi, wf in enumerate(wfs):
                    grp = hdf[f"wf/{wfi}"]
                    for k in grp.keys():
                        wf.parameters[k] = gpu.cp.asarray(grp[k])
            if "iteration" in hdf.keys():
                iteration_offset = np.max(hdf["iteration"][...]) + 1
            if "sub_iteration" in hdf.keys():
                sub_iteration_offset = hdf["sub_iteration"][-1] + 1
            configs.load_hdf(hdf)

    for i in range(iteration_offset, max_iterations):
        for sub_iteration in range(sub_iterations):
            if i == iteration_offset:
                if sub_iteration < sub_iteration_offset:
                    continue
            data_weighted, data_unweighted, configs = pyqmc.method.sample_many.sample_overlap(
                wfs,
                configs,
                None,
                nsteps=10,
                nblocks=20,
                client=client,
                npartitions=npartitions,
                **vmc_kwargs,
            )
            update = updater[sub_iteration]
            norm = np.mean(data_unweighted["overlap"], axis=0)
            if verbose:
                print("Normalization step", norm.diagonal())
            renormalize(wfs, norm.diagonal(), pivot=0)

            data_weighted, data_unweighted, configs = pyqmc.method.sample_many.sample_overlap(
                wfs,
                configs,
                update,
                client=client,
                npartitions=npartitions,
                **vmc_kwargs,
            )
            avg, error = update.block_average(data_weighted, data_unweighted["overlap"])
            if verbose:
                print("Iteration", i, "Energy", avg["total"], "Overlap", avg["overlap"])
            dp, report = update.delta_p([tau], avg, overlap_penalty, verbose=True)
            x = [
                transform.serialize_parameters(wf.parameters)
                for wf, transform in zip(wfs, update.transforms)
            ]
            x = [x_ + dp_[0] for x_, dp_ in zip(x, dp)]
            set_wf_params(wfs, x, update)

            save_data = {
                "energy": avg["total"],
                "energy_error": error["total"],
                "overlap": avg["overlap"],
                "iteration": i,
                "sub_iteration": sub_iteration,
            }
            for k in ["total"]:
                save_data[str(k)] = data_weighted[k]
            save_data.update(data_unweighted)
            save_data.update(report)
            hdf_save(hdf_file, save_data, {}, wfs)

    return wfs
