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

# This must be done BEFORE importing numpy or anything else.
# Therefore it must be in your main script.
import os
import numpy as np
import h5py
import logging
import pyqmc.method.hdftools as hdftools
import time

def initial_guess(mol, nconfig, r=1.0):
    """Generate an initial guess by distributing electrons near atoms
    proportional to their charge.

    assign electrons to atoms based on atom charges
    assign the minimum number first, and assign the leftover ones randomly
    this algorithm chooses atoms *with replacement* to assign leftover electrons

    :parameter mol: A PySCF-like molecule object. Should have atom_charges(), atom_coords(), and nelec
    :parameter nconfig: How many configurations to generate.
    :parameter r: How far from the atoms to distribute the electrons
    :returns: (nconfig,nelectrons,3) array of electron positions randomly distributed near the atoms.
    :rtype: ndarray

    """
    from pyqmc.configurations.coord import OpenConfigs, PeriodicConfigs

    epos = np.zeros((nconfig, np.sum(mol.nelec), 3))
    wts = mol.atom_charges()
    wts = wts / np.sum(wts)

    for s in [0, 1]:
        neach = np.array(
            np.floor(mol.nelec[s] * wts), dtype=int
        )  # integer number of elec on each atom
        #nleft = (
        #    mol.nelec[s] * wts - neach
        #)  # fraction of electron unassigned on each atom
        nassigned = np.sum(neach)  # number of electrons assigned
        totleft = int(mol.nelec[s] - nassigned)  # number of electrons not yet assigned
        ind0 = s * mol.nelec[0]
        epos[:, ind0:ind0 + nassigned, :] = np.repeat(
            mol.atom_coords(), neach, axis=0
        )  # assign core electrons
        if totleft > 0:
            #bins = np.cumsum(nleft) / totleft
            inds = np.argpartition(
                np.random.random((nconfig, len(wts))), totleft, axis=1
            )[:, :totleft]
            epos[:, ind0 + nassigned:ind0 + mol.nelec[s], :] = mol.atom_coords()[
                inds
            ]  # assign remaining electrons

    epos += r * np.random.randn(*epos.shape)  # random shifts from atom positions
    if hasattr(mol, "a"):
        epos = PeriodicConfigs(epos, mol.lattice_vectors())
    else:
        epos = OpenConfigs(epos)
    return epos


def limdrift(g, cutoff=1):
    """
    Limit a vector to have a maximum magnitude of cutoff while maintaining direction

    :parameter g: a [nconf,ndim] vector
    :parameter cutoff: the maximum magnitude
    :returns: The vector with the cutoff applied.
    """
    tot = np.linalg.norm(g, axis=1)
    mask = tot > cutoff
    # by using where we can avoid modifying the original array
    # and so we have JAX compatibility
    g = np.where(mask[:,np.newaxis], cutoff * g / tot[:, np.newaxis], g)
    return g


def vmc_file(hdf_file, data, attr, configs):

    if hdf_file is not None:
        with h5py.File(hdf_file, "a") as hdf:
            if "configs" not in hdf.keys():
                hdftools.setup_hdf(hdf, data, attr)
                configs.initialize_hdf(hdf)
            hdftools.append_hdf(hdf, data)
            configs.to_hdf(hdf)


def vmc_worker(wf, configs, tstep, nsteps, accumulators):
    """
    Run VMC for nsteps.

    :return: a dictionary of averages from each accumulator.
    """
    nconf, nelec, _ = configs.configs.shape
    block_avg = {}
    wf.recompute(configs)

    for _ in range(nsteps):
        acc = 0.0
        start_move = time.perf_counter()
        for e in range(nelec):
            # Propose move
            g, _, _ = wf.gradient_value(e, configs.electron(e))
            grad = limdrift(np.real(g.T))
            gauss = np.random.normal(scale=np.sqrt(tstep), size=(nconf, 3))
            newcoorde = configs.configs[:, e, :] + gauss + grad * tstep
            newcoorde = configs.make_irreducible(e, newcoorde)

            # Compute reverse move
            g, new_val, saved = wf.gradient_value(e, newcoorde)
            new_grad = limdrift(np.real(g.T))
            forward = np.sum(gauss**2, axis=1)
            backward = np.sum((gauss + tstep * (grad + new_grad)) ** 2, axis=1)

            # Acceptance
            t_prob = np.exp(1 / (2 * tstep) * (forward - backward))
            ratio = np.abs(new_val) ** 2 * t_prob
            accept = ratio > np.random.rand(nconf)

            # Update wave function
            configs.move(e, newcoorde, accept)
            wf.updateinternals(e, newcoorde, configs, mask=accept, saved_values=saved)
            acc += np.mean(accept) / nelec
        end_move = time.perf_counter()

        # Rolling average on step
        start_average = time.perf_counter()
        for k, accumulator in accumulators.items():
            dat = accumulator.avg(configs, wf)
            for m, res in dat.items():
                if k + m not in block_avg:
                    block_avg[k + m] = res / nsteps
                else:
                    block_avg[k + m] += res / nsteps
        end_average = time.perf_counter()
        block_avg["acceptance"] = acc
        block_avg['move time'] = end_move - start_move
        block_avg['accumulator time'] = end_average - start_average
    return block_avg, configs


def vmc_parallel(
    wf, configs, tstep, nsteps_per_block, accumulators, client, npartitions
):
    config = configs.split(npartitions)
    runs = [
        client.submit(vmc_worker, wf, conf, tstep, nsteps_per_block, accumulators)
        for conf in config
    ]
    allresults = list(zip(*[r.result() for r in runs]))
    configs.join(allresults[1])
    confweight = np.array([len(c.configs) for c in config], dtype=float)
    confweight /= np.mean(confweight) * npartitions
    block_avg = {}
    for k in allresults[0][0].keys():
        block_avg[k] = np.sum(
            [res[k] * w for res, w in zip(allresults[0], confweight)], axis=0
        )
    return block_avg, configs


def vmc(
    wf,
    configs,
    tstep=0.5,
    nblocks=10,
    nsteps_per_block=10,
    nsteps=None,
    blockoffset=0,
    accumulators=None,
    verbose=False,
    hdf_file=None,
    continue_from=None,
    client=None,
    npartitions=None,
):
    """Run a Monte Carlo sample of a given wave function.

    :parameter wf: trial wave function for VMC
    :type wf: a PyQMC wave-function-like object
    :parameter configs: (nconfig, nelec, 3) - initial electron coordinates to start calculation.
    :type configs: PyQMC configs object
    :parameter float tstep: Time step for move proposals. Only affects efficiency.
    :parameter int nblocks: Number of VMC blocks to run. If a calculation is continued (either from continue_from or from using the same hdf_file as a previous call), nblocks includes the blocks from previous calls; i.e., nblocks is the total number of blocks run over all the calls to vmc.
    :parameter int nsteps_per_block: Number of steps to run per block
    :parameter int nsteps: (Deprecated) Number of steps to run, maps to nblocks = nsteps, nsteps_per_block = 1
    :parameter int blockoffset: If continuing a run, what to start the block numbering at. The calculation will stop when the block number reaches nblocks.
    :parameter accumulators: A dictionary of functor objects that take in (configs,wf) and return a dictionary of quantities to be averaged. np.mean(quantity,axis=0) should give the average over configurations. If None, then the coordinates will only be propagated with acceptance information.
    :parameter boolean verbose: Print out step information
    :parameter str hdf_file: Hdf_file to store vmc output.
    :parameter str continue_from: Hdf_file to continue vmc calculation from.
    :parameter client: an object with submit() functions that return futures
    :parameter int npartitions: the number of workers to submit at a time
    :returns: (df,configs)
       df: A list of dictionaries nstep long that contains all results from the accumulators. These are averaged across all walkers.

       configs: The final coordinates from this calculation.
    :rtype: list of dictionaries, pyqmc.coord.Configs

    """
    if nsteps is not None:
        nblocks = nsteps
        nsteps_per_block = 1

    if accumulators is None:
        accumulators = {}
        if verbose:
            print("WARNING: running VMC with no accumulators")

    # Restart
    if continue_from is None:
        continue_from = hdf_file
    elif not os.path.isfile(continue_from):
        raise RuntimeError("cannot continue from {0}; the file does not exist!")
    elif hdf_file is not None and os.path.isfile(hdf_file):
        raise RuntimeError(
            "continue_from is not None but hdf_file={0} already exists! Delete or rename {0} and try again.".format(
                hdf_file
            )
        )
    if continue_from is not None and os.path.isfile(continue_from):
        with h5py.File(continue_from, "r") as hdf:
            if "configs" in hdf.keys():
                blockoffset = hdf["block"][-1] + 1
                configs.load_hdf(hdf)
                if verbose:
                    print(
                        f"Restarting calculation {continue_from} from block {blockoffset}"
                    )

    df = []

    if blockoffset >= nblocks:
        logging.warning(
            f"blockoffset {blockoffset} >= nblocks {nblocks}; no steps will be run."
        )
    for block in range(blockoffset, nblocks):
        if verbose:
            print("-", end="", flush=True)
        if client is None:
            block_avg, configs = vmc_worker(
                wf, configs, tstep, nsteps_per_block, accumulators
            )
        else:
            block_avg, configs = vmc_parallel(
                wf, configs, tstep, nsteps_per_block, accumulators, client, npartitions
            )
        # Append blocks
        block_avg["block"] = block
        block_avg["nconfig"] = nsteps_per_block * configs.configs.shape[0]
        vmc_file(hdf_file, block_avg, dict(tstep=tstep), configs)
        df.append(block_avg)
    if verbose:
        print("vmc done")

    df_return = {}
    if len(df) > 0:
        for k in df[0].keys():
            df_return[k] = np.asarray([d[k] for d in df])
    return df_return, configs
