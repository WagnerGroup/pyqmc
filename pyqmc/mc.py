# This must be done BEFORE importing numpy or anything else.
# Therefore it must be in your main script.
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import h5py


def initial_guess(mol, nconfig, r=1.0):
    """ Generate an initial guess by distributing electrons near atoms
    proportional to their charge.

    assign electrons to atoms based on atom charges
    assign the minimum number first, and assign the leftover ones randomly
    this algorithm chooses atoms *with replacement* to assign leftover electrons

    Args: 

     mol: A PySCF-like molecule object. Should have atom_charges(), atom_coords(), and nelec

     nconfig: How many configurations to generate.

     r: How far from the atoms to distribute the electrons

    Returns: 

     A numpy array with shape (nconfig,nelectrons,3) with the electrons randomly distributed near 
     the atoms.
    
    """
    from pyqmc.coord import OpenConfigs, PeriodicConfigs

    epos = np.zeros((nconfig, np.sum(mol.nelec), 3))
    wts = mol.atom_charges()
    wts = wts / np.sum(wts)

    for s in [0, 1]:
        neach = np.array(
            np.floor(mol.nelec[s] * wts), dtype=int
        )  # integer number of elec on each atom
        nleft = (
            mol.nelec[s] * wts - neach
        )  # fraction of electron unassigned on each atom
        nassigned = np.sum(neach)  # number of electrons assigned
        totleft = int(mol.nelec[s] - nassigned)  # number of electrons not yet assigned
        ind0 = s * mol.nelec[0]
        epos[:, ind0 : ind0 + nassigned, :] = np.repeat(
            mol.atom_coords(), neach, axis=0
        )  # assign core electrons
        if totleft > 0:
            bins = np.cumsum(nleft) / totleft
            inds = np.argpartition(
                np.random.random((nconfig, len(wts))), totleft, axis=1
            )[:, :totleft]
            epos[:, ind0 + nassigned : ind0 + mol.nelec[s], :] = mol.atom_coords()[
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

    Args:
      g: a [nconf,ndim] vector
      
      cutoff: the maximum magnitude

    Returns: 
      The vector with the cut off applied.
    """
    tot = np.linalg.norm(g, axis=1)
    mask = tot > cutoff
    g[mask, :] = cutoff * g[mask, :] / tot[mask, np.newaxis]
    return g


def vmc_file(hdf_file, data, attr, configs):
    import pyqmc.hdftools as hdftools

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

    Return a dictionary of averages from each accumulator.  
    """
    nconf, nelec, _ = configs.configs.shape
    block_avg = {}
    wf.recompute(configs)

    for _ in range(nsteps):
        acc = 0.0
        for e in range(nelec):
            # Propose move
            grad = limdrift(np.real(wf.gradient(e, configs.electron(e)).T))
            gauss = np.random.normal(scale=np.sqrt(tstep), size=(nconf, 3))
            newcoorde = configs.configs[:, e, :] + gauss + grad * tstep
            newcoorde = configs.make_irreducible(e, newcoorde)

            # Compute reverse move
            new_grad = limdrift(np.real(wf.gradient(e, newcoorde).T))
            forward = np.sum(gauss ** 2, axis=1)
            backward = np.sum((gauss + tstep * (grad + new_grad)) ** 2, axis=1)

            # Acceptance
            t_prob = np.exp(1 / (2 * tstep) * (forward - backward))
            ratio = np.multiply(wf.testvalue(e, newcoorde) ** 2, t_prob)
            accept = ratio > np.random.rand(nconf)

            # Update wave function
            configs.move(e, newcoorde, accept)
            wf.updateinternals(e, newcoorde, mask=accept)
            acc += np.mean(accept) / nelec

        # Rolling average on step
        for k, accumulator in accumulators.items():
            dat = accumulator.avg(configs, wf)
            for m, res in dat.items():
                if k + m not in block_avg:
                    block_avg[k + m] = res / nsteps
                else:
                    block_avg[k + m] += res / nsteps
        block_avg["acceptance"] = acc
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
    nblocks=10,
    nsteps_per_block=10,
    nsteps=None,
    tstep=0.5,
    accumulators=None,
    verbose=False,
    stepoffset=0,
    hdf_file=None,
    client=None,
    npartitions=None,
):
    """Run a Monte Carlo sample of a given wave function.

    Args:
      wf: A Wave function-like class. recompute(), gradient(), and updateinternals() are used, as well as 
      anything (such as laplacian() ) used by accumulators
      
      configs: Initial electron coordinates

      nblocks: Number of VMC blocks to run 

      nsteps_per_block: Number of steps to run per block

      nsteps: (Deprecated) Number of steps to run, maps to nblocks = 1, nsteps_per_block = nsteps

      tstep: Time step for move proposals. Only affects efficiency.

      accumulators: A dictionary of functor objects that take in (configs,wf) and return a dictionary of quantities to be averaged. np.mean(quantity,axis=0) should give the average over configurations. If None, then the coordinates will only be propagated with acceptance information.
      
      verbose: Print out step information 

      stepoffset: If continuing a run, what to start the step numbering at.
  
      hdf_file: Hdf_file to store vmc output.

      client: an object with submit() functions that return futures

      nworkers: the number of workers to submit at a time

    Returns: (df,configs)
       df: A list of dictionaries nstep long that contains all results from the accumulators. These are averaged across all walkers.

       configs: The final coordinates from this calculation.
       
    """
    if nsteps is not None:
        nblocks = nsteps
        nsteps_per_block = 1

    if accumulators is None:
        accumulators = {}
        if verbose:
            print("WARNING: running VMC with no accumulators")

    # Restart
    if hdf_file is not None:
        with h5py.File(hdf_file, "a") as hdf:
            if "configs" in hdf.keys():
                stepoffset = hdf["block"][-1] + 1
                configs.load_hdf(hdf)
                if verbose:
                    print("Restarting calculation from step", stepoffset)

    df = []

    for block in range(nblocks):
        if verbose:
            print(f"-", end="", flush=True)
        if client is None:
            block_avg, configs = vmc_worker(
                wf, configs, tstep, nsteps_per_block, accumulators
            )
        else:
            block_avg, configs = vmc_parallel(
                wf, configs, tstep, nsteps_per_block, accumulators, client, npartitions
            )
        # Append blocks
        block_avg["block"] = stepoffset + block
        block_avg["nconfig"] = nsteps_per_block * configs.configs.shape[0]
        vmc_file(hdf_file, block_avg, dict(tstep=tstep), configs)
        df.append(block_avg)
    if verbose:
        print("vmc done")

    df_return = {}
    for k in df[0].keys():
        df_return[k] = np.asarray([d[k] for d in df])
    return df_return, configs
