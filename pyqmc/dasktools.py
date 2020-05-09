import os
import pyqmc
import numpy as np
import pandas as pd
import h5py
import pyqmc
import pyqmc.optimize_orthogonal

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import dask.distributed

dask.distributed.protocol.utils.msgpack_opts["strict_map_key"] = False


def _avg_func(df):
    scalar_df = df.loc[:, df.dtypes.values != np.dtype("O")]
    scalar_df = scalar_df.groupby("step", as_index=False).mean()

    steps = df["step"].values
    obj_cols = df.columns.values[df.dtypes.values == np.dtype("O")]
    obj_dfs = [df[col].groupby(steps).apply(np.mean, axis=0) for col in obj_cols]
    return pd.concat([scalar_df] + obj_dfs, axis=1).reset_index()


def distvmc(
    wf,
    coords,
    accumulators=None,
    nblocks=100,
    nsteps_per_block=1,
    nsteps=None,
    hdf_file=None,
    npartitions=None,
    nsteps_per=None,
    client=None,
    verbose=False,
    **kwargs
):
    """ 
    Args: 
    wf: a wave function object

    coords: nconf x nelec x 3 

    nblocks: number of VMC blocks

    nsteps_per_block: number of steps per block

    nsteps: (Deprecated) how many steps to move each walker, maps to nblocks = 100, nsteps_per_blocks = 1 

    """

    if nsteps is not None:
        nblocks = nsteps
        nsteps_per_block = 1

    if nsteps_per is None:
        nsteps_per = nblocks

    if hdf_file is not None:
        with h5py.File(hdf_file, "a") as hdf:
            if "configs" in hdf.keys():
                coords.configs = np.array(hdf["configs"])
                if verbose:
                    print("Restarted calculation", flush=True)

    if accumulators is None:
        accumulators = {}
    if npartitions is None:
        npartitions = sum([x for x in client.nthreads().values()])
    niterations = int(nblocks / nsteps_per)
    coord = coords.split(npartitions)
    alldata = []
    for epoch in range(niterations):
        wfs = []
        thiscoord = []
        for i in range(npartitions):
            wfs.append(wf)
            thiscoord.append(coord[i])

        runs = client.map(
            pyqmc.vmc,
            wfs,
            thiscoord,
            **{
                "nblocks": nsteps_per,
                "nsteps_per_block": nsteps_per_block,
                "accumulators": accumulators,
                "stepoffset": epoch * nsteps_per,
            },
            **kwargs
        )

        allresults = list(zip(*[r.result() for r in runs]))
        coords.join(allresults[1])
        iterdata = list(map(pd.DataFrame, allresults[0]))
        confweight = np.array([len(c.configs) for c in coord], dtype=float)
        confweight /= confweight.mean()
        for i, df_ in enumerate(iterdata):
            df_.loc[:, df_.columns != "step"] *= confweight[i]
        df = pd.concat(iterdata)
        collected_data = _avg_func(df).to_dict("records")
        alldata.extend(collected_data)
        for d in collected_data:
            pyqmc.mc.vmc_file(hdf_file, d, kwargs, coords)
        if verbose:
            print("epoch", epoch, "finished", flush=True)

    return alldata, coords


def dist_lm_sampler(
    wf, configs, params, pgrad_acc, npartitions=None, client=None, lm_sampler=None
):
    """
    Evaluates accumulator on the same set of configs for correlated sampling of different wave function parameters.  Parallelized with parsl.

    Args:
        wf: wave function object
        configs: (nconf, nelec, 3) array
        params: (nsteps, nparams) array 
            list of arrays of parameters (serialized) at each step
        pgrad_acc: PGradAccumulator 

    kwargs:
        npartitions: number of tasks for parallelization
            divides configs array into npartitions chunks

    Returns:
        data: list of dicts, one dict for each sample
            each dict contains arrays returned from pgrad_acc, weighted by psi**2/psi0**2 
    """
    if lm_sampler is None:
        from pyqmc.linemin import lm_sampler

    if npartitions is None:
        npartitions = sum([x for x in client.nthreads().values()])

    configspart = configs.split(npartitions)
    allruns = []
    for p in range(npartitions):
        allruns.append(client.submit(lm_sampler, wf, configspart[p], params, pgrad_acc))

    stepresults = []
    for r in allruns:
        stepresults.append(r.result())

    keys = stepresults[0][0].keys()
    # This will be a list of dictionaries
    final_results = []
    for p in range(len(params)):
        df = {}
        for k in keys:
            df[k] = np.concatenate([x[p][k] for x in stepresults], axis=0)
        final_results.append(df)

    return final_results


def line_minimization(*args, client, **kwargs):
    import pyqmc

    if "vmcoptions" not in kwargs:
        kwargs["vmcoptions"] = {}
    if "lmoptions" not in kwargs:
        kwargs["lmoptions"] = {}
    kwargs["vmcoptions"]["client"] = client
    kwargs["lmoptions"]["client"] = client
    return pyqmc.line_minimization(*args, vmc=distvmc, lm=dist_lm_sampler, **kwargs)


def cvmc_optimize(*args, client, **kwargs):
    import pyqmc
    from pyqmc.cvmc import lm_cvmc

    if "vmcoptions" not in kwargs:
        kwargs["vmcoptions"] = {}
    if "lmoptions" not in kwargs:
        kwargs["lmoptions"] = {}
    kwargs["vmcoptions"]["client"] = client
    kwargs["lmoptions"]["client"] = client
    kwargs["lmoptions"]["lm_sampler"] = lm_cvmc
    return pyqmc.cvmc_optimize(*args, vmc=distvmc, lm=dist_lm_sampler, **kwargs)


def distdmc_propagate(wf, configs, weights, *args, client, npartitions=None, **kwargs):
    import pyqmc.dmc

    if npartitions is None:
        npartitions = sum([x for x in client.nthreads().values()])

    coord = configs.split(npartitions)
    weight = np.array_split(weights, npartitions)
    allruns = []
    for nodeconfigs, nodeweight in zip(coord, weight):
        allruns.append(
            client.submit(
                pyqmc.dmc.dmc_propagate, wf, nodeconfigs, nodeweight, *args, **kwargs
            )
        )

    import pandas as pd

    allresults = list(zip(*[r.result() for r in allruns]))
    configs.join(allresults[1])
    coordret = configs
    weightret = np.hstack(allresults[2])

    confweight = np.array([len(c.configs) for c in coord], dtype=float)
    confweight /= confweight.mean()
    iterdata = list(map(pd.DataFrame, allresults[0]))
    for i, df_ in enumerate(iterdata):
        df_.loc[:, df_.columns != "step"] *= confweight[i]
    df = pd.concat(iterdata)
    notavg = ["weight", "weightvar", "weightmin", "weightmax", "acceptance", "step"]
    # Here we reweight the averages since each step on each node
    # was done with a different average weight.

    for k in df.keys():
        if k not in notavg:
            df[k] *= df["weight"].values
    df = df.groupby("step").aggregate(np.mean, axis=0).reset_index()
    for k in df.keys():
        if k not in notavg:
            df[k] /= df["weight"].values
    print("df step weight acceptance\n", df[["step", "weight", "acceptance"]])
    print("energytotal")
    print(df["energytotal"].values)
    return df, configs, weightret


def dist_sample_overlap(wfs, configs, *args, client, npartitions=None, **kwargs):
    if npartitions is None:
        npartitions = sum([x for x in client.nthreads().values()])

    coord = configs.split(npartitions)
    allruns = []

    for nodeconfigs in coord:
        allruns.append(
            client.submit(
                pyqmc.optimize_orthogonal.sample_overlap,
                wfs,
                nodeconfigs,
                *args,
                **kwargs
            )
        )

    # Here we reweight the averages since each step on each node
    # was done with a different average weight.
    confweight = np.array([len(c.configs) for c in coord], dtype=float)
    confweight /= confweight.mean()

    # Memory efficient implementation, bit more verbose
    final_coords = []
    df = {}
    for i, r in enumerate(allruns):
        result = r.result()
        result[0]["weight"] *= confweight[i]
        final_coords.append(result[1])
        keys = result[0].keys()
        for k in keys:
            if k not in df:
                df[k] = np.zeros(result[0][k].shape)
            if k != "weight" and k != "overlap" and k != "overlap_gradient":
                if len(df[k].shape) == 1:
                    df[k] += result[0][k] * result[0]["weight"][:, -1]
                elif len(df[k].shape) == 2:
                    df[k] += result[0][k] * result[0]["weight"][:, -1, np.newaxis]
                elif len(df[k].shape) == 3:
                    df[k] += (
                        result[0][k]
                        * result[0]["weight"][:, -1, np.newaxis, np.newaxis]
                    )
                else:
                    raise NotImplementedError(
                        "too many/too few dimension in dist_sample_overlap"
                    )
            else:
                df[k] += result[0][k] * confweight[i] / len(allruns)

    for k in keys:
        if k != "weight" and k != "overlap" and k != "overlap_gradient":
            if len(df[k].shape) == 1:
                df[k] /= len(allruns) * df["weight"][:, -1]
            elif len(df[k].shape) == 2:
                df[k] /= len(allruns) * df["weight"][:, -1, np.newaxis]
            elif len(df[k].shape) == 3:
                df[k] /= len(allruns) * df["weight"][:, -1, np.newaxis, np.newaxis]

    configs.join(final_coords)
    coordret = configs
    return df, coordret


def dist_correlated_sample(wfs, configs, *args, client, npartitions=None, **kwargs):

    if npartitions is None:
        npartitions = sum([x for x in client.nthreads().values()])

    coord = configs.split(npartitions)
    allruns = []
    for nodeconfigs in coord:
        allruns.append(
            client.submit(
                pyqmc.optimize_orthogonal.correlated_sample,
                wfs,
                nodeconfigs,
                *args,
                **kwargs
            )
        )

    allresults = [r.result() for r in allruns]
    df = {}
    for k in allresults[0].keys():
        df[k] = np.array([x[k] for x in allresults])
    confweight = np.array([len(c.configs) for c in coord], dtype=float)
    confweight /= confweight.mean()
    rhowt = np.einsum("i...,i->i...", df["rhoprime"], confweight)
    wt = df["weight"] * rhowt
    df["total"] = np.average(df["total"], weights=wt, axis=0)
    df["overlap"] = np.average(df["overlap"], weights=confweight, axis=0)
    df["weight"] = np.average(df["weight"], weights=rhowt, axis=0)

    # df["weight"] = np.mean(df["weight"], axis=0)
    df["rhoprime"] = np.mean(rhowt, axis=0)
    return df


def optimize_orthogonal(*args, client, **kwargs):
    if "sample_options" not in kwargs:
        kwargs["sample_options"] = {}
    if "correlated_options" not in kwargs:
        kwargs["correlated_options"] = {}

    kwargs["sample_options"]["client"] = client
    kwargs["correlated_options"]["client"] = client

    return pyqmc.optimize_orthogonal.optimize_orthogonal(
        *args,
        sampler=dist_sample_overlap,
        correlated_sampler=dist_correlated_sample,
        **kwargs
    )
