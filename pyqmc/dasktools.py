import os
import pyqmc
import numpy as np
import pandas as pd
import h5py
import pyqmc
import pyqmc.optimize_orthogonal
#import dask.distributed

#dask.distributed.protocol.utils.msgpack_opts["strict_map_key"] = False


def _avg_func(df):
    scalar_df = df.loc[:, df.dtypes.values != np.dtype("O")]
    scalar_df = scalar_df.groupby("step", as_index=False).mean()

    steps = df["step"].values
    obj_cols = df.columns.values[df.dtypes.values == np.dtype("O")]
    obj_dfs = [df[col].groupby(steps).apply(np.mean, axis=0) for col in obj_cols]
    return pd.concat([scalar_df] + obj_dfs, axis=1).reset_index()



def dist_sample_overlap(wfs, configs, *args, client, npartitions=None, **kwargs):
    if npartitions is None:
        npartitions = sum(client.nthreads().values())

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
            if k not in ["weight", "overlap", "overlap_gradient"]:
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
        if k not in ["weight", "overlap", "overlap_gradient"]:
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
        npartitions = sum(client.nthreads().values())

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
