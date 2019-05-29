import pyblock
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import os
#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["NUMEXPR_NUM_THREADS"] = "1"
#os.environ["OMP_NUM_THREADS"] = "1"

####
# This is a wrapper for PyBlock. It renames the functions as 
# [subpackage]_[original_function_name]
###

# From pyblock.blocking
def blocking_reblock(data, rowvar=1, ddof=None, weights=None):
    return pyblock.blocking.reblock(data, rowvar, ddof, weights)

def blocking_find_optimal_block(ndata, stats):
    return pyblock.blocking.find_optimal_block(ndata, stats)


# From pyblock error
def error_ratio(stats_A, stats_B, cov_AB, data_len):
    return pyblock.error.ratio(stats_A, stats_B, cov_AB, data_len)

def error_product(stats_A, stats_B, cov_AB, data_len):
    return pyblock.error.product(stats_A, stats_B, cov_AB, data_len)

def error_subtraction(stats_A, stats_B, cov_AB, data_len):
    return pyblock.error.subtraction(stats_A, stats_B, cov_AB, data_len)

def error_addition(stats_A, stats_B, cov_AB, data_len):
    return pyblock.error.addition(stats_A, stats_B, cov_AB, data_len)

def error_pretty_fmt_err(val, err):
    return pyblock.error.pretty_fmt_err(val, err)


# From pyblock pd_utils
def pd_utils_reblock(data, axis=0, weights=None):
    return pyblock.pd_utils.reblock(data, axis, weights)

def pd_utils_optimal_block(block_sub_info):
    return pyblock.pd_utils.reblock(block_sub_info)

def pd_utils_reblock_summary(block_sub_info):
    return pyblock.pd_utils.reblock_summary(block_sub_info)


# From pyblock plot
def plot_plot_reblocking(block_info, plotfile=None, plotshow=True):
    return pyblock.plot.plot_reblocking(block_info, plotfile=plotfile, plotshow=plotshow)


def optimally_reblocked(data):
    #print(data)
    print(data.columns)
    stats = pd.DataFrame(pyblock.pd_utils.reblock(data[["energytotal", "energyee", "energyei", "energyke"]])[1])
    print(stats)
    stats = stats.drop("optimal block", level=1, axis=1).astype(np.float64)
    print(stats)
    pass


def test_pd_utils():
    #data = pd.read_json("./pyqmc/dmcdata.json").sort_index()
    data = pd.read_csv("./data.csv")

    # Test pyblock.blocking
    stats_etot = blocking_reblock(data["energytotal"].values)
    val = blocking_find_optimal_block(len(data["energytotal"].values), stats_etot)
    summary = stats_etot[val[0]]

    # Testing publock.pd_utils
    pd_stats_etot = pd_utils_reblock(data["energytotal"])[1]
    pd_stats_etot = pd_stats_etot["energytotal"]
    optimal_block = pd_stats_etot["optimal block"]
    pd_stats_etot = pd_stats_etot.drop("optimal block", axis=1).astype(np.float64)
    opt_block = pd_utils_optimal_block(pd_stats_etot)
    pd_stats_etot['optimal block'] = optimal_block

    pd_summary = pd_utils_reblock_summary(pd_stats_etot)

    assert pd_summary["mean"].values[0] == summary.mean, "Means are not the same"
    assert pd_summary["standard error"].values[0] == summary.std_err, \
                         "Standard errors are not the same"
    assert pd_summary.index[0] == summary.block, "Different blocks selected as optimal"
    assert pd_summary["standard error error"].values[0] == summary.std_err_err

    # Test pyblock.plot
    #plot_plot_reblocking(pd_stats_etot)


def test_reblock():
    data = pd.read_json("./dmcdata.json").sort_index()
    stats_etot = blocking_reblock(data["energytotal"].values)
    edat = data["energytotal"].values
    i = 0

    pyblock_data = pd.DataFrame(stats_etot)
    reblock_data = pd.DataFrame(columns=["block", "ndata", "mean", "cov", "std_err",
                                         "std_err_err"])
    n = len(edat)
    while(len(edat)>2):
        sem = pd.Series(edat).sem()
        mean = np.mean(edat)
        c = 1/len(edat)*sum((edat[j]-mean)**2 for j in range(len(edat)))
        ser = np.sqrt(c/(len(edat)-1)) * 1/(np.sqrt(2*(len(edat)-1)))
        row = [i, len(edat), mean, np.cov(edat), pd.Series(edat).sem(), ser]
        reblock_data.loc[i+1] = row

        edat_prime = []
        for j in range(1, int(len(edat)/2+1)):
            edat_prime.append((edat[2*j-2] + edat[2*j-1])/2)
        edat = edat_prime
        i += 1

    mean_sum = sum(~(pyblock_data["mean"].values == reblock_data["mean"].values))
    block_sum = sum(~(pyblock_data["block"].values == reblock_data["block"].values))
    ndata_sum = sum(~(pyblock_data["ndata"].values == reblock_data["ndata"].values))
    cov_sum = sum(~(pyblock_data["cov"].values == reblock_data["cov"].values))
    std_err_sum = sum(~(np.isclose(pyblock_data["std_err"].astype(np.float64).values,
                     reblock_data["std_err"].values, rtol=1e-10, atol=1e-13)))
    std_err_err_sum = sum(~(np.isclose(pyblock_data["std_err_err"].astype(np.float64).values,
                     reblock_data["std_err_err"].values, rtol=1e-10, atol=1e-13)))
    total_sum = mean_sum + block_sum + ndata_sum + cov_sum + std_err_sum + std_err_err_sum

    assert total_sum == 0, "Pyblock and Computed Data are Different"


def test_error():
    #data = pd.read_json("./pyqmc/dmcdata.json").sort_index()
    data = pd.read_csv("./data.csv")

    # Test pyblock.error
    stats_etot = blocking_reblock(data["energytotal"].values)
    stats_ke = blocking_reblock(data["energyke"].values)
    cov = data["energytotal"].cov(data["energyke"])
    stats_etot = pd.DataFrame(stats_etot)
    stats_ke = pd.DataFrame(stats_ke)
    stats_etot["standard error"] = stats_etot["std_err"]
    stats_ke["standard error"] = stats_ke["std_err"]

    stats_etot = stats_etot.astype(np.float64)
    stats_ke = stats_ke.astype(np.float64)

    e_rat = error_ratio(stats_etot, stats_ke, cov, len(data["energytotal"].values))
    error_product(stats_etot, stats_ke, cov, len(data["energytotal"].values))
    error_subtraction(stats_etot, stats_ke, cov, len(data["energytotal"].values))
    error_addition(stats_etot, stats_ke, cov, len(data["energytotal"].values))
    #print(error_pretty_fmt_err(stats_etot["mean"].values[0],
    #      stats_etot["standard error"].values[0]))

    assert e_rat == data["energytotal"].sem()

    assert True, "Literally Impossible"



if __name__ == '__main__':
    #test_reblock()
    #test_pd_utils()
    #test_error()
    data = pd.read_json("./dmcdata.json")
    optimally_reblocked(data)
