import pandas as pd
import numpy as np


def reblock(df, nblocks, weights=None):
    """
    Reblock df into nblocks new blocks (nblocks is the length of the returned data)

    :param df: data to reblock
    :type df: pandas DataFrame, Series, or numpy array
    :param nblocks: number of resulting blocks
    :type nblocks: int
    :param weights: weights used to average data
    :type weights: pandas Series or numpy array
    :return: reblocked data
    :rtype: same as input df
    """

    if isinstance(df, pd.Series):
        return reblock_series(df, nblocks, weights)
    elif isinstance(df, pd.DataFrame):
        return reblock_dataframe(df, nblocks, weights)
    elif isinstance(df, np.ndarray):
        return reblock_array(df, nblocks, weights)
    else:
        raise TypeError("type {0} not recognized by reblock".format(type(df)))


def reblock_array(df, nblocks, weights=None):
    """
    Reblock df into nblocks new blocks

    :param df: data to reblock
    :type df: numpy array-like
    :param nblocks: number of resulting blocks
    :type nblocks: int
    :param weights: weights used to average data
    :type weights: array
    :return: reblocked data, length nblocks
    :rtype: ndarray
    """
    if weights is None:
        weights = np.ones(len(df))
    return np.stack(_reblock(df, nblocks, weights), axis=0)


def reblock_series(df, nblocks, weights=None):
    """
    Reblock df into nblocks new blocks

    :param df: data to reblock
    :type df: pandas Series
    :param nblocks: number of resulting blocks
    :type nblocks: int
    :param weights: weights used to average data
    :type weights: pandas Series
    :return: reblocked data, length nblocks
    :rtype: pandas Series
    """
    if weights is None:
        weights = np.ones(len(df))
    return pd.Series(_reblock(df.values, nblocks, weights))


def reblock_dataframe(df, nblocks, weights=None):
    """
    Reblock df into nblocks new blocks

    :param df: data to reblock
    :type df: pandas DataFrame
    :param nblocks: number of resulting blocks
    :type nblocks: int
    :param weights: weights used to average data
    :type weights: pandas Series (single column)
    :return: reblocked data, length nblocks
    :rtype: pandas DataFrame
    """
    if weights is None:
        weights = np.ones(len(df))
    rbdf = {col: _reblock(df[col].values, nblocks, weights) for col in df.columns}
    return pd.DataFrame(rbdf)


def _reblock(array, nblocks, weights):
    """
    Helper function to reblock(); this function actually does the reblocking.
    """
    vals = np.array_split(array, nblocks, axis=0)
    weights = np.array_split(weights, nblocks, axis=0)
    # The transposes are to use the broadcast rules of numpy in the case of vector data
    return [(v.T * w.T).T.mean(axis=0) / w.mean(axis=0) for v, w in zip(vals, weights)]


def reblock_summary(df, nblocks=(16, 32, 48, 64), weights=None):
    if hasattr(nblocks, "__iter__"):
        summary_data = [
            _reblock_summary_single(df, nb, weights) for nb in nblocks if nb < len(df)
        ]
    else:
        summary_data = _reblock_summary_single(df, nblocks, weights)
    return pd.DataFrame(summary_data)


def _reblock_summary_single(df, nblocks, weights):
    rbdf = reblock(df, nblocks, weights)
    if hasattr(rbdf, "values") and not hasattr(rbdf, "columns"):
        rbdf = rbdf.values
    serr = rbdf.std(axis=0) / np.sqrt(len(rbdf) - 1)
    return {
        "mean": rbdf.mean(axis=0),
        "standard error": serr,
        "standard error error": serr / np.sqrt(2 * (len(rbdf) - 1)),
        "nblocks": nblocks,
        "nsteps_per_block": len(df) // nblocks,
    }


def optimally_reblocked(data):
    """
    Find optimal reblocking of input data. Takes in pandas
    DataFrame of raw data to reblock, returns DataFrame
    of reblocked data.
    """
    opt = opt_block(data)
    n_reblock = int(np.amax(opt))
    rb_data = reblock_by2(data, n_reblock)
    serr = rb_data.sem(axis=0)
    d = {
        "mean": rb_data.mean(axis=0),
        "standard error": serr,
        "standard error error": serr / np.sqrt(2 * (len(rb_data) - 1)),
        "reblocks": n_reblock,
    }
    return pd.DataFrame(d)


def reblock_by2(df, ntimes, c=None):
    """
    Reblocks data according to “Error estimates on averages of correlated data”,
    H. Flyvbjerg, H.G. Petersen, J. Chem. Phys. 91, 461 (1989).
    """
    newdf = df.copy()
    if c is not None:
        newdf = newdf[c]
    for i in range(ntimes):
        m = newdf.shape[0]
        lasteven = m - int(m % 2 == 1)
        newdf = (newdf[:lasteven:2] + newdf[1::2].values) / 2
    return newdf


def opt_block(df):
    """
    Finds optimal block size for each variable in a dataset
    df is a dataframe where each row is a sample and each column is a calculated quantity
    reblock each column over samples to find the best block size
    Returns optimal_block, a 1D array with the optimal size for each column in df
    """
    newdf = df.copy()
    iblock = 0
    ndata, nvariables = tuple(df.shape[:2])
    optimal_block = np.array([float("NaN")] * nvariables)
    serr0 = df.sem(axis=0).values
    statslist = []
    while newdf.shape[0] > 1:
        serr = newdf.sem(axis=0).values
        serrerr = serr / (2 * (newdf.shape[0] - 1)) ** 0.5
        statslist.append((iblock, serr.copy()))

        n = newdf.shape[0]
        lasteven = n - int(n % 2 == 1)
        newdf = (newdf[:lasteven:2] + newdf[1::2].values) / 2
        iblock += 1
    for iblock, serr in reversed(statslist):
        B3 = 2 ** (3 * iblock)
        inds = np.where(B3 >= 2 * ndata * (serr / serr0) ** 4)[0]
        optimal_block[inds] = iblock

    return optimal_block


def test_reblocking():
    """
    Tests reblocking against known distribution.
    """
    import scipy.stats

    def corr_data(N, L):
        """
        Creates correlated data. Taken from
        https://pyblock.readthedocs.io/en/latest/tutorial.html.
        """
        return np.convolve(np.random.randn(2**N), np.ones(2**L) / 10, "same")

    n = 11
    cols = ["test_data1", "test_data2"]
    dat1 = corr_data(n, 4)
    dat2 = corr_data(n, 7)
    test_data = pd.DataFrame(data={cols[0]: dat1, cols[1]: dat2})
    reblocked_data = optimally_reblocked(test_data[cols])
    for c in cols:
        row = reblocked_data.loc[c]
        reblocks = reblocked_data["reblocks"].values[0]
        std_err = scipy.stats.sem(reblock_by2(test_data, reblocks, c))
        std_err_err = std_err / np.sqrt(2 * (2 ** (n - reblocks) - 1))

        assert np.isclose(
            row["mean"], np.mean(test_data[c]), 1e-10, 1e-12
        ), "Means are not equal"
        assert np.isclose(
            row["standard error"], std_err, 1e-10, 1e-12
        ), "Standard errors are not equal"
        assert np.isclose(
            row["standard error error"], std_err_err, 1e-10, 1e-12
        ), "Standard error errors are not equal"

    statlist = ["mean", "sem", lambda x: x.sem() / np.sqrt(2 * (len(x) - 1))]
    rb1 = reblock(test_data, len(test_data) // 4).agg(statlist).T
    rb2 = reblock_by2(test_data, 2).agg(statlist).T
    for c in rb1.columns:
        assert np.isclose(rb1[c], rb2[c], 1e-10, 1e-12).all(), (c, rb1[c], rb2[c])


if __name__ == "__main__":
    test_reblocking()
