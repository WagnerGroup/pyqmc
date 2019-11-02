import pandas as pd
import numpy as np


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


def _reblock(array, nblocks):
    vals = np.array_split(array, nblocks, axis=0)
    return [v.mean(axis=0) for v in vals]


def reblock(df, nblocks):
    size, nbig = np.divmod(len(df), nblocks)
    if isinstance(df, pd.Series):
        return pd.Series(_reblock(df.values, nblocks))
    elif isinstance(df, pd.DataFrame):
        rbdf = {col: _reblock(df[col].values, nblocks) for col in df.columns}
        return pd.DataFrame(rbdf)
    elif isinstance(df, np.ndarray):
        return _reblock(df, nblocks)
    else:
        print("WARNING: can't reblock data of type", type(df), "-- not reblocking.")
        return df


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


def reblock_summary(df, nblocks):
    df = reblock(df, nblocks)
    serr = df.sem()
    d = {
        "mean": df.mean(axis=0),
        "standard error": serr,
        "standard error error": serr / np.sqrt(2 * (len(df) - 1)),
        "n_blocks": nblocks,
    }
    return pd.DataFrame(d)


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
    from scipy.stats import sem

    def corr_data(N, L):
        """
            Creates correlated data. Taken from 
            https://pyblock.readthedocs.io/en/latest/tutorial.html.
        """
        return np.convolve(np.random.randn(2 ** N), np.ones(2 ** L) / 10, "same")

    n = 11
    cols = ["test_data1", "test_data2"]
    dat1 = corr_data(n, 4)
    dat2 = corr_data(n, 7)
    test_data = pd.DataFrame(data={cols[0]: dat1, cols[1]: dat2})
    reblocked_data = optimally_reblocked(test_data[cols])
    for c in cols:
        row = reblocked_data.loc[c]
        reblocks = reblocked_data["reblocks"].values[0]
        std_err = sem(reblock_by2(test_data, reblocks, c))
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
