import pandas as pd
import numpy as np

def optimally_reblocked(data):
    """
        Find optimal reblocking of input data. Takes in pandas
        DataFrame of raw data to reblock, returns DataFrame
        of reblocked data.
    """
    opt = opt_block(data)
    reblocked_data = reblock(data, np.amax(opt))
    return reblocked_data

def optimally_reblocked_pyblock(data):  # , cols):
    import pyblock
    """
        Uses pyblock to find optimal reblocking of input data. Takes in pandas
        DataFrame of raw data and selected columns to reblock, returns DataFrame
        of reblocked data.
    """
    stats = pyblock.pd_utils.reblock(data)
    reblocked_data = pyblock.pd_utils.reblock_summary(stats[1]).squeeze()
    reblocks = pyblock.pd_utils.optimal_block(stats[1])
    reblocked_data["reblocks"] = reblocks
    return reblocked_data

def reblock(df, n, c): 
    start = time.time()
    newdf = df.copy()
    if c is not None:
        newdf = newdf[c]
    for i in range(n):
        m = newdf.shape[0]
        lasteven = m - int(m%2==1)
        newdf = (newdf[:lasteven:2]+newdf[1::2].values)/2
    return newdf

def opt_block(df):
    """
    Finds optimal block size for each variable in a dataset
    df is a dataframe where each row is a sample and each column is a calculated quantity
    reblock each column over samples to find the best block size
    Returns optimal_block, a 1D array with the optimal size for each column in df
    """
    start = time.time()
    newdf = df.copy()
    iblock = 0
    ndata, nvariables = tuple(df.shape[:2])
    optimal_block = np.array([float('NaN')]*nvariables)
    serr0 = df.sem(axis=0).values
    print('serr0.shape',serr0.shape, time.time()-start)
    print('ndata',ndata, time.time()-start)
    statslist = []
    while(newdf.shape[0]>1):
        serr = newdf.sem(axis=0).values
        serrerr = serr/(2*(newdf.shape[0]-1))**.5
        statslist.append((iblock, serr.copy()))

        n = newdf.shape[0]
        print(iblock, n, time.time()-start)
        lasteven = n - int(n%2==1)
        newdf = (newdf[:lasteven:2]+newdf[1::2].values)/2
        iblock += 1
    for iblock, serr in reversed(statslist):
        B3 = 2**(3*iblock)
        inds = np.where(B3 >= 2*ndata*(serr/serr0)**4)[0]
        optimal_block[inds] = iblock
        print(iblock, len(inds), time.time()-start)

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

    def reblock(data, reblocks, col):
        """
            Reblocks data according to “Error estimates on averages of correlated data”,
            H. Flyvbjerg, H.G. Petersen, J. Chem. Phys. 91, 461 (1989).
        """
        edat = data[col].values
        n = len(edat)
        for i in range(reblocks):
            edat_prime = []
            for j in range(1, int(len(edat) / 2 + 1)):
                edat_prime.append((edat[2 * j - 2] + edat[2 * j - 1]) / 2)
            edat = edat_prime
        return np.array(edat)

    n = 11
    cols = ["test_data1", "test_data2"]
    dat1 = corr_data(n, 4)
    dat2 = corr_data(n, 7)
    test_data = pd.DataFrame(data={cols[0]: dat1, cols[1]: dat2})
    reblocked_data = optimally_reblocked(test_data[cols])
    for c in cols:
        row = reblocked_data.loc[c]
        reblocks = reblocked_data["reblocks"].values[0]
        std_err = sem(reblock(test_data, reblocks, c))
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


if __name__ == "__main__":
    test_reblocking()
