import pyblock
import pandas as pd
import numpy as np
from scipy.stats import sem


def reblock(data, reblocks, col):
    '''
        Reblocks data according to “Error estimates on averages of correlated data”,
        H. Flyvbjerg, H.G. Petersen, J. Chem. Phys. 91, 461 (1989).
    '''
    edat = data[col].values
    n = len(edat)
    for i in range(reblocks):
        edat_prime = []
        for j in range(1, int(len(edat)/2+1)):
            edat_prime.append((edat[2*j-2] + edat[2*j-1])/2)
        edat = edat_prime
    return np.array(edat)


def optimally_reblocked(data, cols):
    '''
        Uses pyblock to find optimal reblocking of input data. Takes in pandas
        DataFrame of raw data and selected columns to reblock, returns DataFrame
        of reblocked data.
    '''
    stats = pyblock.pd_utils.reblock(data[cols])
    reblocked_data = pd.DataFrame(columns=['mean', 'standard error', 'standard error error',
                                           'reblocked_data', 'reblocks'], index=cols)
    for c in cols:
        reblocked_data.loc[c] = pyblock.pd_utils.reblock_summary(stats[1][c]).squeeze()
        reblocks = pyblock.pd_utils.optimal_block(stats[1][c])
        reblocked_data.at[c, "reblocks"] = reblocks
        reblocked_data.at[c, "reblocked_data"] = reblock(data, reblocks, c)
    return reblocked_data


def test_reblocking():
    '''
        Tests reblocking against known distribution.
    '''
    def corr_data(N,L):
        '''
            Creates correlated data. Taken from 
            https://pyblock.readthedocs.io/en/latest/tutorial.html.
        '''
        return np.convolve(np.random.randn(2**N), np.ones(2**L)/10, 'same')

    cols = ['test_data1', 'test_data2']
    dat1 = corr_data(11,7)
    dat2 = corr_data(11,4)
    test_data = pd.DataFrame(data={cols[0]:dat1, cols[1]:dat2})
    reblocked_data = optimally_reblocked(test_data, cols)
    for c in cols:
        row = reblocked_data.loc[c]
        std_err = sem(reblock(test_data, reblocked_data.loc[c,"reblocks"], c))
        std_err_err = std_err/np.sqrt(2*(len(reblocked_data.loc[c,"reblocked_data"])-1))

        assert np.isclose(row["mean"], np.mean(test_data[c]), 1e-10, 1e-12), \
               "Means are not equal"
        assert np.isclose(row["standard error"], std_err, 1e-10, 1e-12), \
               "Standard errors are not equal"
        assert np.isclose(row["standard error error"], std_err_err, 1e-10, 1e-12), \
               "Standard error errors are not equal"


if __name__ == '__main__':
    test_reblocking()
