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
                                           'reblocked_data'], index=cols)
    for c in cols:
        reblocked_data.loc[c] = pyblock.pd_utils.reblock_summary(stats[1][c]).squeeze()
        reblocks = pyblock.pd_utils.optimal_block(stats[1][c])
        reblocked_data.at[c, "reblocked_data"] = reblock(data, reblocks, c)
    return reblocked_data


def test_reblocking():
    '''
        Tests reblock function above against PyBlock values
    '''

    cols = ["energytotal", "energyee", "energyei", "energyke"]
    #data = pd.read_csv("./data.csv")
    data = pd.read_json("./dmcdata.json")
    reblocked_data = optimally_reblocked(data, cols)
    for c in cols:
        row = reblocked_data.loc[c]
        assert row["mean"] == np.mean(row["reblocked_data"]), "Means are not equal"
        assert np.isclose(row["standard error"], sem(row["reblocked_data"]), 1e-10, 1e-12),\
                 "Standard errors are not equal"


if __name__ == '__main__':
    test_reblocking()
