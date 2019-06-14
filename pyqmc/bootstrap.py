import numpy as np
import pandas as pd

def compute_derivs(df, cols, derivcols, dppsi):
    """
    Given a DataFrame (possibly a resampled one), compute parameter derivatives from expectation values as 
    < dpH dpPsi > - < H > < dpPsi >

    Args:
      df: pandas DataFrame as output by vmc or dmc
      
      cols: list of strings
        The columns to compute derivatives for

      derivcols: list of strings
        The columns that correspond to the parameter derivatives of cols

      dppsi: string
        The column name that stores parameter derivatives of psi (dpPsi/Psi)
    """
    assert len(cols)==len(derivcols)

    #undo_normalizations(df)
    rsmeans = pd.DataFrame(df.mean(axis=0)).T
    #apply_normalizations(rsmeans)

    for col,dcol in zip(cols,derivcols):
        rsmeans[dcol] -= rsmeans[col].values[np.newaxis] *\
                         rsmeans[dppsi].values  
        rsmeans[dcol] = 2*np.real( rsmeans[dcol].values)
    return rsmeans

def bootstrap_derivs(df, nresamples, cols, derivcols, dppsi='dppsi'):
    '''
    Bootstrap to get errors on E, rdm, dE/dp, and dRDM/dp
    df is a pandas DataFrame as loaded by gather_json_df(); should be obdm symmetrized
    '''
    nsamples = len(df[cols[0]])

    resamples = [compute_derivs(df.sample(n=nsamples, replace=True, axis=0), cols, derivcols, dppsi) for rs in range(nresamples)]

    rsdf = pd.concat(resamples, axis=0)
    return rsdf 

