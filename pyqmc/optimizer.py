import numpy as np
import pandas as pd
from pyqmc.reblock import optimally_reblock
from pyqmc.bootstrap import bootstrap_derivs
from mc import vmc
 
class WFoptimizer:
    """ Wrapper class for optimizing a wave function using general minimization routines
        Includes functions for converting parameters to a 1d vector
        Note: changes parameters of wf in-place!  
    """
    def __init__(self, wf, configs, pgradtransform, **vmckwargs):
        self.wf=wf
        self.configs=configs
        self.wf.recompute(configs)
        self.pgrad = pgradtransform
        self.vmckwargs = vmckwargs
        self.vmckwargs['accumulators']={'pgrad':pgradtransform}

    def compute(self, params0):
        """Perform VMC with PGradTransform
           This is the function to give to the minimizer
        """
        self.wf.parameters.update(self.pgrad.transform.deserialize(params0).items()) 
        df, configs = vmc(wf, configs, **self.vmckwargs)
        df = pd.DataFrame(df)
        self.configs = configs

        val = df['pgradtotal']
        grad = np.mean(
                  2*np.real(df['pgraddpH']) - val[:,np.newaxis]*2*np.real(df['pgraddppsi'])
                  , axis=0)
        return val, grad

    def compute_err(self, params0):
        """Perform VMC with PGradTransform
           Also return the error and covariance of the gradient, to do Hessian estimation
        """
        self.wf.parameters.update(self.pgrad.transform.deserialize(params0).items()) 
        df, configs = vmc(wf, configs, **self.vmckwargs)
        df = pd.DataFrame(df)
        self.configs = configs

        orbdf = optimally_reblock(df)
        rsdf = bootstrap_derivs(orbdf, ['pgradtotal'],['pgraddpH'],dppsi='pgraddppsi')
        val = rsdf['pgradtotal'].mean()
        err = rsdf['pgradtotal'].std()
        grad = rsdf['pgraddpH'].mean(axis=0)
        gradcov = np.cov(np.stack(rsdf['pgraddpH']).T) 
        return val, err,  grad, gradcov

