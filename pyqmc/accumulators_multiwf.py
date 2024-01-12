import numpy as np
import h5py



def invert_list_of_dicts(A, asarray=True):
    """
    if we have a list [ {'A':1,'B':2}, {'A':3, 'B':5}], invert the structure to
    {'A':[1,3], 'B':[2,5]}.
    If not all keys are present in all lists, error.
    """
    if asarray:
        return {k: np.asarray([a[k] for a in A]) for k in A[0].keys()}
    else:
        return {k: [a[k] for a in A] for k in A[0].keys()}

class EnergyAccumulatorMultipleWF:
    """ 
    """

    def __init__(self, enacc):
        """
        """
        self.enacc = enacc

    def avg(self, configs, wfs:list, weights:np.ndarray):
        """
        weights: [nwf, nwf, configs]
        wfs: [nwf]
        configs: PeriodicConfigs or OpenConfigs object
        Returns: 

        """
        energies = invert_list_of_dicts([self.enacc(configs, wf) for wf in wfs]).T
        weighted_dat = {}
        nconfig = configs.configs.shape[0]
        for k, en in energies.items():
            weighted_dat[k] = np.einsum("jc,ijc->ij", en, weights) / nconfig

        return weighted_dat

    def keys(self):
        return self.enacc.keys()

    def shapes(self):
        """
        Note that the shapes here do not include the number of wave functions.
        """
        return self.enacc.shapes()




