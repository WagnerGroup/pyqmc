import pyqmc.sample_many
import numpy as np

def set_wf_params(wfs, params, updater):
    for wf, p, transform in zip(wfs, params, updater.transforms):
        newparms = transform.deserialize(wf, p)
        for k in newparms.keys():
            wf.parameters[k] = newparms[k]


def renormalize(wfs, norms, pivot = 0, N=1):
    """
    Normalizes the last wave function, given a current value of the normalization. Assumes that we want N to be 0.5

    .. math::

    """
    for i, wf in enumerate(wfs):
        if i == pivot:
            continue
        renorm = np.sqrt(norms[pivot]/norms[i]*N)
        if "wf1det_coeff" in wfs[-1].parameters.keys():
            wf.parameters["wf1det_coeff"] *= renorm
        elif "det_coeff" in wfs[-1].parameters.keys():
            wf.parameters["det_coeff"] *= renorm
        else:
            raise NotImplementedError("need wf1det_coeff or det_coeff in parameters")


def optimize_ensemble(
        wfs,
        configs, 
        updater, 
        hdf_file,
        tau = 1,
        max_iterations = 100,
        overlap_penalty = None,
        vmc_kwargs = {},
):
    """Optimize a set of wave functions using ensemble VMC.


    Returns
    -------

    wfs : list of optimized wave functions
    """

    nwf = len(wfs)
    if overlap_penalty is None: 
        overlap_penalty = np.ones((nwf, nwf))*.5

    for i in range(max_iterations):
        data_weighted, data_unweighted, configs = pyqmc.sample_many.sample_overlap(wfs, configs, None, nsteps=10, nblocks=20)
        norm = np.mean(data_unweighted['overlap'], axis=0)
        print("Normalization step", norm.diagonal())
        renormalize(wfs, norm.diagonal(), pivot=1)

        data_weighted, data_unweighted, configs = pyqmc.sample_many.sample_overlap(wfs, configs, updater, nsteps=10, nblocks=20)
        avg, error = updater.block_average(data_weighted, data_unweighted['overlap'])
        print("Iteration", i, "Energy", avg['total'], "Overlap", avg['overlap'])
        dp, report = updater.delta_p([tau], avg, overlap_penalty, verbose=True)
        x = [transform.serialize_parameters(wf.parameters) for wf, transform in zip(wfs, updater.transforms)]
        x = [x_ - dp_[0] for x_, dp_ in zip(x, dp)]
        set_wf_params(wfs, x, updater)


    return wfs