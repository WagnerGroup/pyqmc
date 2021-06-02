import numpy as np
import scipy.optimize
import pyqmc.energy


def optvariance(energy, wf, coords, params=None, **kwargs):
    """Optimizes variance of wave function against parameters indicated by params.

    Does not use gradient information, and assumes that only the kinetic energy changes.

    Args:
      energy: An Accumulator object that returns total energy in 'total' and kinetic energy in 'ke'

      coords: (nconfig,nelec,3)

      params: list of dictionary entries in wf.parameters to optimize

      kwargs: options for scipy.minimize

    Returns:
      opt_variance, modifying params into optimized values.

    """
    if params is None:
        params = list(wf.parameters.keys())

    # scipy.minimize() needs 1d argument
    x0 = np.concatenate([wf.parameters[k].flatten() for k in params])
    shapes = np.array([wf.parameters[k].shape for k in params])
    slices = np.array([np.prod(s) for s in shapes])
    Enref = energy(coords, wf)

    def variance_cost_function(x):
        x_sliced = np.split(x, slices[:-1])
        for i, k in enumerate(params):
            wf.parameters[k] = x_sliced[i].reshape(wf.parameters[k].shape)
        wf.recompute(coords)
        ke = pyqmc.energy.kinetic(coords, wf)
        # Here we assume the ecp is fixed and only recompute
        # kinetic energy
        En = Enref["total"] - Enref["ke"] + ke
        return np.std(En) ** 2

    def callback(xk):
        print(xk, variance_cost_function(xk))
        return False

    res = scipy.optimize.minimize(
        variance_cost_function, x0=x0, callback=callback, **kwargs
    )

    opt_pars = np.split(res.x, slices[:-1])
    for i, k in enumerate(params):
        wf.parameters[k] = opt_pars[i].reshape(shapes[i])

    return res.fun, wf
