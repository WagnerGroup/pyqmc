name = "pyqmc"
from pyqmc.mc import vmc, initial_guess
from pyqmc.slater import PySCFSlater
from pyqmc.multislater import MultiSlater

from pyqmc.multiplywf import MultiplyWF
from pyqmc.jastrowspin import JastrowSpin
from pyqmc.manybody_jastrow import J3

from pyqmc.accumulators import EnergyAccumulator, PGradTransform, LinearTransform
from pyqmc.func3d import (
    PolyPadeFunction,
    PadeFunction,
    GaussianFunction,
    CutoffCuspFunction,
)
from pyqmc.optvariance import optvariance
from pyqmc.linemin import line_minimization
from pyqmc.dmc import rundmc
from pyqmc.cvmc import cvmc_optimize
from pyqmc.reblock import reblock as avg_reblock


def slater_jastrow(mol, mf, abasis=None, bbasis=None):
    raise NotImplementedError(
        "slater_jastrow() is no longer supported. Please use default_sj instead."
    )


def gradient_generator(mol, wf, to_opt=None, **ewald_kwargs):
    return PGradTransform(
        EnergyAccumulator(mol, **ewald_kwargs), LinearTransform(wf.parameters, to_opt)
    )


def default_slater(mol, mf, optimize_orbitals=False):
    import numpy as np

    wf = PySCFSlater(mol, mf)
    to_opt = {}
    if optimize_orbitals:
        for k in ["mo_coeff_alpha", "mo_coeff_beta"]:
            to_opt[k] = np.ones(wf.parameters[k].shape).astype(bool)
            # maxval = np.argmax(np.abs(wf.parameters[k]))
            # print(maxval)
            # to_opt[k][maxval] = False
    return wf, to_opt


def default_multislater(mol, mf, mc, tol=None, freeze_orb=None):
    import numpy as np

    # Nothing provided, nothing frozen
    if freeze_orb is None:
        freeze_orb = [[], []]

    wf = MultiSlater(mol, mf, mc, tol, freeze_orb)
    to_opt = ["det_coeff"]
    to_opt = {"det_coeff": np.ones(wf.parameters["det_coeff"].shape).astype(bool)}
    to_opt["det_coeff"][0] = False  # Determinant coefficient pivot

    for s, k in enumerate(["mo_coeff_alpha", "mo_coeff_beta"]):
        to_freeze = np.zeros(wf.parameters[k].shape, dtype=bool)
        to_freeze[:, freeze_orb[s]] = True
        if to_freeze.sum() < np.prod(to_freeze.shape):
            to_opt[k] = ~to_freeze

    return wf, to_opt


def default_jastrow(mol, ion_cusp=False):
    """         
    Default 2-body jastrow from qwalk,
    Args:
      ion_cusp (bool): add an extra term to satisfy electron-ion cusp.
    Returns:
      jastrow, to_opt
    """
    import numpy as np

    def expand_beta_qwalk(beta0, n):
        """polypade expansion coefficients 
        for n basis functions with first 
        coeff beta0"""
        beta = np.zeros(n)
        beta[0] = beta0
        beta1 = np.log(beta0 + 1.00001)
        for i in range(1, n):
            beta[i] = np.exp(beta1 + 1.6 * i) - 1
        return beta

    beta_abasis = expand_beta_qwalk(0.2, 4)
    beta_bbasis = expand_beta_qwalk(0.5, 3)
    if ion_cusp:
        abasis = [CutoffCuspFunction(gamma=24, rcut=7.5)]
    else:
        abasis = []
    abasis += [PolyPadeFunction(beta=beta_abasis[i], rcut=7.5) for i in range(4)]
    bbasis = [CutoffCuspFunction(gamma=24, rcut=7.5)]
    bbasis += [PolyPadeFunction(beta=beta_bbasis[i], rcut=7.5) for i in range(3)]

    jastrow = JastrowSpin(mol, a_basis=abasis, b_basis=bbasis)
    if ion_cusp:
        jastrow.parameters["acoeff"][:, 0, :] = mol.atom_charges()[:, None]
    jastrow.parameters["bcoeff"][0, [0, 1, 2]] = np.array([-0.25, -0.50, -0.25])

    to_opt = {}
    to_opt["acoeff"] = np.ones(jastrow.parameters["acoeff"].shape).astype(bool)
    if ion_cusp:
        to_opt["acoeff"][:, 0, :] = False  # Cusp conditions
    to_opt["bcoeff"] = np.ones(jastrow.parameters["bcoeff"].shape).astype(bool)
    to_opt["bcoeff"][0, [0, 1, 2]] = False  # Cusp conditions
    return jastrow, to_opt


def default_msj(mol, mf, mc, tol=None, freeze_orb=None, ion_cusp=False):
    wf1, to_opt1 = default_multislater(mol, mf, mc, tol, freeze_orb)
    wf2, to_opt2 = default_jastrow(mol, ion_cusp)
    wf = MultiplyWF(wf1, wf2)
    to_opt = {"wf1" + x: opt for x, opt in to_opt1.items()}
    to_opt.update({"wf2" + x: opt for x, opt in to_opt2.items()})

    return wf, to_opt


def default_sj(mol, mf, ion_cusp=False):
    wf1, to_opt1 = default_slater(mol, mf)
    wf2, to_opt2 = default_jastrow(mol, ion_cusp)
    wf = MultiplyWF(wf1, wf2)
    to_opt = {"wf1" + x: opt for x, opt in to_opt1.items()}
    to_opt.update({"wf2" + x: opt for x, opt in to_opt2.items()})

    return wf, to_opt
