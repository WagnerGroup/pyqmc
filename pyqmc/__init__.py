name = "pyqmc"
from pyqmc.mc import vmc, initial_guess
from pyqmc.slateruhf import PySCFSlaterUHF
from pyqmc.multislater import MultiSlater

from pyqmc.multiplywf import MultiplyWF
from pyqmc.jastrowspin import JastrowSpin
from pyqmc.multiplynwf import MultiplyNWF
from pyqmc.manybody_jastrow import J3

from pyqmc.accumulators import EnergyAccumulator, PGradTransform, LinearTransform
from pyqmc.func3d import (
    PolyPadeFunction,
    PadeFunction,
    GaussianFunction,
    CutoffCuspFunction,
)
from pyqmc.optvariance import optvariance
from pyqmc.optsr import gradient_descent
from pyqmc.linemin import line_minimization
from pyqmc.dmc import rundmc
from pyqmc.cvmc import cvmc_optimize
from pyqmc.reblock import reblock as avg_reblock


def slater_jastrow(mol, mf, abasis=None, bbasis=None):
    if abasis is None:
        abasis = [GaussianFunction(0.8), GaussianFunction(1.6), GaussianFunction(3.2)]
    if bbasis is None:
        bbasis = [
            CutoffCuspFunction(2.0, 1.5),
            GaussianFunction(0.8),
            GaussianFunction(1.6),
            GaussianFunction(3.2),
        ]

    wf = MultiplyWF(
        PySCFSlaterUHF(mol, mf), JastrowSpin(mol, a_basis=abasis, b_basis=bbasis)
    )
    return wf


def gradient_generator(mol, wf, to_opt=None, freeze=None, **ewald_kwargs):
    return PGradTransform(
        EnergyAccumulator(mol, **ewald_kwargs),
        LinearTransform(wf.parameters, to_opt, freeze),
    )


def default_slater(mol, mf, optimize_orbitals=False):
    import numpy as np

    wf = PySCFSlaterUHF(mol, mf)
    if optimize_orbitals:
        to_opt = ["mo_coeff_alpha", "mo_coeff_beta"]
        freeze = {}
        for k in ["mo_coeff_alpha", "mo_coeff_beta"]:
            freeze[k] = np.zeros(wf.parameters[k].shape).astype(bool)
            maxval = np.argmax(np.abs(wf.parameters[k]))
            freeze[k][maxval] = True
    else:
        to_opt = []
        freeze = {}
    return wf, to_opt, freeze

def default_multislater(mol, mf, mc, tol=None):
    import numpy as np

    wf = MultiSlater(mol, mf, mc, tol)
    freeze = {}
    freeze["det_coeff"] = np.zeros(wf.parameters["det_coeff"].shape).astype(bool)
    freeze["det_coeff"][0] = True  # Determinant coefficient pivot
    to_opt = ["det_coeff"]  # Don't have orbital coeff opt on this yet
    return wf, to_opt, freeze


def default_jastrow(mol, ion_cusp=False, rcut=7.5):
    """         
    Default 2-body jastrow from qwalk,
    Args:
      ion_cusp (bool): add an extra term to satisfy electron-ion cusp.
    Returns:
      jastrow, to_opt and freeze
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
        abasis = [CutoffCuspFunction(gamma=24, rcut=rcut)]
    else:
        abasis = []
    abasis += [PolyPadeFunction(beta=beta_abasis[i], rcut=rcut) for i in range(4)]
    bbasis = [CutoffCuspFunction(gamma=24, rcut=rcut)]
    bbasis += [PolyPadeFunction(beta=beta_bbasis[i], rcut=rcut) for i in range(3)]

    jastrow = JastrowSpin(mol, a_basis=abasis, b_basis=bbasis)
    if ion_cusp:
        jastrow.parameters["acoeff"][:, 0, :] = mol.atom_charges()[:, None]
    jastrow.parameters["bcoeff"][0, [0, 1, 2]] = np.array([-0.25, -0.50, -0.25])

    freeze = {}
    freeze["acoeff"] = np.zeros(jastrow.parameters["acoeff"].shape).astype(bool)
    if ion_cusp:
        freeze["acoeff"][:, 0, :] = True  # Cusp conditions
    freeze["bcoeff"] = np.zeros(jastrow.parameters["bcoeff"].shape).astype(bool)
    freeze["bcoeff"][0, [0, 1, 2]] = True  # Cusp conditions
    to_opt = ["acoeff", "bcoeff"]
    return jastrow, to_opt, freeze


def default_msj(mol, mf, mc, tol=None, rcut=7.5):
    wf1, to_opt1, freeze1 = default_multislater(mol, mf, mc, tol)
    wf2, to_opt2, freeze2 = default_jastrow(mol, rcut=rcut)
    wf = MultiplyWF(wf1, wf2)
    to_opt = ["wf1" + x for x in to_opt1] + ["wf2" + x for x in to_opt2]
    freeze = {}
    for k in to_opt1:
        freeze["wf1" + k] = freeze1[k]
    for k in to_opt2:
        freeze["wf2" + k] = freeze2[k]

    return wf, to_opt, freeze


def default_sj(mol, mf, ion_cusp=False):
    wf1, to_opt1, freeze1 = default_slater(mol, mf)
    wf2, to_opt2, freeze2 = default_jastrow(mol, ion_cusp)
    wf = MultiplyWF(wf1, wf2)
    to_opt = ["wf1" + x for x in to_opt1] + ["wf2" + x for x in to_opt2]
    freeze = {}
    for k in to_opt1:
        freeze["wf1" + k] = freeze1[k]
    for k in to_opt2:
        freeze["wf2" + k] = freeze2[k]

    return wf, to_opt, freeze
