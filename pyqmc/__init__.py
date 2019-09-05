name = "pyqmc"
from pyqmc.mc import vmc, initial_guess
from pyqmc.slateruhf import PySCFSlaterUHF
from pyqmc.multislater import MultiSlater
from pyqmc.multiplywf import MultiplyWF
from pyqmc.jastrowspin import JastrowSpin

from pyqmc.accumulators import EnergyAccumulator, PGradTransform, LinearTransform
from pyqmc.func3d import PadeFunction, GaussianFunction, ExpCuspFunction
from pyqmc.optvariance import optvariance
from pyqmc.optsr import gradient_descent
from pyqmc.linemin import line_minimization
from pyqmc.dmc import rundmc


def slater_jastrow(mol, mf, abasis=None, bbasis=None):
    if abasis is None:
        abasis = [GaussianFunction(0.8), GaussianFunction(1.6), GaussianFunction(3.2)]
    if bbasis is None:
        bbasis = [
            ExpCuspFunction(2.0, 1.5),
            GaussianFunction(0.8),
            GaussianFunction(1.6),
            GaussianFunction(3.2),
        ]

    wf = MultiplyWF(
        PySCFSlaterUHF(mol, mf), JastrowSpin(mol, a_basis=abasis, b_basis=bbasis)
    )
    return wf


def gradient_generator(mol, wf, to_opt=None, freeze=None):
    return PGradTransform(
        EnergyAccumulator(mol), LinearTransform(wf.parameters, to_opt, freeze)
    )


def default_multislater(mol, mf, mc):
    import numpy as np

    wf = MultiSlater(mol, mf, mc)
    freeze = {}
    freeze["det_coeff"] = np.zeros(wf.parameters["det_coeff"].shape).astype(bool)
    freeze["det_coeff"][0] = True  # Determinant coefficient pivot
    to_opt = ["det_coeff"]  # Don't have orbital coeff opt on this yet
    return wf, to_opt, freeze


def default_jastrow(mol):
    import numpy as np

    abasis = [GaussianFunction(0.8), GaussianFunction(1.6), GaussianFunction(3.2)]
    bbasis = [
        ExpCuspFunction(2.0, 1.5),
        GaussianFunction(0.8),
        GaussianFunction(1.6),
        GaussianFunction(3.2),
    ]
    wf = JastrowSpin(mol, a_basis=abasis, b_basis=bbasis)
    wf.parameters["bcoeff"][0, [0, 1, 2]] = np.array([-0.25, -0.50, -0.25])
    freeze = {}
    freeze["acoeff"] = np.zeros(wf.parameters["acoeff"].shape).astype(bool)
    freeze["bcoeff"] = np.zeros(wf.parameters["bcoeff"].shape).astype(bool)
    freeze["bcoeff"][0, [0, 1, 2]] = True  # Cusp conditions
    to_opt = ["acoeff", "bcoeff"]
    return wf, to_opt, freeze


def default_msj(mol, mf, mc):
    wf1, to_opt1, freeze1 = default_multislater(mol, mf, mc)
    wf2, to_opt2, freeze2 = default_jastrow(mol)
    wf = MultiplyWF(wf1, wf2)
    to_opt = ["wf1" + x for x in to_opt1] + ["wf2" + x for x in to_opt2]
    freeze = {}
    for k in to_opt1:
        freeze["wf1" + k] = freeze1[k]
    for k in to_opt2:
        freeze["wf2" + k] = freeze2[k]

    return wf, to_opt, freeze
