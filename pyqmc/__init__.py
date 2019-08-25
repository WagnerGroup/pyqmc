name = "pyqmc"
from pyqmc.mc import vmc, initial_guess
from pyqmc.slateruhf import PySCFSlaterUHF

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

def default_constraints(wf,det_pivot=0):
    import numpy as np 
    freeze = {}
    if('det_coeff' in wf.parameters.keys()): 
        freeze['det_coeff'] = np.zeros(wf.parameters['det_coeff'].shape).astype(bool)
        freeze['det_coeff'][det_pivot] = True #Determinant coefficient pivot 

    return freeze 

from pyscf import lib, gto, scf, mcscf
import pyqmc.testwf as testwf
from pyqmc.mc import vmc, initial_guess
from pyqmc.accumulators import EnergyAccumulator
from pyqmc.multislater import MultiSlater
mol = gto.M(atom="Li 0. 0. 0.; H 0. 0. 1.5", basis="cc-pvtz", unit="bohr", spin=0)
mf = scf.RHF(mol).run()
mc = mcscf.CASCI(mf,ncas=2,nelecas=(1,1))
mc.kernel()
wf = MultiSlater(mol, mf, mc)
f = default_constraints(wf)
print(f)
