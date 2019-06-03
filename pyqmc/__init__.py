name = "pyqmc"
from pyqmc.mc import vmc,initial_guess
from pyqmc.jastrow import Jastrow2B
from pyqmc.slater import PySCFSlaterRHF
from pyqmc.multiplywf import MultiplyWF
from pyqmc.jastrowspin import JastrowSpin

from pyqmc.accumulators import EnergyAccumulator,PGradAccumulator
from pyqmc.func3d import PadeFunction,GaussianFunction,ExpCuspFunction
from pyqmc.optvariance import optvariance
from pyqmc.optsr import gradient_descent
from pyqmc.dmc import dmc



def slater_jastrow(mol,mf,abasis=None,bbasis=None):
    if abasis is None:
        abasis=[GaussianFunction(0.8),GaussianFunction(1.6),GaussianFunction(3.2)]
    if bbasis is None:
        bbasis=[ExpCuspFunction(2.0,1.5),GaussianFunction(0.8),GaussianFunction(1.6),GaussianFunction(3.2)]

        
    wf=MultiplyWF(PySCFSlaterRHF(mol,mf),
           JastrowSpin(mol,a_basis=abasis,b_basis=bbasis))
    return wf

