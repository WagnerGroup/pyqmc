import numpy as np
from pyscf import lib, gto, scf
import pyqmc
from pyqmc.slateruhf import PySCFSlaterUHF
from pyqmc.jastrowspin import JastrowSpin
# from pyqmc.jastrow import Jastrow2B
from pyqmc.coord import OpenConfigs
import time
from pyqmc.manybody_jastrow import J3
import pyqmc.testwf as test
from pyqmc.wf import WaveFunction
from pyqmc.multiplywf import MultiplyWF
# import pandas as pd

mol = gto.M(atom="Li 0. 0. 0.; Li 0. 0. 1.5", basis="sto-3g", unit="bohr")
mf = scf.UHF(mol).run()
wf1 = PySCFSlaterUHF(mol, mf)
wf2 = JastrowSpin(mol)
# wf3 = J3(mol)
wf = WaveFunction([wf1, wf2])
wfmultiply = MultiplyWF(wf1, wf2)
configs = OpenConfigs(np.random.randn(10, np.sum(mol.nelec), 3))
wf.recompute(configs)
# res = test.test_wf_gradient(wf, configs)
e=2
epos = configs.electron(e)
test.test_mask(wf, 3, epos)
