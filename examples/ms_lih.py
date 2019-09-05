import os

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np
import pandas as pd
from pyscf import lib, gto, scf, mcscf
import pyqmc.testwf as testwf
from pyqmc.mc import vmc, initial_guess
from pyqmc.accumulators import EnergyAccumulator
from pyqmc.multislater import MultiSlater
from pyqmc.coord import OpenConfigs
import pyqmc

mol = gto.M(atom="Li 0. 0. 0.; H 0. 0. 1.5", basis="cc-pvtz", unit="bohr", spin=0)
mf = scf.RHF(mol).run()
mc = mcscf.CASCI(mf,ncas=4,nelecas=(1,1))
mc.kernel()
wf = MultiSlater(mol, mf, mc)

#Quick VMC test
nconf = 5000
nsteps = 100 
warmup = 30
coords = initial_guess(mol, nconf)
df, coords = vmc(
    wf, coords, nsteps=nsteps, accumulators={"energy": EnergyAccumulator(mol)}
)   

df = pd.DataFrame(df)
en = np.mean(df["energytotal"][warmup:])
err = np.std(df["energytotal"][warmup:]) / np.sqrt(nsteps - warmup)
assert en - mc.e_tot < 10 * err 
