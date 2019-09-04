import pandas as pd
from pyqmc.mc import vmc, initial_guess
from pyscf import gto, scf
from pyqmc.slateruhf import PySCFSlaterUHF
from pyqmc.accumulators import EnergyAccumulator
import numpy as np
import time

def test_ecp():

    mol = gto.M(atom=
    '''H 0. 0. 0. 
    H 1 0 0 
    H 2 0 0 
    H 3 0 0''', ecp="bfd", basis="bfd_vtz")
    mf = scf.RHF(mol).run()
    nconf = 1000
    wf = PySCFSlaterUHF(mol, mf)
    
    coords = initial_guess(mol, nconf)
    warmup = 30
    cutoffs = [10,5,2,1,0.75,0.5]
    
    for cutoff in cutoffs:
        start = time.time()
        df, coords = vmc(
            wf, coords, nsteps=100, accumulators={"energy": EnergyAccumulator(mol,cutoff)}
        )
        df = pd.DataFrame(df)
        end = time.time()
        t = end - start
        
        e = np.mean(df["energytotal"][warmup:])
        err = np.std(df["energytotal"][warmup:])/np.sqrt(70)
        assert int(abs(e - mf.energy_tot())/err) < 5
        print("Runtime cutoff "+str(cutoff)+" :",t)

if __name__ == "__main__":
    test_ecp()
