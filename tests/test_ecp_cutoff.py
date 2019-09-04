import pandas as pd
from pyqmc.mc import vmc, initial_guess
from pyscf import gto, scf, mcscf
from pyqmc.slateruhf import PySCFSlaterUHF
from pyqmc.jastrowspin import JastrowSpin
from pyqmc.accumulators import EnergyAccumulator
from pyqmc.multiplywf import MultiplyWF
from pyqmc.multislater import MultiSlater
import numpy as np
import time

def test_ecp():

    mol = gto.M(atom=
    '''H 0. 0. 0. 
    H 1 0 0 
    H 2 0 0 
    H 3 0 0''', ecp="bfd", basis="bfd_vtz")
    mf = scf.RHF(mol).run()
    nconf = 10000
    coords = initial_guess(mol, nconf)
    cutoffs = [9,5,4,3,2,1]
    
    label = ['S','J','SJ']
    ind = 0
    for wf in [PySCFSlaterUHF(mol, mf),
               JastrowSpin(mol),
               MultiplyWF(PySCFSlaterUHF(mol,mf),JastrowSpin(mol))]:
      wf.recompute(coords)
      print(label[ind])
      ind += 1
      for cutoff in cutoffs:
          eacc = EnergyAccumulator(mol, cutoff)
          start = time.time()
          eacc(coords, wf)
          end = time.time()
          print('Cutoff=',cutoff, np.around(end - start, 2),'s')
    
    mc = mcscf.CASCI(mf,ncas=4,nelecas=(2,2))
    mc.kernel()

    label = ['MS','MSJ']
    ind = 0
    for wf in [MultiSlater(mol,mf,mc),
               MultiplyWF(MultiSlater(mol,mf,mc),JastrowSpin(mol))]:
      wf.recompute(coords)
      print(label[ind])
      ind += 1
      for cutoff in cutoffs:
          eacc = EnergyAccumulator(mol, cutoff)
          start = time.time()
          eacc(coords, wf)
          end = time.time()
          print('Cutoff=',cutoff, np.around(end - start, 2),'s')

if __name__ == "__main__":
    test_ecp()
