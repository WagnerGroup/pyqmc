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
    mol = gto.M(
        atom="""C 0 0 0 
       C 1 0 0 
    """,
        ecp="bfd",
        basis="bfd_vtz",
    )
    mf = scf.RHF(mol).run()
    nconf = 1000
    coords = initial_guess(mol, nconf)
    thresholds = [1e15, 100, 50, 20, 10, 5, 1]
    label = ["S", "J", "SJ"]
    ind = 0
    for wf in [
        PySCFSlaterUHF(mol, mf),
        JastrowSpin(mol),
        MultiplyWF(PySCFSlaterUHF(mol, mf), JastrowSpin(mol)),
    ]:
        wf.recompute(coords)
        print(label[ind])
        ind += 1
        for threshold in thresholds:
            eacc = EnergyAccumulator(mol, threshold)
            start = time.time()
            eacc(coords, wf)
            end = time.time()
            print("Threshold=", threshold, np.around(end - start, 2), "s")
    mc = mcscf.CASCI(mf, ncas=4, nelecas=(2, 2))
    mc.kernel()

    label = ["MS"]
    ind = 0
    for wf in [MultiSlater(mol, mf, mc)]:
        wf.recompute(coords)
        print(label[ind])
        ind += 1
        for threshold in thresholds:
            eacc = EnergyAccumulator(mol, threshold)
            start = time.time()
            eacc(coords, wf)
            end = time.time()
            print("Threshold=", threshold, np.around(end - start, 2), "s")


if __name__ == "__main__":
    test_ecp()
