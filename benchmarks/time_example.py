import numpy as np
import os
import time
import pyqmc.api as pyq


def _compile(chkfile):
    t0 = time.perf_counter()
    mol, mf = pyq.recover_pyscf(chkfile)
    wf, _ = pyq.generate_slater(mol, mf)
    configs = pyq.initial_guess(mol, 10)
    wf.recompute(configs)
    e = 0
    wf.gradient(e, configs.electron(e))
    en = pyq.EnergyAccumulator(mol)
    en(configs, wf)
    t1 = time.perf_counter()
    print("compile time", t1-t0)

def run(chkfile):
    with open("../.git/HEAD", "r") as f:
        print(f.read().strip())
    mol, mf = pyq.recover_pyscf(chkfile)
    if "diamond_primitive" in chkfile:
        S = np.ones((3, 3)) - 2*np.eye(3)
        mol = pyq.get_supercell(mol, S)
    nelec = sum(mol.nelec)
    nconfig = int(1e6 / nelec**2.3) # try to guess a good number of benchmark configs
    wf, _ = pyq.generate_slater(mol, mf, eval_gto_precision=1e-4)
    configs = pyq.initial_guess(mol, nconfig)
    en = pyq.EnergyAccumulator(mol)

    t0 = time.perf_counter()
    pyq.vmc(wf, configs, nsteps=1, accumulators=dict(energy=en))

    t1 = time.perf_counter()

    chkname = chkfile.split("/")[-1]
    d = dict(chkfile=chkname, time=t1-t0, nconfig=nconfig)
    print(d)
    return d
    

if __name__ == "__main__":
    import sys
    import pandas as pd

    for chkfile in sys.argv[1:]:
        _compile(chkfile)
        run(chkfile)

    #d = []
    #for chkfile in sys.argv[1:]:
    #    _compile(chkfile)
    #    for i in range(8):
    #        d.append(run(chkfile, int(5*2**i)))
    #df = pd.DataFrame(d)
    #print(df)
