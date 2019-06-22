import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from pyscf import gto, scf, lo
from numpy.linalg import solve
from pyqmc import PySCFSlaterUHF
from pyqmc.mc import initial_guess, vmc
from pyqmc.accumulators import EnergyAccumulator
from pandas import DataFrame
from pyqmc.obdm import OBDMAccumulator,normalize_obdm

def test():

    mol = gto.M(
        atom="Li 0. 0. 0.; Li 0. 0. 1.5", basis="sto-3g", unit="bohr", verbose=0
    )
    mf = scf.RHF(mol).run()

    # Lowdin orthogonalized AO basis.
    lowdin = lo.orth_ao(mol, "lowdin")

    # MOs in the Lowdin basis.
    mo = solve(lowdin, mf.mo_coeff)

    # make AO to localized orbital coefficients.
    mfobdm = mf.make_rdm1(mo, mf.mo_occ)

    ### Test OBDM calculation.
    nconf = 500
    nsteps = 400
    obdm_steps = 4
    warmup = 15
    wf = PySCFSlaterUHF(mol, mf)
    configs = initial_guess(mol, nconf)
    energy = EnergyAccumulator(mol)
    obdm = OBDMAccumulator(mol=mol, orb_coeff=lowdin, nstep=obdm_steps)
    obdm_up = OBDMAccumulator(mol=mol, orb_coeff=lowdin, nstep=obdm_steps,spin=0)
    obdm_down = OBDMAccumulator(mol=mol, orb_coeff=lowdin, nstep=obdm_steps,spin=1)
    
    df, coords = vmc(
        wf, configs, nsteps=nsteps, accumulators={"energy": energy, "obdm": obdm,
            "obdm_up": obdm_up, "obdm_down": obdm_down }
    )
    df = DataFrame(df)

    obdm_est={}
    for k in ['obdm','obdm_up','obdm_down']:
        avg_norm = np.array(df.loc[warmup:, k+"norm"].values.tolist()).mean(axis=0)
        avg_obdm = np.array(df.loc[warmup:, k+"value"].values.tolist()).mean(axis=0)
        obdm_est[k]=normalize_obdm(avg_obdm, avg_norm)
    
    print("Average OBDM(orb,orb)", obdm_est['obdm'].diagonal().round(3))
    print("mf obdm",mfobdm.diagonal().round(3))
    assert np.max(np.abs(obdm_est['obdm']-mfobdm)) < 0.05
    print(obdm_est['obdm_up'].diagonal().round(3))
    print(obdm_est['obdm_down'].diagonal().round(3))
    assert np.max(np.abs(obdm_est['obdm_up']+obdm_est['obdm_down']-mfobdm)) < 0.05

if __name__=="__main__":
    test()
