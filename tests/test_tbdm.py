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
from pyqmc.tbdm import TBDMAccumulator, normalize_tbdm

def gen_slater_tbdm(obdm):
  ''' Generates the TBDM from a OBDM in the special case of a Slater determinant
  wave function.  '''
  norb=obdm.shape[1]
  tbdm=np.zeros([2,2,norb,norb,norb,norb])+np.nan

  # Compute TBDM.
  for spin in 0,1:
    tbdm[spin,spin]=\
        np.einsum('ik,jl->ijkl',obdm[spin],obdm[spin]) -\
        np.einsum('il,jk->ijkl',obdm[spin],obdm[spin])
  tbdm[0,1]=np.einsum('ik,jl->ijkl',obdm[0],obdm[1])
  tbdm[1,0]=np.einsum('ik,jl->ijkl',obdm[1],obdm[0])

  return tbdm

def test():
    mol = gto.M(
        atom="Li 0. 0. 0.; Li 0. 0. 1.5", basis="sto-3g", unit="bohr", verbose=0
    )
    mf = scf.RHF(mol).run()
    lowdin = lo.orth_ao(mol, "lowdin")
    mo = solve(lowdin, mf.mo_coeff)

    # make AO to localized orbital coefficients.
    norb = 1
    mfobdm = mf.make_rdm1(mo, mf.mo_occ)[:norb][:,:norb]
    print(mfobdm)
    mftbdm = gen_slater_tbdm(np.array([mfobdm/2, mfobdm/2]))
    mftbdm_tot = mftbdm[0,0] + mftbdm[0,1] +\
                 mftbdm[1,0] + mftbdm[1,1]

    ### Test TBDM calculation.
    nconf = 500
    nsteps = 400
    tbdm_steps = 4
    warmup = 15
    wf = PySCFSlaterUHF(mol, mf)
    configs = initial_guess(mol, nconf)
    energy = EnergyAccumulator(mol)
      
    tbdm = TBDMAccumulator(mol=mol, orb_coeff=lowdin[:,:norb], nstep=tbdm_steps)
    tbdm_up_up = TBDMAccumulator(mol=mol, orb_coeff=lowdin[:,:norb], nstep=tbdm_steps, spin=0)
    tbdm_up_dn = TBDMAccumulator(mol=mol, orb_coeff=lowdin[:,:norb], nstep=tbdm_steps, spin=1)
    tbdm_dn_up = TBDMAccumulator(mol=mol, orb_coeff=lowdin[:,:norb], nstep=tbdm_steps, spin=2)
    tbdm_dn_dn = TBDMAccumulator(mol=mol, orb_coeff=lowdin[:,:norb], nstep=tbdm_steps, spin=3)
      
    df, coords = vmc(
        wf,
        configs,
        nsteps=nsteps,
        accumulators={
            "tbdm": tbdm,
            "tbdm_up_up": tbdm_up_up,
            "tbdm_up_dn": tbdm_up_dn,
            "tbdm_dn_up": tbdm_dn_up,
            "tbdm_dn_dn": tbdm_dn_dn,
        },
    )

    df = DataFrame(df)
    tbdm_est = {}
    for k in ["tbdm_up_dn","tbdm_dn_up"]:
        avg_norm = np.array(df.loc[warmup:, k + "norm"].values.tolist()).mean(axis=0)
        avg_tbdm = np.array(df.loc[warmup:, k + "value"].values.tolist()).mean(axis=0)
        tbdm_est[k] = normalize_tbdm(avg_tbdm, avg_norm)
        print(avg_norm, avg_tbdm, mftbdm[0,1], mftbdm[1,0])
    exit(0) 
    print(tbdm_est["tbdm"],mftbdm_tot)

    assert np.max(np.abs(tbdm_est["tbdm"] - mftbdm_tot)) < 0.05
    assert np.max(np.abs(tbdm_est["tbdm_up_up"] - mftbdm[0,0])) < 0.05
    assert np.max(np.abs(tbdm_est["tbdm_up_dn"] - mftbdm[0,1])) < 0.05
    assert np.max(np.abs(tbdm_est["tbdm_dn_up"] - mftbdm[1,0])) < 0.05
    assert np.max(np.abs(tbdm_est["tbdm_dn_dn"] - mftbdm[1,1])) < 0.05

if __name__ == "__main__":
    test()
