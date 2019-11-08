import numpy as np
from pyscf import gto, scf, lo




###########################################################
def singledet_tbdm(mf,mfobdm):
    '''Computes the single Sltater determinant tbdm.'''
    if isinstance(mf, scf.hf.RHF):
        norb=mf.mo_energy.size
        mftbdm=np.tile(np.nan,[norb,norb,norb,norb])
        mftbdm=np.einsum('ik,jl->ijkl',mfobdm,mfobdm) - np.einsum('il,jk->ijkl',mfobdm,mfobdm)
        # Rotation into pySCF's 2-RDMs notation
        mftbdm=np.transpose(mftbdm,axes=(0,2,1,3))
    elif isinstance(mf, scf.uhf.UHF):
        norb=mf.mo_energy[0].size
        mftbdm=np.tile(np.nan,[2,2]+[norb,norb,norb,norb])
        for spin in 0,1:
            mftbdm[spin,spin]=np.einsum('ik,jl->ijkl',mfobdm[spin],mfobdm[spin]) - np.einsum('il,jk->ijkl',mfobdm[spin],mfobdm[spin])
        mftbdm[0,1]=np.einsum('ik,jl->ijkl',mfobdm[0],mfobdm[1])
        mftbdm[1,0]=np.einsum('ik,jl->ijkl',mfobdm[1],mfobdm[0])
        # Rotation into pySCF's 2-RDMs notation
        mftbdm=np.transpose(mftbdm,axes=(0,1,2,4,3,5))
        
    return mftbdm
###########################################################



###########################################################
def make_combined_spin_iaos(cell,mf,mos,iao_basis='minao'):
  ''' Make IAOs for up and down MOs together for all k points. 
  Args:
    cell (PySCF cell): Cell for the calculation.
    mf (PySCF UKS or UHF object): Contains the MOs information.
    mos (array): indices of the MOs to use to make the IAOs.
    basis (basis): IAO basis desired (in PySCF format).
  Returns:
    iaos_all (list): each list entry is np array of IAO orbitals 
                     in the basis of cell for a given k point.
  '''
  #print("Making combined spin-up and spin-dw IAOs...")
  ovlp = mf.get_ovlp()
  # Concatenates the spin-up and the spin-down chosen MOs
  coefs = np.array(mf.mo_coeff)[:,:,mos] # Notice that, unlike the KUHF case, here we do not need to transpose the matrix
  coefs = np.concatenate([coefs[0].T,coefs[1].T]).T
  iaos=lo.iao.iao(cell, coefs, minao=iao_basis)  
  iaos=lo.vec_lowdin(iaos, ovlp)
  return iaos
###########################################################



###########################################################
def reps_combined_spin_iaos(iaos,mf,mos):
  ''' Representation of MOs in IAO basis.
  Args:
    iaos (array): coefficients of IAOs in AO basis.
    mf (UKS or UHF object): the MOs are in here.
    mos (array): MOs to find representation of. Not necessarily the same as in make_combined_spin_iaos()!!!
  Returns:
    array: MO coefficients in IAO basis (remember that MOs are the columns).
  '''
  # Checks if mos has 2 spins
  if len(mos)!=2:
    mos=np.array([mos,mos])
  #print('mos:\n',mos)
  # Computes MOs passed in array 'mos' in terms of the 'iaos' basis
  #print('iaos.shape =',iaos.shape)
  #print('get_ovlp.shape =',(mf.get_ovlp()[0]).shape)
  #print('np.array(mf.mo_coeff)[0,0,:,mos].shape =',(np.array(mf.mo_coeff)[0,0,:,mos]).shape)
  #print('np.array(mf.mo_coeff)[0,0,:,mos]:\n',(np.array(mf.mo_coeff)[0,0,:,mos]))
  iao_reps = [np.dot( np.dot(iaos.T,mf.get_ovlp()), (np.array(mf.mo_coeff)[s,:,mos[s]]).transpose((1,0)) ) for s in range(np.array(mf.mo_coeff).shape[0])]
  
  return iao_reps
###########################################################



if __name__ == "__main__":
    from numpy.linalg import solve
    from pyqmc import PySCFSlaterUHF
    from pyqmc.mc import initial_guess, vmc
    from pyqmc.accumulators import EnergyAccumulator
    from pyqmc.tbdm import TBDMAccumulator, normalize_tbdm
    from pandas import DataFrame

    mol = gto.M(
        atom="He 0. 0. 0.; He 0. 0. 1.5", basis="minao", unit="bohr", verbose=4
    )
    #mf = scf.RHF(mol).run()
    mf = scf.UHF(mol).run()

    # IAOs
    iaos=make_combined_spin_iaos(mol,mf,np.array([i for i in range(mol.natm)]),iao_basis='minao')
    # MOs in the IAO basis
    mo = solve(iaos, mf.mo_coeff)
    # OBDM in IAO basis
    mfobdm = mf.make_rdm1(mo, mf.mo_occ)
    # TBDM in IAO basis
    mftbdm=singledet_tbdm(mf,mfobdm)
    # IAOs, OBDM and TBDM testing (with different construction)
    #obdm, tbdm = uhf_rdm12_calc(mol,mf,basis='IAOs',convention='pySCF')
    #reps_iaos = reps_combined_spin_iaos(iaos,mf,np.einsum('i,j->ji',np.arange(mf.mo_coeff[0].shape[1]),np.array([1,1])))
    #print('mo-reps_iaos:\n',mo-reps_iaos,np.all(abs(mo-reps_iaos)<10e-15))
    #print('mfobdm-obdm:\n',mfobdm-obdm,np.all(abs(mfobdm-obdm)<10e-15))
    #print('mftbdm-tbdm:\n',mftbdm-tbdm,np.all(abs(mftbdm-tbdm)<10e-15))
    
    
    ### Test TBDM calculation.
    # VMC params
    nconf = 200
    n_vmc_steps = 100
    vmc_tstep = 0.01
    vmc_warmup = 25
    # TBDM params
    tbdm_sweeps = 15
    tbdm_tstep = 0.1

    wf = PySCFSlaterUHF(mol, mf) # Single-Slater (no jastrow) wf
    configs = initial_guess(mol, nconf)
    energy = EnergyAccumulator(mol)
    #tbdm = TBDMAccumulator(mol=mol, orb_coeff=lowdin, nstep=tbdm_steps)
    tbdm_upup = TBDMAccumulator(mol=mol, orb_coeff=iaos, spin=[0,0], nsweeps=tbdm_sweeps, tstep=tbdm_tstep)
    tbdm_updw = TBDMAccumulator(mol=mol, orb_coeff=iaos, spin=[0,1], nsweeps=tbdm_sweeps, tstep=tbdm_tstep)
    tbdm_dwup = TBDMAccumulator(mol=mol, orb_coeff=iaos, spin=[1,0], nsweeps=tbdm_sweeps, tstep=tbdm_tstep)
    tbdm_dwdw = TBDMAccumulator(mol=mol, orb_coeff=iaos, spin=[1,1], nsweeps=tbdm_sweeps, tstep=tbdm_tstep)

    
    #print('tbdm._mol:\n',tbdm._mol)
    #print('tbdm._orb_coeff:\n',tbdm._orb_coeff)
    #print('tbdm._nstep:\n',tbdm._nstep)
    #print('tbdm._tstep:\n',tbdm._tstep)
    #print('tbdm._extra_config:\n',tbdm._extra_config.shape)
    #print('tbdm._electrons1:\n',tbdm._electrons1)
    #print('tbdm._electrons2:\n',tbdm._electrons2)

    #print('tbdm_upup._electrons1:\n',tbdm_upup._electrons1)
    #print('tbdm_upup._electrons2:\n',tbdm_upup._electrons2)

    #print('tbdm_updw._electrons1:\n',tbdm_updw._electrons1)
    #print('tbdm_updw._electrons2:\n',tbdm_updw._electrons2)

    #print('tbdm_dwup._electrons1:\n',tbdm_dwup._electrons1)
    #print('tbdm_dwup._electrons2:\n',tbdm_dwup._electrons2)

    #print('tbdm_dwdw._electrons1:\n',tbdm_dwdw._electrons1)
    #print('tbdm_dwdw._electrons2:\n',tbdm_dwdw._electrons2)

    
    print('VMC...')
    df, coords = vmc(
        wf,
        configs,
        nsteps=n_vmc_steps,
        tstep=vmc_tstep,
        accumulators={
            "energy": energy,
            #"tbdm": tbdm,
            "tbdm_upup": tbdm_upup,
            "tbdm_updw": tbdm_updw,
            "tbdm_dwup": tbdm_dwup,
            "tbdm_dwdw": tbdm_dwdw,
        },
        verbose=True,
    )
    df = DataFrame(df)
    print(df)

    print(df.keys())
    tbdm_est = {}
    for k in ["tbdm_upup", "tbdm_updw", "tbdm_dwup", "tbdm_dwdw"]: #"tbdm",
        avg_norm = np.array(df.loc[vmc_warmup:, k + "norm"].values.tolist()).mean(axis=0)
        avg_tbdm = np.array(df.loc[vmc_warmup:, k + "value"].values.tolist()).mean(axis=0)
        tbdm_est[k] = normalize_tbdm(avg_tbdm, avg_norm)

    qmctbdm=np.array([[tbdm_est["tbdm_upup"],tbdm_est["tbdm_updw"]],[tbdm_est["tbdm_dwup"],tbdm_est["tbdm_dwdw"]]])
    for sa,sb in [[0,0],[0,1],[1,0],[1,1]]:
      print('QMC tbdm[%d,%d]:\n'%(sa,sb),qmctbdm[sa,sb])
      print('MF tbdm[%d,%d]:\n'%(sa,sb),mftbdm[sa,sb])
      print('diff[%d,%d]:\n'%(sa,sb),qmctbdm[sa,sb]-mftbdm[sa,sb])
    print(qmctbdm.shape,mftbdm.shape)
    exit()
    
    print("Average TBDM(orb,orb,orb,orb)", tbdm_est["tbdm"].diagonal().round(3))
    print("mf tbdm", mftbdm.diagonal().round(3))
    assert np.max(np.abs(obdm_est["obdm"] - mfobdm)) < 0.05
    print(obdm_est["obdm_up"].diagonal().round(3))
    print(obdm_est["obdm_down"].diagonal().round(3))
    assert np.max(np.abs(obdm_est["obdm_up"] + obdm_est["obdm_down"] - mfobdm)) < 0.05
