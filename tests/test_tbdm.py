import numpy as np
from pyscf import gto, scf, lo

###########################################################
##  Calculates 1- and 2-RDM in AO basis.                 ##
###########################################################
def uhf_rdm12_calc(cell,mf,basis='AOs',convention='QWalk'):
  '''Calculates the 1-RDM and 2-RDM for a single Slater determinant in the AO basis.
 
  NOTE: the 2-RDM calculated below (from downfold_tools.py) is in QWalk notation, i.e. tbdm[s1,s2,i,j,k,l]= < c_s1,i^+ 
        c_s2,j^+ c_s1,k c_s2,l > , and not in pyscf's notation as the other method of calculating the 2-RDM (see 
        uhf_rdm12_calc), which is written as tbdm[s1,s2,i,j,k,l]= < c_s1,i^+ c_s2,k^+ c_s1,j c_s2,l > . 
  Args:
    cell (Cell): Cell object.
    mf (MeanField): Mean-field object.
    basis (str): specifies the basis in which RDMs are returned.
    convention (str): whether 2RDM is output in QWalk's or pySCF's index convention: 
                      QWalk --> tbdm[s1,s2,i,j,k,l] = < c_s1,i^+ c_s2,j^+ c_s1,k c_s2,l > ;
                      pySCF --> tbdm[s1,s2,i,j,k,l] = < c_s1,i^+ c_s2,k^+ c_s1,j c_s2,l > ;
  Returns:
    obdm (array): contains the [rdm1_alpha, rdm1_beta].
    tbdm (array): contains the [[tbdm_aa, tbdm_ab], [tbdm_ba,tbdm_bb]] in pyscf notation.
  '''  
  # Calculates the 1RDM from the UHF class (in AO basis)
  obdm0 = np.array(mf.make_rdm1(mf.mo_coeff,mf.mo_occ))
  
  # Computes the TBDM for a single Slater determinant (in AO basis).
  norb=obdm0.shape[1]
  nelec=mf.nelec
  tbdm0=np.tile(np.nan,[2,2]+[norb,norb,norb,norb])
  for spin in 0,1:
    tbdm0[spin,spin]=np.einsum('ik,jl->ijkl',obdm0[spin],obdm0[spin]) - np.einsum('il,jk->ijkl',obdm0[spin],obdm0[spin])
  tbdm0[0,1]=np.einsum('ik,jl->ijkl',obdm0[0],obdm0[1])
  tbdm0[1,0]=np.einsum('ik,jl->ijkl',obdm0[1],obdm0[0])
  # Rotation into pySCF's 2-RDMs notation
  tbdm0=np.transpose(tbdm0,axes=(0,1,2,4,3,5)) 

  # Sets the basis in which the RDMs will be output
  if basis=='IAOs': # Rotation matrices from AOs into IAOs
    # Constructs IAOs and MO representations (in IAOs basis)
    iaos=make_combined_spin_iaos(cell,mf,np.array([i for i in range(cell.natm)]),iao_basis='minao')
    mo_coeff_a=np.dot( iaos.T, mf.get_ovlp() ).T
    mo_coeff_b=np.dot( iaos.T, mf.get_ovlp() ).T
  elif basis=='MOs': # Rotation matrices from AOs into MOs
    mo_coeff_a=np.dot( (mf.mo_coeff[0]).T, mf.get_ovlp() ).T
    mo_coeff_b=np.dot( (mf.mo_coeff[1]).T, mf.get_ovlp() ).T
  #else: # RDMs in AOs basis
  
  # Rotation of the 1-RDM and the 2-RDM into desired basis
  if ( (basis=='IAOs') | (basis=='MOs') ):
    mo_coeff=[mo_coeff_a,mo_coeff_b]
    # Transforming the RDM1
    obdm=np.array([ np.dot(np.dot(mo_coeff_a.T.conj(),obdm0[0]),mo_coeff_a) , np.dot(np.dot(mo_coeff_b.T.conj(),obdm0[1]),mo_coeff_b) ])
  
    # Transforming the 2-RDM: we change from AO (R_{ijkl}) into MOs or IAOs (R_{alpha,beta,gamma,sigma}) basis: 
    #  R_{alpha,beta,gamma,sigma} = [C^T^*]_{alpha,i} [C^T^*]_{gamma,k} Rab_{ijkl} C_{j,beta} C_{l,sigma}
    tbdm=np.zeros((tbdm0.shape[0],tbdm0.shape[1],mo_coeff_a.T.shape[0],mo_coeff_b.T.shape[0],mo_coeff_a.shape[1],mo_coeff_b.shape[1]))
    for s1 in range(tbdm0.shape[0]):
      for s2 in range(tbdm0.shape[1]):
        rdm2_rot=np.tensordot(mo_coeff[s1].T.conj(), tbdm0[s1,s2], axes=([[1],[0]])) # \sum_i [Ca^T^*]_{alpha}^{i} Rab_{ijkl}
        rdm2_rot=np.transpose( np.tensordot(rdm2_rot, mo_coeff[s1], axes=([[1],[0]])) , axes=(0,3,1,2)) # \sum_j Rab_{ijkl} Ca^{j}_{beta}
        rdm2_rot=np.transpose( np.tensordot(mo_coeff[s2].T.conj(), rdm2_rot, axes=([[1],[2]])), axes=(1,2,0,3)) #\sum_k [Cb^T^*]_{gamma}^{k} Rab_{ijkl}
        tbdm[s1,s2]=np.tensordot(rdm2_rot, mo_coeff[s2], axes=([[3],[0]])) # \sum_l Rab_{ijkl} Cb^{l}_{sigma}
  else:
    obdm=obdm0
    tbdm=tbdm0    


    
  # Changes in QWalk's index convention
  if convention=='QWalk':
    tbdm=np.transpose(tbdm,axes=(0,1,2,4,3,5)) 

  return obdm, tbdm
###########################################################


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

    # Lowdin orthogonalized AO basis.
    #lowdin = lo.orth_ao(mol, "lowdin")
    # MOs in the Lowdin basis.
    #mo = solve(lowdin, mf.mo_coeff)

    # IAOs
    lowdin=make_combined_spin_iaos(mol,mf,np.array([i for i in range(mol.natm)]),iao_basis='minao')
    # MOs in the IAO basis.
    mo = solve(lowdin, mf.mo_coeff)

    # make AO to localized orbital coefficients.
    mfobdm = mf.make_rdm1(mo, mf.mo_occ)
    # make AO 2rdm for single slater
    mftbdm=singledet_tbdm(mf,mfobdm)

    
    
    ### Test TBDM calculation.
    nconf = 4
    nsteps = 400
    tbdm_steps = 5
    warmup = 15
    wf = PySCFSlaterUHF(mol, mf)
    configs = initial_guess(mol, nconf)
    energy = EnergyAccumulator(mol)
    tbdm = TBDMAccumulator(mol=mol, orb_coeff=lowdin, nstep=tbdm_steps)
    tbdm_upup = TBDMAccumulator(mol=mol, orb_coeff=lowdin, nstep=tbdm_steps, spin=[0,0])
    tbdm_updw = TBDMAccumulator(mol=mol, orb_coeff=lowdin, nstep=tbdm_steps, spin=[0,1])
    tbdm_dwup = TBDMAccumulator(mol=mol, orb_coeff=lowdin, nstep=tbdm_steps, spin=[1,0])
    tbdm_dwdw = TBDMAccumulator(mol=mol, orb_coeff=lowdin, nstep=tbdm_steps, spin=[1,1])

    
    print('tbdm._mol:\n',tbdm._mol)
    print('tbdm._orb_coeff:\n',tbdm._orb_coeff)
    print('tbdm._nstep:\n',tbdm._nstep)
    print('tbdm._tstep:\n',tbdm._tstep)
    print('tbdm._extra_config:\n',tbdm._extra_config.shape)
    print('tbdm._electrons1:\n',tbdm._electrons1)
    print('tbdm._electrons2:\n',tbdm._electrons2)

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
        nsteps=nsteps,
        accumulators={
            "energy": energy,
            "tbdm": tbdm,
            "tbdm_upup": tbdm_upup,
            "tbdm_updw": tbdm_updw,
            "tbdm_dwup": tbdm_dwup,
            "tbdm_dwdw": tbdm_dwdw,
        },
    )
    df = DataFrame(df)
    print(df)

    print(df.keys())
    tbdm_est = {}
    for k in ["tbdm", "tbdm_upup", "tbdm_updw", "tbdm_dwup", "tbdm_dwdw"]:
        avg_norm = np.array(df.loc[warmup:, k + "norm"].values.tolist()).mean(axis=0)
        avg_tbdm = np.array(df.loc[warmup:, k + "value"].values.tolist()).mean(axis=0)
        tbdm_est[k] = normalize_tbdm(avg_tbdm, avg_norm)

    qmctbdm=np.array([[tbdm_est["tbdm_upup"],tbdm_est["tbdm_updw"]],[tbdm_est["tbdm_dwup"],tbdm_est["tbdm_dwdw"]]])
    print(qmctbdm)
    print(mftbdm)
    print(qmctbdm.shape,mftbdm.shape)
    exit()
    
    print("Average TBDM(orb,orb,orb,orb)", tbdm_est["tbdm"].diagonal().round(3))
    print("mf tbdm", mftbdm.diagonal().round(3))
    assert np.max(np.abs(obdm_est["obdm"] - mfobdm)) < 0.05
    print(obdm_est["obdm_up"].diagonal().round(3))
    print(obdm_est["obdm_down"].diagonal().round(3))
    assert np.max(np.abs(obdm_est["obdm_up"] + obdm_est["obdm_down"] - mfobdm)) < 0.05
