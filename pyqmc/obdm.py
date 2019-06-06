''' Evaluate the OBDM for a wave function object. '''
import numpy as np
from copy import deepcopy
from pyqmc.mc import initial_guess
# Implementation TODO:
# - [x] Set up test calculation RHF Li calculation.
# - [x] Evaluate orbital ratios and normalizations.
# - [x] Sample from orbitals distribution.
# - [x] Run in VMC sampling routine on Slater Det for RHF Li and check if it matches in MO and AO basis.
# - [ ] Run in VMC with Jastrow etc. to check if it makes sense in MO and AO basis.
# - [ ] Figure out good defaults for parameters.

# Notes
# - Might gain some performance by vectorizing extra samples.

class OBDMAccumulator:
  ''' Return the obdm as an array with indices rho[spin][i][k] = <c_{spin,i}c^+_{spin,j}>
  Args:
    mol (Mole): PySCF Mole object.
    configs (array): electron positions.
    wf (pyqmc wave function object): wave function to evaluate on.
    orb_coeff (array): coefficients with size (nbasis,norb) relating mol basis to basis 
      of 1-RDM desired.
    tstep (float): width of the Gaussian to update a walker position for the 
      extra coordinate.
  '''
  def __init__(self,mol,orb_coeff,nstep=10,tstep=0.50,warmup=100,naux=500):
    assert len(orb_coeff.shape)==2, "orb_coeff should be a list of orbital coefficients."

    self._orb_coeff = orb_coeff
    self._tstep = tstep
    self._mol = mol
    #self._extra_config = np.random.normal(scale=tstep,size=3) # not zero to avoid sitting on top of atom.
    nelec=sum(self._mol.nelec)
    self._extra_config=initial_guess(mol,int(naux/nelec)+1).reshape(-1,3)

    self._nstep = nstep

    # Maybe shouldn't do this here?
    for i in range(warmup):
      accept,self._extra_config = sample_onebody(mol,orb_coeff,self._extra_config,tstep)



  def __call__(self,configs,wf):
    ''' Returns numerator and denominator and their errors in equation (9) of DOI:10.1063/1.4793531'''

    results = {
        'value':np.zeros((configs.shape[0],self._orb_coeff.shape[1],self._orb_coeff.shape[1])),
        'norm':np.zeros((configs.shape[0],self._orb_coeff.shape[0]))
      }
    acceptance = 0
    naux=self._extra_config.shape[0]
    nelec=configs.shape[1]

    for step in range(self._nstep):
        e=np.random.randint(0,configs.shape[1])
        
        #print(self._extra_config.shape,configs.shape)
        points = np.concatenate([self._extra_config,
                                 configs[:,e,:]])
        ao = self._mol.eval_gto('GTOval_sph',points)
        borb = ao.dot(self._orb_coeff) 
        #print(borb.shape)


        # Orbital evaluations at extra coordinate.
        borb_aux = borb[0:naux,:]
        #borb_prim_sq = borb_prim**2
        fsum = np.sum(borb_aux*borb_aux,axis=1)
        norm = borb_aux*borb_aux/fsum[:,np.newaxis]

        # Orbital evaluations at all electrons and configs.
        #borb = borb[1:].reshape(configs.shape[0],configs.shape[1],borb.shape[1])
        borb_configs=borb[naux:,:]        

        # Numerator of observable, given all these quantities.
        auxassignments=np.random.randint(0,naux,size=configs.shape[0])
        wfratio = wf.testvalue(e,self._extra_config[auxassignments,:])
        
        orbratio = np.einsum("ij,ik->ijk",borb_aux[auxassignments,:]/fsum[auxassignments,np.newaxis],borb_configs)

        results['value'] += nelec*np.einsum('i,ijk->ijk',wfratio,orbratio)
        results['norm'] += norm[auxassignments]

        # Update extra coord.
        accept,self._extra_config = sample_onebody(self._mol,self._orb_coeff,self._extra_config,tstep=self._tstep)

        # Keep track of internal acceptance.
        acceptance += np.mean(accept)

    print("OBDM sample acceptance ratio",acceptance/self._nstep)

    results['value'] /= self._nstep
    results['norm'] = results['norm']/self._nstep

    return results

def sample_onebody(mol,orb_coeff,configs,tstep=2.0):
  ''' For a set of orbitals defined by orb_coeff, return samples from f(r) = \sum_i phi_i(r)^2. '''
  config_pack = np.concatenate([configs,configs+np.sqrt(tstep)*np.random.randn(*configs.shape)],axis=0)

  ao = mol.eval_gto('GTOval_sph',config_pack)
  borb = ao.dot(orb_coeff)
  fsum = (borb**2).sum(axis=1)

  n=configs.shape[0]
  accept = fsum[n:]/fsum[0:n] > np.random.rand(n)
  newconf=config_pack[n:,:]
  configs[accept,:]=newconf[accept,:]
  return accept,configs

def test_sample_onebody(mol,orb_coeff,mf,nsample=int(1e4)):
  ''' Test the one-body sampling by sampling the integral of f(r).'''

  nwarm = nsample//4
  orb_coeff = mf.mo_coeff
  #samples = [sample_onebody(mol,orb_coeff,configs) for sample in range(nsample)]

  print("Generating samples.")
  samples = np.zeros((nsample+nwarm,3))
  accept=0
  for sidx in range(1,nsample+nwarm):
    did_accept,samples[sidx] = sample_onebody(mol,orb_coeff,samples[sidx-1])
    if sidx > nwarm: accept += did_accept
  print("accept ratio",accept/(nsample))
  samples = samples[nwarm:]

  print("Performing integration.")
  ao = mol.eval_gto('GTOval_sph',samples)
  morb = ao.dot(mf.mo_coeff)
  borb = ao.dot(orb_coeff)
  denom = (borb**2).sum(axis=1)
  orb_ovlp = morb.T@(morb*borb.shape[1]/denom[:,np.newaxis])/samples.shape[0]
  print("Mean of error",abs(orb_ovlp - np.eye(*orb_ovlp.shape)).mean())
  print("trace,norb",orb_ovlp.trace(),orb_coeff.shape[1])

def test():
  from pyscf import gto,scf,lo
  from numpy.linalg import solve
  from pyqmc.slater import PySCFSlaterRHF
  from pyqmc.mc import initial_guess,vmc
  from pyqmc.accumulators import EnergyAccumulator
  from pandas import DataFrame

  ### Generate some basic objects.
  # Simple Li2 run.
  mol = gto.M(atom='Li 0. 0. 0.; Li 0. 0. 1.5', basis='sto-3g',unit='bohr',verbose=0)
  mf = scf.RHF(mol).run()

  # Lowdin orthogonalized AO basis.
  lowdin = lo.orth_ao(mol, 'lowdin')

  # MOs in the Lowdin basis.
  mo = solve(lowdin, mf.mo_coeff)

  # make AO to localized orbital coefficients.
  mfobdm = mf.make_rdm1(mo, mf.mo_occ)

  #print(mfobdm.diagonal().round(2))

  ### Test one-body sampler.
  #test_sample_onebody(mol,lowdin,mf,nsample=int(1e4))
  #test_sample_onebody(mol,lowdin,mf,nsample=int(4e4))
  #test_sample_onebody(mol,lowdin,mf,nsample=int(1e5))

  ### Test OBDM calculation.
  nconf = 500
  nsteps = 400
  obdm_steps = 2
  warmup = 15
  wf = PySCFSlaterRHF(mol,mf)
  configs = initial_guess(mol,nconf) 
  energy = EnergyAccumulator(mol)
  obdm = OBDMAccumulator(mol=mol,orb_coeff=mf.mo_coeff,nstep=obdm_steps)
  df,coords = vmc(wf,configs,nsteps=nsteps,accumulators={'energy':energy,'obdm':obdm})
  df = DataFrame(df)
  df['obdm'] = df[['obdmvalue','obdmnorm']]\
      .apply(lambda x:normalize_obdm(x['obdmvalue'],x['obdmnorm']),axis=1)
  print(df[['obdmvalue','obdmnorm','obdm']].applymap(lambda x:x.ravel()[0]))
  avg_norm = np.array(df.loc[warmup:,'obdmnorm'].values.tolist()).mean(axis=0)
  avg_obdm = np.array(df.loc[warmup:,'obdm'].values.tolist()).mean(axis=0)
  std_obdm = np.array(df.loc[warmup:,'obdm'].values.tolist()).std(axis=0)/nsteps**0.5
  print("Average norm(orb)",avg_norm)
  print("Average OBDM(orb,orb)",avg_obdm.diagonal().round(3))
  print("OBDM error (orb,orb)",std_obdm.diagonal().round(3)) # Note this needs reblocking to be accurate.
  print("AO occupation",mfobdm[0,0])
  print('mean field',mf.energy_tot(),'vmc estimation', np.mean(df['energytotal'][warmup:]),np.std(df['energytotal'][warmup:]))

def normalize_obdm(obdm,norm):
  return obdm/(norm[np.newaxis,:]*norm[:,np.newaxis])**0.5

if __name__=="__main__":
  test()
