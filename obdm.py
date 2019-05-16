''' Evaluate the OBDM for a wave function object. '''
import numpy as np
from copy import deepcopy

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
  def __init__(self,mol,orb_coeff,nstep=10,tstep=0.50,warmup=100):
    assert len(orb_coeff.shape)==2, "orb_coeff should be a list of orbital coefficients."

    self._orb_coeff = orb_coeff
    self._tstep = tstep
    self._mol = mol
    self._extra_config = np.random.normal(scale=tstep,size=3) # not zero to avoid sitting on top of atom.
    self._nstep = nstep

    # Maybe shouldn't do this here?
    for i in range(warmup):
      accept,self._extra_config = sample_onebody(mol,orb_coeff,self._extra_config,tstep)

  def __call__(self,configs,wf):
    ''' Returns numerator and denomenator and their errors in equation (9) of DOI:10.1063/1.4793531'''

    results = {
        'value':np.zeros((configs.shape[0],self._orb_coeff.shape[0],self._orb_coeff.shape[1])),
        'norm':np.zeros(self._orb_coeff.shape[0])
      }
    acceptance = 0

    for step in range(self._nstep):
      points = np.concatenate((self._extra_config.reshape(1,3),configs.reshape(configs.shape[0]*configs.shape[1],configs.shape[2])))
      ao = self._mol.eval_gto('GTOval_sph',points)
      borb = ao.dot(self._orb_coeff) 

      # Orbital evaluations at extra coordinate.
      borb_prim = borb[0]
      borb_prim_sq = borb_prim**2
      fsum = borb_prim_sq.sum()
      norm = borb_prim_sq/fsum

      # Orbital evaluations at all electrons and configs.
      borb = borb[1:].reshape(configs.shape[0],configs.shape[1],borb.shape[1])

      # Numerator of obervable, given all these quantities.
      # TODO loop necessary for wfratio?
      wfratio = np.array([wf.testvalue(esel,self._extra_config[np.newaxis,:]) for esel in range(configs.shape[1])]).T
      orbratio = (borb_prim/fsum)[np.newaxis,np.newaxis,:,np.newaxis]*borb[:,:,np.newaxis,:]

      # Accumulate results for old extra coord.
      results['value'] += ( wfratio[:,:,np.newaxis,np.newaxis]*orbratio ).sum(axis=1)
      results['norm'] += norm

      # Update extra coord.
      accept,self._extra_config = sample_onebody(self._mol,self._orb_coeff,self._extra_config,tstep=self._tstep)
      
      # Keep track of internal acceptance.
      acceptance += accept

    print("OBDM sample acceptance ratio",acceptance/self._nstep)

    results['value'] /= self._nstep
    results['norm'] = results['norm'][np.newaxis,:]/self._nstep

    return results

def sample_onebody(mol,orb_coeff,epos,tstep=2.0):
  ''' For a set of orbitals defined by orb_coeff, return samples from f(r) = \sum_i phi_i(r)^2. '''
  configs = np.array((epos,epos+np.random.normal(scale=tstep,size=3)))

  ao = mol.eval_gto('GTOval_sph',configs)
  borb = ao.dot(orb_coeff)
  fsum = (borb**2).sum(axis=1)

  accept = fsum[1]/fsum[0] > np.random.rand()

  if accept:
    return 1,configs[1]
  else:
    return 0,configs[0]

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
  from slater import PySCFSlaterRHF
  from mc import initial_guess_vectorize,vmc
  from accumulators import EnergyAccumulator
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
  nconf = 5000
  nsteps = 50
  obdm_steps = 20
  warmup = 15
  wf = PySCFSlaterRHF(nconf,mol,mf)
  configs = initial_guess_vectorize(mol,nconf) 
  energy = EnergyAccumulator(mol)
  obdm = OBDMAccumulator(mol=mol,orb_coeff=mf.mo_coeff,nstep=obdm_steps)
  df,coords = vmc(mol,wf,configs,nsteps=nsteps,accumulators={'energy':energy,'obdm':obdm})
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
