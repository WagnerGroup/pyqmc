''' Evaluate the OBDM for a wave function object. '''
import numpy as np
from copy import deepcopy

# Implementation TODO:
# - [x] Set up test calculation RHF Li calculation.
# - [x] Evaluate orbital ratios and normalizations.
# - [x] Sample from orbitals distribution.
# - [ ] Run in VMC sampling routine on Slater Det for RHF Li and check if it matches in MO and AO basis.
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
  def __init__(self,mol,orb_coeff,nstep=10,tstep=0.5,warmup=100):
    assert len(orb_coeff.shape)==2, "orb_coeff should be a list of orbital coefficients."

    self._orb_coeff = orb_coeff
    self._tstep = tstep
    self._mol = mol
    self._extra_config = np.zeros(3)
    self._nstep = nstep

    # Maybe shouldn't do this here?
    for i in range(warmup):
      accept,self._extra_config = sample_onebody(mol,orb_coeff,self._extra_config,tstep)

  # NOTE plan is to remove mol from this, but mc.py hasn't been updated.
  def __call__(self,mol,configs,wf):
    ''' Returns expectations of numerator, denomenator, and their errors in equation (9) of DOI:10.1063/1.4793531'''

    esel = 0 # TODO Later should iterate over all electrons.
    results = {
        'value':np.zeros((configs.shape[0],self._orb_coeff.shape[0],self._orb_coeff.shape[1])),
        'norm':np.zeros(self._orb_coeff.shape[0])
      }
    acceptance = 0

    # TODO Will need to sample borb[0] and average self._nstep times.
    for step in range(self._nstep):
      print("OBDM step")
      points = np.concatenate((self._extra_config.reshape(1,3),configs[:,esel,:]))
      ao = self._mol.eval_gto('GTOval_sph',points)
      borb = ao.dot(self._orb_coeff) 

      extra_configs = configs.copy()
      extra_configs[:,esel,:] = self._extra_config[np.newaxis,:]

      phi_prim_sq = borb[0]**2
      fsum = phi_prim_sq.sum()
      norm = (phi_prim_sq/fsum)**0.5

      wfratio = wf.testvalue(esel,extra_configs)
      #old_orbratio = np.einsum('k,ci->kci',(borb[0]/fsum),borb[1:]).swapaxes(0,1)
      orbratio = (borb[0]/fsum)[np.newaxis,:,np.newaxis]*borb[1:][:,np.newaxis,:]
      #assert np.allclose(old_orbratio,orbratio) # Hard to believe this works!

      # Accumulate results for old extra coord.
      results['value'] += wfratio[:,np.newaxis,np.newaxis]*orbratio
      results['norm'] += norm

      # Update extra coord.
      accept,self._extra_config = sample_onebody(self._mol,self._orb_coeff,self._extra_config)
      
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
  from energy import energy
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
  obdm_steps = 50
  wf = PySCFSlaterRHF(nconf,mol,mf)
  configs = initial_guess_vectorize(mol,nconf) 
  obdm = OBDMAccumulator(mol=mol,orb_coeff=lowdin,nstep=obdm_steps)
  df,coords = vmc(mol,wf,configs,nsteps=nsteps,accumulators={'energy':energy,'obdm':obdm})
  df = DataFrame(df)
  df['obdm'] = df[['obdmvalue','obdmnorm']]\
      .apply(lambda x:normalize_obdm(x['obdmvalue'],x['obdmnorm']),axis=1)
  print(df.loc[range(nsteps-10,nsteps),['obdmvalue','obdmnorm','obdm']].applymap(lambda x:x.ravel()[0]))
  print("correct",mfobdm[0,0])

def normalize_obdm(obdm,norm):
  return obdm/(norm[np.newaxis,:]*norm[:,np.newaxis])

if __name__=="__main__":
  test()
