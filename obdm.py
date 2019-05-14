''' Evaluate the OBDM for a wave function object. '''
import numpy as np
from mc import initial_guess
from copy import deepcopy

# Implementation TODO:
# - [x] Set up test calculation RHF Li calculation.
# - [ ] Evaluate orbital ratios and normalizations.
# - [ ] Sample from orbitals distribution.
# - [ ] Evaluate observable for each electron update and save in array.
# - [ ] Run in VMC sampling routine on Slater Det for RHF Li and check if it matches in MO and AO basis.
# - [ ] Run in VMC with Jastrow etc. to check if it makes sense in MO and AO basis.

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
  def __init__(self,mol,orb_coeff,tstep=0.5):
    assert len(orb_coeff.shape)>2, "orb_coeff should be a list of orbital coefficients."

    self._orb_coeff = orb_coeff
    self._tstep = tstep
    self._mol = mol
    self._extra_config = np.array((0.0,0.0,0.0))

    ao = mol.eval_gto('GTOval_sph',self._extra_config)
    borb = ao.dot(orb_coeff)
    self._extra_config_prob = (borb**2).sum() 

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
  from mc import initial_guess

  # Simple Li2 run.
  mol = gto.M(atom='Li 0. 0. 0.; Li 0. 0. 1.5', basis='cc-pvdz',unit='bohr',verbose=0)
  mf = scf.RHF(mol).run()

  # Lowdin orthogonalized AO basis.
  lowdin = lo.orth_ao(mol, 'lowdin')

  # MOs in the Lowdin basis.
  mo = solve(lowdin, mf.mo_coeff)

  # make AO to localized orbital coefficients.
  mfobdm = mf.make_rdm1(mo, mf.mo_occ)

  #print(mfobdm.diagonal().round(2))

  ### VMC obdm run.
  test_sample_onebody(mol,lowdin,mf,nsample=int(1e4))
  test_sample_onebody(mol,lowdin,mf,nsample=int(4e4))
  test_sample_onebody(mol,lowdin,mf,nsample=int(1e5))

if __name__=="__main__":
  test()
