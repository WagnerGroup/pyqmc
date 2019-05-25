import numpy as np 
from pyqmc.energy import energy

class EnergyAccumulator:
  """returns energy of each configuration in a dictionary. 
  Keys and their meanings can be found in energy.energy """
  def __init__(self, mol):
    self.mol = mol

  def __call__(self,configs, wf):
    return energy(self.mol, configs, wf)

class PGradAccumulator:
  """returns parameter derivatives of energy for each configuration"""
  def __init__ (self,enacc):
    self.enacc = enacc

  def __call__(self,configs,wf):
    nconfig=configs.shape[0]
    d=self.enacc(configs,wf)
    energy = d['total']
    pgrad = wf.pgradient()
    for k,grad in pgrad.items():
        d['dpH_'+k] = np.einsum('i,ij->ij',energy,grad)
        d['dppsi_'+k] = grad
    for k1,grad1 in pgrad.items():
        for k2,grad2 in pgrad.items():
            d['dpidpj_'+k1+k2] = np.einsum('ij,ik->ijk',grad1.reshape((nconfig,-1)),grad2.reshape((nconfig,-1)))

    return d

