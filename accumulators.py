import numpy as np 
from energy import * 

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
    d=self.enacc(configs,wf)
    energy = d['total']
    pgrad = wf.pgradient()
    for k,grad in pgrad.items():
        d['dpH_'+k] = energy[:,np.newaxis,np.newaxis]*grad
        d['dppsi_'+k] = grad
    return d

