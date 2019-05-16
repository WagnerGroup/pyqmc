import numpy as np 
from energy import * 
"""returns energy of each configuration in a dictionary"""
class EnergyAccumulator:
  def __init__(self, mol):
    self.mol = mol

  def __call__(self,configs, wf):
    return energy(self.mol, configs, wf)

"""returns parameter derivatives of energy for each configuration"""
class PGradAccumulator:
  def __init__ (self,EnergyAccumulator):
    self.EnergyAccumulator = EnergyAccumulator

  def __call__(self,configs,wf):
    d=self.EnergyAccumulator(configs,wf)
    energy = d['total']
    pgrad = wf.pgradient()
    for k,grad in pgrad.items():
        d['dpH_'+k] = energy[:,np.newaxis,np.newaxis]*grad
        d['dppsi_'+k] = grad
    return d

