import numpy as np 
from energy import * 

class EnergyAccumulator:
  def __init__(self, mol):
    self.mol = mol

  def __call__(self,configs, wf):
    return energy(self.mol, configs, wf)



class PGradAccumulator:
  def __init__ (self,EnergyAccumalator):
    self.EnergyAccumulator = EnergyAccumulator

  def __call__(configs,wf):
    d=EnergyAccumulator(configs,wf)
    energy = d['total']
    pgrad = wf.pgradient()
    for k,grad in pgrad.items():
        d['dpH_'+k] = energy*grad
        d['dppsi_'+k] = grad
    return d

