import numpy as np 
from energy import * 

def EnergyAccumulator():
  def __init__(self, mol):
    self.mol = mol

  def __call__(configs, wf):
    return energy(mol, configs, wf)

def PGradAccumulator():
  def __init__ (self,EnergyAccumalator):
    self.EnergyAccumulator = EnergyAccumulator

  def __call__(configs,wf):
    d = {}
    energy = EnergyAccumulator(configs,wf)['total']
    pgrad = wf.pgradient()
    for k,grad in pgrad.items():
        d['dpH_'+k] = energy*grad
        d['dppsi_'+k] = grad
    return d

