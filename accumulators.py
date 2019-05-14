import numpy as np 
from energy import * 

def EnergyAccumulator:
  def __init__(self, mol):
    self.mol = mol

  def __call__(configs, wf):
    return energy(mol, configs, wf)
