import numpy as np
import scipy
import scipy.spatial
import pyqmc.eval_ecp  as eval_ecp
from pyqmc.distance import RawDistance


def ee_energy(configs):
    ne=configs.shape[1]
    if ne==1:
        return np.zeros(configs.shape[0])
    ee=np.zeros(configs.shape[0])
    d=RawDistance()
    ee,ij=d.dist_matrix(configs)
    ee=np.linalg.norm(ee,axis=2)
    return np.sum(1./ee,axis=1)

def ei_energy(mol,configs):
    ei=0.0
    for c,coord in zip(mol.atom_charges(),mol.atom_coords()):
        delta=configs-coord[np.newaxis,np.newaxis,:]
        deltar=np.sqrt(np.sum(delta**2,axis=2))
        ei+=-c*np.sum(1./deltar,axis=1)
    return ei


def ii_energy(mol):
    ei=0.0
    d=RawDistance()
    rij,ij = d.dist_matrix(mol.atom_coords()[np.newaxis,:,:])
    if len(ij)==0:
        return np.array([0.0])
    rij=np.linalg.norm(rij,axis=2)[0,:]
    iitot=0
    c=mol.atom_charges()
    for (i,j),r in zip(ij,rij):
        iitot+=c[i]*c[j]/r
    return iitot

def get_ecp(mol,configs,wf):
    return eval_ecp.ecp(mol, configs, wf)
    

def kinetic(configs,wf):
    nconf=configs.shape[0]
    ke=np.zeros(nconf)
    nelec=configs.shape[1]
    for e in range(nelec):
        ke+=-0.5*wf.laplacian(e,configs[:,e,:])
    return ke

def energy(mol,configs,wf):
    """Compute the local energy of a set of configurations.
    
    Args:
      mol: A pyscf-like 'Mole' object. nelec, atom_charges(), atom_coords(), and ._ecp are used.

      configs: a nconfiguration x nelectron x 3 numpy array
       
      wf: A Wavefunction-like object. Functions used include recompute(), lapacian(), and testvalue()

    Returns: 
      a dictionary with energy components ke, ee, ei, and total
      """
    ee=ee_energy(configs)
    ei=ei_energy(mol,configs)
    ecp_val = get_ecp(mol,configs,wf)
    ii=ii_energy(mol)
    ke=kinetic(configs,wf)
    #print(ke,ee,ei,ii)
    return {'ke':ke,
            'ee':ee,
            'ei':ei+ecp_val,
            'total':ke+ee+ei+ecp_val+ii }
