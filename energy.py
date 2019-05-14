import numpy as np
import scipy
import scipy.spatial


def ee_energy(epos):
    ne=epos.shape[1]
    ee=np.zeros(epos.shape[0])
    for i in range(ne):
        for j in range(i+1,ne):
            ee+=1./np.sqrt(np.sum((epos[:,i,:]-epos[:,j,:])**2,axis=1))
    return ee

def ei_energy(mol,epos):
    ei=0.0
    for c,coord in zip(mol.atom_charges(),mol.atom_coords()):
        delta=epos-coord[np.newaxis,np.newaxis,:]
        deltar=np.sqrt(np.sum(delta**2,axis=2))
        ei+=-c*np.sum(1./deltar,axis=1)
    return ei


def ii_energy(mol):
    ei=0.0
    r=scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(mol.atom_coords()))
    c=mol.atom_charges()
    ind=np.triu_indices_from(r,1)
    ii=np.sum((np.outer(c,c)/r)[ind])
    return ii

def ecp(mol,epos,wf):
    
    pass

def energy(mol,epos,wf):
    ee=ee_energy(epos)
    ei=ei_energy(mol,epos)
    ii=ii_energy(mol)
    nconf=epos.shape[0]
    ke=np.zeros(nconf)
    nelec=epos.shape[1]
    for e in range(nelec):
        ke+=-0.5*wf.laplacian(e,epos[:,e,:])
    return {'ke':ke,
            'ee':ee,
            'ei':ei,
            'total':ke+ee+ei+ii } 
