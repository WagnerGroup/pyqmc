import numpy as np
import scipy
import scipy.spatial


def ee_energy(configs):
    ne=configs.shape[1]
    ee=np.zeros(configs.shape[0])
    for i in range(ne):
        for j in range(i+1,ne):
            ee+=1./np.sqrt(np.sum((configs[:,i,:]-configs[:,j,:])**2,axis=1))
    return ee

def ei_energy(mol,configs):
    ei=0.0
    for c,coord in zip(mol.atom_charges(),mol.atom_coords()):
        delta=configs-coord[np.newaxis,np.newaxis,:]
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

def get_ecp(mol,configs,wf):
    import eval_ecp  # this needs to be fixed later...
    return eval_ecp.ecp(mol, configs, wf)
    

def kinetic(configs,wf):
    nconf=configs.shape[0]
    ke=np.zeros(nconf)
    nelec=configs.shape[1]
    for e in range(nelec):
        ke+=-0.5*wf.laplacian(e,configs[:,e,:])
    return ke

def energy(mol,configs,wf):
    ee=ee_energy(configs)
    ei=ei_energy(mol,configs)
    ecp_val = get_ecp(mol,configs,wf)
    ii=ii_energy(mol)
    nconf=configs.shape[0]
    ke=kinetic(configs,wf)
    return {'ke':ke,
            'ee':ee,
            'ei':ei+ecp_val,
            'total':ke+ee+ei+ecp_val+ii }
