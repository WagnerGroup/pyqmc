import numpy as np
from pyscf import lib, gto, scf

def determinant(mol,mo_coeff,mo_occ,epos):
    mycoords=epos.reshape((epos.shape[0]*epos.shape[1],epos.shape[2]))
    ao = mol.eval_gto('GTOval_sph', mycoords)
    mo = ao.dot(mo_coeff)
    mo=mo.reshape((epos.shape[0],epos.shape[1],mo.shape[-1]))
    nup=np.sum(mo_occ[0])
    uporbs=mo[:,0:nup,mo_occ[0]]
    detup=[np.linalg.det(x) for x in uporbs]
    dnorbs=mo[:,nup:,mo_occ[1]]
    detdn=[np.linalg.det(x) for x in dnorbs]
    return np.asarray(detup)*np.asarray(detdn)


def detupdate(row,i,inverse):
    """return the ratio of the old determinant to the new one when row i is changed, 
    given the inverse of the matrix"""
    


def distance_table(a,b):
    na=a.shape[1]
    nb=b.shape[2]
    nconf=a.shape[0]
    assert a.shape[0]==b.shape[0]
    delta=np.zeros((nconf,na,nb,3))
    for i in range(na):
        for j in range(nb):
            delta[:,i,j,:]=b[:,j,:]-a[:,i,:]
    deltar=np.sqrt(np.sum(delta**2,axis=2))
    return delta,deltar

def ee_energy(epos):
    delta,deltar=distance_table(epos,epos)
    ee=np.sum(np.triu(1./deltar),axis=1)
    return ee

def ei_energy(mol,epos):
    ei=0.0
    for c,coord in zip(mol.atom_charges(),mol.atom_coords()):
        print(coord)
        delta=epos-coord[np.newaxis,np.newaxis,:]
        deltar=np.sqrt(np.sum(delta**2,axis=2))
        ei+=-c*np.sum(1./deltar,axis=1)
    return ei



mol = gto.M(atom='Li 0. 0. 0.; H 0. 0. 1.5', basis='cc-pvtz',unit='bohr')
mf = scf.RHF(mol).run()
occ=np.asarray([mf.mo_occ > 1.0,mf.mo_occ > 1.0])
print(occ)
nelec=np.sum(mol.nelec)
nconf=5000

coords = np.random.normal(scale=1.,size=(nconf,nelec,3))

nsteps=100
tstep=0.1
olddets=determinant(mol,mf.mo_coeff,occ,coords)

df=[]
for step in range(nsteps):
    print("step",step)
    for e in range(nelec):
        newcoords=coords.copy()
        newcoords[:,e,:]=coords[:,e,:]+np.random.normal(scale=tstep,size=(nconf,3))
        newdets=determinant(mol,mf.mo_coeff,occ,newcoords)
        accept=newdets**2 / olddets**2 > np.random.rand(nconf)
        
        olddets[accept]=newdets[accept]
        coords[accept,:,:]=newcoords[accept,:,:]
        print("accept",np.mean(accept))
    ee=ee_energy(coords)
    ei=ei_energy(mol,coords)
    df.append({'avgwf':np.sqrt(np.mean(olddets**2)),
               'ee':np.mean(ee),
               'ei':np.mean(ei),
               'pos':np.mean(np.mean(coords,axis=0),axis=0)
               })
import pandas as pd
df=pd.DataFrame(df)
df.to_csv("data.csv")
