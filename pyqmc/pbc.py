import numpy as np



def enforce_pbc_orth(lattvecs,init_epos,translation):
    '''
    Args:

      lattvecs: orthogonal lattice vectors defining 3D torus: (3,3)

      init_epos: initial electron coordinates: (nconfig,3)

      translation: attempted new electron coordinates: (nconfig,3)

    Returns:
    
      final_epos: final electron coordinates with PBCs imposed: (nconfig,3)

      wraparound: vector used to bring a given electron back to the simulation cell written in terms of lattvecs: (nconfig,3)
    '''
    # Checks lattice vecs orthogonality. Note: assuming latt vecs are columns, [[v1],[v2],[v3]]=lattvecs.T
    orth=np.dot(lattvecs,lattvecs.T)
    np.fill_diagonal(orth,0)
    assert np.all(orth==0), "Using enforce_pbc_orth() but lattice vectors are not orthogonal."

    # If lattvecs non-diagonal, rotates lattvecs into [[a,0,0],[0,b,0],[0,0,c]]
    #uu=np.eye(lattvecs[0])
    #if np.any( lattvecs-np.diag(np.diagonal(lattvecs))!=0. ):
    #    lattvecs,uu=np.linalg.eig(lattvecs)
    #    #print('d_lv',diag_lvecs)
    #    #print('u\n',uu)
    #    #print(np.einsum('ij,jk,kl->il',uu.conj().T,lattvecs,uu))

    #print(lattvecs)
    #print(lattvecs*lattvecs)
    #lvecs_mod=np.sqrt(np.sum(lattvecs*lattvecs,axis=0))
    #print(lvecs_mod)
    #print(lattvecs/lvecs_mod)
    #print(np.einsum('ij,j->ij',lattvecs,1/np.linalg.norm(lattvecs,axis=0)))
    #nlattvecs=np.einsum('ij,j->ij',lattvecs,1/np.linalg.norm(lattvecs,axis=0))
    #print(np.einsum('ij,jk->ik',init_epos,nlattvecs))
    diagg,uu=np.linalg.eigh(lattvecs)
    lattvecs0=np.diag(diagg)
    print('lattvecs0\n',lattvecs0)

    translation0=np.einsum('ij,kj->ki',uu.T,translation)
    print(translation0)
    inv_norm=1/np.linalg.norm(lattvecs0,axis=0)
    print(inv_norm)
    nlvecs=np.einsum('ij,j->ij',lattvecs0,inv_norm**2)
    tmp=np.einsum('ij,jk->ik',translation0,nlvecs)
    print('tmp0--\n',tmp)

    
    #print(lattvecs)
    #print(uu)
    inv_norm=1/np.linalg.norm(lattvecs,axis=0)
    print(inv_norm,1/inv_norm)
    nlvecs=np.einsum('ij,j->ij',lattvecs,inv_norm**2)
    print(lattvecs)
    print(nlvecs)
    diagg0,uu0=np.linalg.eigh(nlvecs)
    print(uu0)

    ESTA A DAR A NORMA MAL...NAO, esta certa

    tmp=np.einsum('ij,jk->ik',translation,nlvecs)
    print('tmp--\n',tmp)
    print(np.einsum('ij,kj->ki',uu0.T,tmp))

    exit()
    #tmp0=[ np.divmod(d,np.ones(3)) for d in tmp ]
    #tmp00=np.divmod(tmp,1)
    tmp=np.divmod(tmp,1)
    #print('tmp00\n',tmp00)
    #print(tmp00[1])
    print(tmp[1])
    #print(np.all(tmp00[1]==np.array([ v[1] for v in tmp0 ])))
    #wraparound=np.array([ v[0] for v in tmp0 ])
    wraparound=tmp[0]
    #print(np.array([ v[1] for v in tmp ]))
    #final_epos=np.einsum('ij,kj->ki',lattvecs,np.array([ v[1] for v in tmp0 ]))
    print(lattvecs)
    print('tmp[1]\n',tmp[1])
    #final_epos=np.einsum('ij,kj->ki',lattvecs,tmp[1])
    final_epos=np.einsum('ij,kj->ki',lattvecs,tmp[1])
    print('Fin:\n',final_epos)
    
    print('wraparound:\n',wraparound)
    #print(np.all(tmp00[0]==wraparound))
    
    diag,uu=np.linalg.eigh(lattvecs)
    print(np.einsum('ij,kj->ki',uu.T,final_epos))
    #print(init_epos%lattvecs[0])
    #print([ np.divmod(init_epos[0],l) for l in lattvecs])
    exit()

    return final_epos, wraparound 



def test_orth():
    nconf=10
    
    # TEST 1: Check if any electron in new config
    #         is out of the simulation box for set
    #         of lattice vectors along x,y,z.
    #diag=np.array([1.2,0.9,0.8])
    #lattvecs=np.diag(diag)
    # Initial config
    #epos=np.random.random((nconf,3))
    #epos=np.einsum('ij,j->ij',epos,diag)
    # New config
    #step=0.5
    #trans=epos+step*(np.random.random((nconf,3))-0.5*np.ones((nconf,3)))
    #print(epos)
    #print('Attempt:\n',trans)
    #final_epos,wrap = enforce_pbc_orth(lattvecs,epos,trans)
    #test1=np.all(final_epos>np.array([0,0,0])) * np.all(final_epos<=diag)
    #print('Test 1 success:',test1)


    # TEST 2: Check if any electron in new config
    #         is out of the simulation box for set
    #         of lattice vectors not along x,y,z.
    #lattvecs=np.array([[np.sqrt(2)/2,np.sqrt(2)/2,0],[np.sqrt(2)/2,np.sqrt(2)/2,0],[0,0,2]])
    lattvecs=np.array([[1,0.5,0],[-0.5,1,0],[0,0,2]])
    diag,uu=np.linalg.eigh(lattvecs)
    print('diag\n',diag)
    print('uu\n',uu)
    #print(np.einsum('ij,jk->ik',uu,uu))
    #exit()
    # Initial config
    epos=np.random.random((nconf,3))
    epos=np.einsum('ij,j->ij',epos,diag)
    epos=np.einsum('ij,kj->ki',uu,epos)
    print(epos)
    #exit()
    # New config
    step=0.3
    trans=epos+step*(np.random.random((nconf,3))-0.5*np.ones((nconf,3)))
    #print(epos)
    #print('Attempt:\n',trans)
    final_epos,wrap = enforce_pbc_orth(lattvecs,epos,trans)
    
    print(diag)
    print(np.einsum('ij,jk,kl->il',uu.T,lattvecs,uu))
    
    #print(np.einsum('ij,kj->ki',uu.T,final_epos)>np.array([0,0,0]))
    #print(np.einsum('ij,kj->ki',uu.T,final_epos)<=diag)
    print(diag)
    print(epos)
    print('--\n',np.einsum('ij,kj->ki',uu.T,epos))
    print(np.einsum('ij,kj->ki',uu.T,final_epos))

    print(np.all(np.einsum('ij,kj->ki',uu.T,final_epos)>np.array([0,0,0])))
    print(np.all(np.einsum('ij,kj->ki',uu.T,final_epos)<=diag))
    test2=np.all(np.einsum('ij,kj->ki',uu.T,final_epos)>np.array([0,0,0])) * np.all(np.einsum('ij,kj->ki',uu.T,final_epos)<=diag)
    print('Test 2 success:',test2)
    
    assert (test1 * test2)


                  
if __name__=="__main__":
    test_orth()
