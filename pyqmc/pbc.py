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


    return final_epos, wraparound 



def test_orth():
    nconf=10
    diag=np.array([1.2,0.9,0.8])
    lattvecs=np.diag(diag)
    epos=np.random.random((nconf,3))
    epos=np.einsum('ij,j->ij',epos,diag)
    step=0.3
    trans=epos+step*(np.random.random((nconf,3))-0.5*np.ones((nconf,3)))
    print(epos)
    print(trans)
    #print(trans>np.array([0,0,0]))
    #print(trans<=diag)
    #print(np.all(trans>np.array([0,0,0])),np.all(trans<=diag),np.all(trans>np.array([0,0,0]))*np.all(trans<=diag))
    #exit()
    final_epos,wrap = enforce_pbc_orth(lattvecs,epos,trans)

    assert np.all(final_epos>np.array([0,0,0])) * np.all(final_epos<=diag)


                  
if __name__=="__main__":
    test_orth()
