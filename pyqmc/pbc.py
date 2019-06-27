import numpy as np


def enforce_pbc_orth(lattvecs,epos):
    '''
    Args:

      lattvecs: orthogonal lattice vectors defining 3D torus: (3,3)

      init_epos: attempted new electron coordinates: (nconfig,3)

    Returns:
    
      final_epos: final electron coordinates with PBCs imposed: (nconfig,3)

      wraparound: vector used to bring a given electron back to the simulation cell written in terms of lattvecs: (nconfig,3)
    '''
    # Checks lattice vecs orthogonality. Note: assuming latt vecs are rows, [[v1],[v2],[v3]]=lattvecs.T
    orth=np.einsum('ij,jk->ik',lattvecs,lattvecs.T)
    np.fill_diagonal(orth,0)
    assert np.all(orth==0), "Using enforce_pbc_orth() but lattice vectors are not orthogonal."
    
    # Constructs p.v/|v|^2
    #  p is matrix with electronic positions
    #  v is diag matrix with lattice vecs
    nlattvecs2=np.einsum('ij,i->ij',lattvecs,1/np.linalg.norm(lattvecs,axis=1)**2)
    epos_lvecs_coord=np.einsum('ij,kj->ik',epos,nlattvecs2)

    # Finds position inside box and wraparound vectors (in lattice vector coordinates) 
    tmp=np.divmod(epos_lvecs_coord,1)
    wraparound=tmp[0]
    final_epos=np.einsum('ji,kj->ki',lattvecs,tmp[1])

    return final_epos, wraparound 


