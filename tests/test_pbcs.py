import numpy as np
from pyqmc.pbc import enforce_pbc_orth


def test_orth_lattvecs():
    nconf=6
    
    # TEST 1: Check if any electron in new config
    #         is out of the simulation box for set
    #         of lattice vectors along x,y,z.
    diag=np.array([1.2,0.9,0.8])
    lattvecs=np.diag(diag)
    # Initial config
    epos=np.random.random((nconf,3))
    epos=np.einsum('ij,j->ij',epos,diag)
    # New config
    step=0.5
    trans=epos+step*(np.random.random((nconf,3))-0.5*np.ones((nconf,3)))
    final_epos,wrap = enforce_pbc_orth(lattvecs,epos,trans)
    test1=np.all(final_epos>np.array([0,0,0])) * np.all(final_epos<=diag)
    print('Test 1 success:',test1)


    # TEST 2: Check if any electron in new config
    #         is out of the simulation box for set
    #         of lattice vectors not along ex,ey,ez.
    #         We do controlled comparison between
    #         initial configs and final ones.
    lattvecs=np.array([[1,-0.75,0],[0.5,2/3.,0],[0,0,2]])
    epos=np.zeros((6,3))
    trans=np.array([[0.1,0.1,0.1],[1.1,-0.825,0.2],[0.525,0.7,0.],[1.8,-0.1,0.],[0.2,0.2,-0.2],[-0.375,0.125,-0.1]])
    check_final=np.array([[0.1,0.1,0.1],[0.1,-0.075,0.2],[0.025,1/30,0.],[0.3,-1/60,0.],[0.2,0.2,1.8],[1.125,1/24,1.9]])
    check_wrap=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,-1],[-1,-1,-1]])
    final_trans,final_wrap = enforce_pbc_orth(lattvecs,epos,trans)
    # Checks output configs
    relative_tol=1e-10
    absolute_tol=1e-14
    test2a = np.all(np.isclose(final_trans,check_final,rtol=relative_tol,atol=absolute_tol))
    # Checks wraparound matrix
    test2b = np.all(np.isclose(final_wrap,check_wrap,rtol=relative_tol,atol=absolute_tol))
    test2 = test2a * test2b
    print('Test 2 success:',test2)

    
    # TEST 3: Check if any electron in new config
    #         is out of the simulation box for set
    #         of lattice vectors not along ex,ey,ez.
    nconf=50
    lattvecs=np.array([[np.sqrt(2)/2,-np.sqrt(2)/2,0],[1,1,0],[0,0,2]])    
    diag=np.dot(lattvecs,lattvecs.T)
    rot=np.linalg.inv(lattvecs.T)
    # Old config
    epos=np.random.random((nconf,3))
    epos=np.einsum('ij,j->ij',epos,np.diagonal(diag))
    epos=np.einsum('ij,kj->ki',rot,epos)
    # New config
    step=0.5
    trans=epos+step*(np.random.random((nconf,3))-0.5*np.ones((nconf,3)))
    final_trans,wrap = enforce_pbc_orth(lattvecs,epos,trans)
    # Configs in lattice vectors basis
    nlv2=np.einsum('ij,i->ij',lattvecs,1/np.linalg.norm(lattvecs,axis=1)**2)
    aa=np.einsum('ij,kj->ik',final_trans,nlv2)
    test3 = np.all(aa<1) & np.all(aa>=0)
    print('Test 3 success:',test2)
    
    assert (test1 * test2 * test3)


    
                  
if __name__=="__main__":
    test_orth_lattvecs()
