import numpy as np
from pyqmc.pbc import enforce_pbc_orth, enforce_pbc


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
    final_epos,wrap = enforce_pbc_orth(lattvecs,trans)
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
    final_trans,final_wrap = enforce_pbc_orth(lattvecs,trans)
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
    final_trans,wrap = enforce_pbc_orth(lattvecs,trans)
    # Configs in lattice vectors basis
    nlv2=np.einsum('ij,i->ij',lattvecs,1/np.linalg.norm(lattvecs,axis=1)**2)
    aa=np.einsum('ij,kj->ik',final_trans,nlv2)
    test3 = np.all(aa<1) & np.all(aa>=0)
    print('Test 3 success:',test3)
    
    assert (test1 * test2 * test3)



def test_enforce_pbcs():
        
    # TEST 1: Check if any electron in new config
    #         is out of the simulation box for set
    #         of non-orthogonal lattice vectors. We
    #         do a controlled comparison between
    #         initial configs and final ones.
    nconf=7
    lattvecs=np.array([[1.2,0,0],[0.6,1.2*np.sqrt(3)/2,0],[0,0,0.8]]) # Triangular lattice
    trans=np.array([[0.1,0.1,0.1],[1.3,0,0.2],[0.9,1.8*np.sqrt(3)/2,0],[0,0,1.1],[2.34,1.35099963,0],[0.48,1.24707658,0],[-2.52,2.28630707,-0.32]])
    check_final=np.array([[0.1,0.1,0.1],[0.1,0,0.2],[0.3,0.6*np.sqrt(3)/2,0.],[0,0,0.3],[0.54,0.31176915,0],[1.08,0.2078461,0],[1.08,0.2078461,0.48]])
    check_wrap=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[-1,1,0],[-4,2,-1]])
    final_trans,final_wrap = enforce_pbc(lattvecs,trans)
    # Checks output configs 
    relative_tol=1e-8
    absolute_tol=1e-8
    test1a = np.all(np.isclose(final_trans,check_final,rtol=relative_tol,atol=absolute_tol))
    # Checks wraparound matrix
    test1b = np.all(np.isclose(final_wrap,check_wrap,rtol=relative_tol,atol=absolute_tol))
    test1 = test1a * test1b
    print('Test 1 success:',test1)

    
    # TEST 2: Check if any electron in new config
    #         is out of the simulation box for set
    #         of non-orthogonal lattice vectors.
    nconf=50
    lattvecs=np.array([[1.2,0,0],[0.6,1.2*np.sqrt(3)/2,0],[0,0,0.8]]) # Triangular lattice
    recpvecs=np.linalg.inv(lattvecs)
    # Old config
    epos=np.random.random((nconf,3))
    epos=np.einsum('ij,jk->ik',epos,lattvecs)
    # New config
    step=0.5
    trans=epos+step*(np.random.random((nconf,3))-0.5*np.ones((nconf,3)))
    final_trans,wrap = enforce_pbc(lattvecs,trans)
    # Configs in lattice vectors basis
    ff=np.einsum('ij,jk->ik',final_trans,recpvecs)
    test2 = np.all(ff<1) & np.all(ff>=0)
    print('Test 2 success:',test2)

    assert (test1 * test2)


    
def test_compare_enforcements():
    import time
    nconf=500000
    lattvecs=np.array([[1,-0.75,0],[0.5,2/3.,0],[0,0,2]])
    epos0=np.random.random((nconf,3))
    epos0=np.einsum('ij,jk->ik',epos0,lattvecs)

    # Using enforce_pbc_orth()
    t1=time.time()
    epos=epos0.copy()
    final_epos_o, wrap_o = enforce_pbc_orth(lattvecs,epos)
    t2=time.time()
    # Using enforce_pbc()
    epos=epos0.copy()  
    final_epos_no, wrap_no = enforce_pbc(lattvecs,epos)
    t3=time.time()

    relative_tol=1e-8
    absolute_tol=1e-8
    test1 = np.all(np.isclose(final_epos_o,final_epos_no,rtol=relative_tol,atol=absolute_tol))
    test2 = np.all(np.isclose(wrap_o,wrap_no,rtol=relative_tol,atol=absolute_tol))

    print('Same result for enforce_pbc_orth() and enforce_pbc()?',test1*test2)
    print('time[enforce_pbc_orth()] = %f'%(t2-t1))
    print('time[enforce_pbc()]      = %f'%(t3-t2))
    assert (test1 * test2)

    

if __name__=="__main__":
    test_orth_lattvecs()
    test_enforce_pbcs()
    test_compare_enforcements()
