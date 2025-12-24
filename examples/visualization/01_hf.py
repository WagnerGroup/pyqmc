import numpy
import h5py
from pyscf import gto, scf, lo, tools, mcscf, hci, ao2mo

benzene = [[ 'C'  , ( 4.673795 ,   6.280948 , 0.00  ) ],
           [ 'C'  , ( 5.901190 ,   5.572311 , 0.00  ) ],
           [ 'C'  , ( 5.901190 ,   4.155037 , 0.00  ) ],
           [ 'C'  , ( 4.673795 ,   3.446400 , 0.00  ) ],
           [ 'C'  , ( 3.446400 ,   4.155037 , 0.00  ) ],
           [ 'C'  , ( 3.446400 ,   5.572311 , 0.00  ) ],
           [ 'H'  , ( 4.673795 ,   7.376888 , 0.00  ) ],
           [ 'H'  , ( 6.850301 ,   6.120281 , 0.00  ) ],
           [ 'H'  , ( 6.850301 ,   3.607068 , 0.00  ) ],
           [ 'H'  , ( 4.673795 ,   2.350461 , 0.00  ) ],
           [ 'H'  , ( 2.497289 ,   3.607068 , 0.00  ) ],
           [ 'H'  , ( 2.497289 ,   6.120281 , 0.00  ) ]]

mol = gto.M(atom=benzene,
            basis='ccecp-ccpvtz', ecp='ccecp',
            verbose=5)

mf = scf.RHF(mol)
mf_chkfile = 'benzene.hdf5'
mf.chkfile = mf_chkfile
mf.kernel()


