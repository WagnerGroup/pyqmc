import numpy
import h5py
from pyscf import gto, scf, lo, tools, mcscf, hci, ao2mo, lib
import pyqmc.api as pyq


mol, mf = pyq.recover_pyscf("benzene.hdf5", cancel_outputs=False)

cisolver = hci.SCI(mol)
cutoff = 0.01
cisolver.select_cutoff=cutoff
cisolver.nroots=12
nmo = 36
mo_coeff = mf.mo_coeff[:,:nmo]
nelec = mol.nelec
h1 = mo_coeff.T.dot(mf.get_hcore()).dot(mo_coeff)
h2 = ao2mo.full(mol, mo_coeff)
e, civec = cisolver.kernel(h1, h2, nmo, nelec)
lib.chkfile.save('benzene_hci.hdf5','ci',
                 {'ci':numpy.array(civec),
                  'nmo':nmo,
                  'nelec':nelec,
                  '_strs':cisolver._strs,
                  'select_cutoff':cutoff,
                  'energy':e+mol.energy_nuc(),
                  })
