import pyqmc.orbitals
import pyqmc
import pyscf
import pyscf.pbc
import pyqmc.supercell
import numpy as np 
import os.path

def test_orbitals_all_electron(mol, mf, nconfig=7):
    evaluator = pyqmc.orbitals.PBCOrbitalEvaluatorKpoints.from_mean_field(mol, mf)
    configs = pyqmc.initial_guess(mol, nconfig)
    print(configs.configs.shape)
    aos = evaluator.aos('GTOval_sph',configs)
    aos = aos.reshape(-1,nconfig, configs.configs.shape[1], aos.shape[-1])
    print(aos.shape)
    mos = evaluator.mos(aos,0)
    print(aos.shape, mos.shape)


if __name__=="__main__":
    L = 10
    checkfile = "tmp.chk"
    if not os.path.isfile(checkfile):
        mol = pyscf.pbc.gto.M(atom="H 0. 0. 0.; H 0. 0. 1.6", basis="ccpvdz", unit="bohr", a = [[L,0,0],[0,L,0],[0,0,L]], verbose=5)
        mol.exp_to_discard=0.3
        mol.build()
        kpts = mol.get_kpts([2,2,2])
        mf = pyscf.pbc.scf.KUHF(mol,kpts=kpts).mix_density_fit()
        mf.chkfile = checkfile
        mf.kernel()
    else:
        mol, mf = pyqmc.recover_pyscf(checkfile)
    cell = pyqmc.supercell.get_supercell(mol,np.asarray([[2,0,0],[0,2,0],[0,0,2]]))
    test_orbitals_all_electron(cell ,mf)

    