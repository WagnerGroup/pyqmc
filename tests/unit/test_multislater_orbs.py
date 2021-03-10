import pyqmc
import pyqmc.multislater_orbs
import pyscf
import pyscf.hci
import pyqmc.testwf
import pyscf.pbc

def test_molecule_multislater():
    mol = pyscf.gto.M(atom="H 0. 0. 0.; H 0. 0. 1.6; H 0. 0. 3.2; H 0. 0. 4.8", basis="ccpvdz", unit="bohr", verbose=5)
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    cisolver = pyscf.hci.SCI(mol)
    cisolver.select_cutoff=0.1
    nmo = mf.mo_coeff.shape[1]
    nelec = mol.nelec
    h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
    h2 = pyscf.ao2mo.full(mol, mf.mo_coeff)
    e, civec = cisolver.kernel(h1, h2, nmo, nelec, verbose=4)
    cisolver.ci = civec[0]

    nconfig=5
    wf = pyqmc.multislater_orbs.MultiSlater(mol, mf, cisolver)
    configs = pyqmc.initial_guess(mol, nconfig)
    wf.recompute(configs)
    print('gradient', pyqmc.testwf.test_wf_gradient(wf, configs))

    print("laplacian", pyqmc.testwf.test_wf_laplacian(wf, configs))

    print("updateinternals", pyqmc.testwf.test_updateinternals(wf, configs))

    print("gradient_laplacian", pyqmc.testwf.test_wf_gradient_laplacian(wf, configs))

    print("mask", pyqmc.testwf.test_mask(wf, 0, configs.electron(0), [False, False, True, True, True]))

    print("pgradient", pyqmc.testwf.test_wf_pgradient(wf, configs))


def test_molecule_slater():
    mol = pyscf.gto.M(atom="H 0. 0. 0.; H 0. 0. 1.6; H 0. 0. 3.2; H 0. 0. 4.8", basis="ccpvdz", unit="bohr", verbose=5)
    mf = pyscf.scf.UHF(mol)
    mf.kernel()
    nconfig=5
    wf = pyqmc.multislater_orbs.MultiSlater(mol, mf)
    configs = pyqmc.initial_guess(mol, nconfig)
    wf.recompute(configs)
    print('gradient', pyqmc.testwf.test_wf_gradient(wf, configs))

    print("laplacian", pyqmc.testwf.test_wf_laplacian(wf, configs))

    print("updateinternals", pyqmc.testwf.test_updateinternals(wf, configs))

    print("gradient_laplacian", pyqmc.testwf.test_wf_gradient_laplacian(wf, configs))

    print("mask", pyqmc.testwf.test_mask(wf, 0, configs.electron(0), [False, False, True, True, True]))

    print("pgradient", pyqmc.testwf.test_wf_pgradient(wf, configs))



def test_pbc_slater():
    L = 6.4
    mol = pyscf.pbc.gto.M(atom="H 0. 0. 0.; H 0. 0. 1.6; H 0. 0. 3.2; H 0. 0. 4.8", basis='gth-szv', pseudo='gth-pade', unit="bohr", verbose=5,
    a=[[L,0,0],[0,L,0],[0,0,L]])
    mol.exp_to_discard = 0.3
    mol.build()
    mf = pyscf.pbc.scf.KUHF(mol).mix_density_fit()
    mf.kernel()
    nconfig=5
    wf = pyqmc.multislater_orbs.MultiSlater(mol, mf)
    configs = pyqmc.initial_guess(mol, nconfig)
    wf.recompute(configs)
    print('gradient', pyqmc.testwf.test_wf_gradient(wf, configs))

    print("laplacian", pyqmc.testwf.test_wf_laplacian(wf, configs))

    print("updateinternals", pyqmc.testwf.test_updateinternals(wf, configs))

    print("gradient_laplacian", pyqmc.testwf.test_wf_gradient_laplacian(wf, configs))

    print("mask", pyqmc.testwf.test_mask(wf, 0, configs.electron(0), [False, False, True, True, True]))

    print("pgradient", pyqmc.testwf.test_wf_pgradient(wf, configs))



if __name__=="__main__":
    test_pbc_slater()