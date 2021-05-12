import pyscf
import pyscf.pbc
import pyqmc
import numpy as np
import os


def test_molecule():
    chkname = "test_recover_pyscf.chk"
    mol = pyscf.gto.M(atom="He 0. 0. 0.", basis="bfd_vdz", ecp="bfd", unit="bohr")
    for scf in [pyscf.scf.rhf.RHF, pyscf.scf.uhf.UHF]:
        if os.path.isfile(chkname):
            os.remove(chkname)
        mf = scf(mol)
        mf.chkfile = chkname
        mf.kernel()
        mol2, mf2 = pyqmc.recover_pyscf(chkname)
        print(type(mf2), scf)
        assert isinstance(mf2, scf)


def test_pbc():
    chkname = "test_recover_pyscf.chk"
    mol = pyscf.pbc.gto.M(
        atom="H 0. 0. 0.; H 1. 1. 1.",
        basis="gth-szv",
        pseudo="gth-pade",
        unit="bohr",
        a=(np.ones((3, 3)) - np.eye(3)) * 4,
    )
    for scf in [pyscf.pbc.scf.khf.KRHF, pyscf.pbc.scf.kuhf.KUHF]:
        if os.path.isfile(chkname):
            os.remove(chkname)
        mf = scf(mol, mol.make_kpts((2, 1, 1)))
        mf.chkfile = chkname
        mf.kernel()
        mol2, mf2 = pyqmc.recover_pyscf(chkname)
        print(type(mf2), scf)
        assert isinstance(mf2, scf)


if __name__ == "__main__":
    test_molecule()
    test_pbc()
