# MIT License
# 
# Copyright (c) 2019-2024 The PyQMC Developers
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

import pyscf
import pyscf.pbc
import pyqmc.api as pyq
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
        mol2, mf2 = pyq.recover_pyscf(chkname)
        print(type(mf2), scf)
        assert isinstance(mf2, scf)
    os.remove(chkname)


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
        mol2, mf2 = pyq.recover_pyscf(chkname)
        print(type(mf2), scf)
        assert isinstance(mf2, scf)
    os.remove(chkname)


if __name__ == "__main__":
    test_molecule()
    test_pbc()
