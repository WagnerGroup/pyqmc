import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import pytest
from pyscf import lib, gto, scf
import pyscf.pbc
import numpy as np

""" 
In this file, we set up several pyscf objects that can be reused across the 
tests. Try to use one of these fixtures if at all possible, so our
tests don't have to keep running pyscf.
"""

@pytest.fixture(scope="module")
def LiH_sto3g_rhf():
    mol = gto.M(atom="Li 0. 0. 0.; H 0. 0. 1.5", basis="sto-3g", unit="bohr")
    mf = scf.RHF(mol).run()
    return mol, mf
    #mf_rohf = scf.ROHF(mol).run()
    #mf_uhf = scf.UHF(mol).run()


@pytest.fixture(scope="module")
def LiH_sto3g_uhf():
    mol = gto.M(atom="Li 0. 0. 0.; H 0. 0. 1.5", basis="sto-3g", unit="bohr")
    mf = scf.UHF(mol).run()
    return mol, mf

@pytest.fixture(scope="module")
def H2_ccecp_rhf():
    r = 1.54 / 0.529177
    mol = gto.M(
        atom="H 0. 0. 0.; H 0. 0. %g" % r,
        ecp="ccecp",
        basis="ccecpccpvdz",
        unit="bohr",
        verbose=1,
    )
    mf = scf.RHF(mol).run()
    return mol, mf

@pytest.fixture(scope="module")
def H2_ccecp_uhf():
    r = 1.54 / 0.529177
    mol = gto.M(
        atom="H 0. 0. 0.; H 0. 0. %g" % r,
        ecp="ccecp",
        basis="ccecpccpvdz",
        unit="bohr",
        verbose=1,
    )
    mf = scf.UHF(mol).run()
    return mol, mf

@pytest.fixture(scope="module")
def C2_ccecp_rhf():
    mol = gto.M(
                atom="""C 0 0 0 
                C 1 0 0  """,
                ecp="ccecp",
                basis="ccecpccpvdz",
                )
    mf = scf.RHF(mol).run()
    return mol, mf



@pytest.fixture(scope='module')
def H_pbc_sto3g_krks():
    mol = pyscf.pbc.gto.M(
        atom="H 0. 0. 0.; H 1. 1. 1.",
        basis="sto-3g",
        unit="bohr",
        a=(np.ones((3, 3)) - np.eye(3)) * 4,
    )
    mf = pyscf.pbc.scf.KRKS(mol, mol.make_kpts((2, 2, 2))).run()
    return mol, mf
