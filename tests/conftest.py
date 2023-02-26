import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import pytest
from pyscf import lib, gto, scf
import pyscf.pbc
import numpy as np
#import pyscf.hci

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
def H2_ccecp_hci(H2_ccecp_rhf):
    mol, mf = H2_ccecp_rhf

    cisolver = pyscf.hci.SCI(mol)
    cisolver.select_cutoff = 0.1
    nmo = mf.mo_coeff.shape[1]
    nelec = mol.nelec
    h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
    h2 = pyscf.ao2mo.full(mol, mf.mo_coeff)
    e, civec = cisolver.kernel(h1, h2, nmo, nelec, verbose=4)
    cisolver.ci = civec[0]
    cisolver.energy = e +  mf.energy_nuc()

    return mol, mf, cisolver


@pytest.fixture(scope="module")
def H2_ccecp_casci_s0(H2_ccecp_rhf):
    mol, mf = H2_ccecp_rhf
    mc = pyscf.mcscf.CASCI(mf, ncas=4, nelecas=(1, 1))
    mc.kernel()
    return mol, mf, mc


@pytest.fixture(scope="module")
def H2_ccecp_casci_s2(H2_ccecp_uhf):
    mol, mf = H2_ccecp_uhf
    mc = pyscf.mcscf.CASCI(mf, ncas=4, nelecas=(2, 0))
    mc.kernel()
    return mol, mf, mc


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
def H2_casci():
    mol = gto.M(atom="H 0. 0. 0.0; H 0. 0. 2.4",
            basis=f"ccpvtz",  
            unit="bohr", 
            charge=0, 
            spin=0, 
            verbose=1)  
    mf = scf.ROHF(mol).run()
    mc = pyscf.mcscf.CASCI(mf, 2, 2)
    mc.fcisolver.nroots = 4
    mc.kernel()
    return mol, mf, mc

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


@pytest.fixture(scope="module")
def C_ccecp_rohf():
    mol = gto.M(
                atom="""C 0 0 0 
                C 1 0 0  """,
                ecp="ccecp",
                basis="ccecpccpvdz",
                spin=2
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


@pytest.fixture(scope='module')
def H_pbc_sto3g_kuks():
    mol = pyscf.pbc.gto.M(
        atom="H 0. 0. 0.; H 1. 1. 1.",
        basis="sto-3g",
        unit="bohr",
        a=(np.ones((3, 3)) - np.eye(3)) * 4,
    )
    mf = pyscf.pbc.scf.KUKS(mol, mol.make_kpts((2, 2, 2))).run()
    return mol, mf


@pytest.fixture(scope='module')
def li_cubic_ccecp():
    nk = (2,2,2)
    L = 6.63 * 2
    cell = pyscf.pbc.gto.Cell(
        atom="""Li     {0}      {0}      {0}                
                  Li     {1}      {1}      {1}""".format(
            0.0, L / 4
        ),
        basis="ccecpccpvdz",
        ecp={"Li": "ccecp"},
        spin=0,
        unit="bohr",
    )
    cell.exp_to_discard = 0.1
    cell.build(a=np.eye(3) * L)
    kpts = cell.make_kpts(nk)
    mf = pyscf.pbc.scf.KRKS(cell, kpts)
    mf.xc = "pbe"
    #mf = mf.density_fit()
    #mf = pyscf.pbc.dft.multigrid.multigrid(mf)
    mf = mf.run()
    return cell, mf


@pytest.fixture(scope='module')
def diamond_primitive():
    cell = pyscf.pbc.gto.Cell()
    cell.verbose = 5
    cell.atom=[
        ['C', np.array([0., 0., 0.])], 
        ['C', np.array([0.8925, 0.8925, 0.8925])]
               ]
    cell.a=[[0.0, 1.785, 1.785], 
            [1.785, 0.0, 1.785], 
            [1.785, 1.785, 0.0]]
    cell.basis = 'ccecpccpvdz'
    cell.ecp = 'ccecp'
    cell.exp_to_discard=0.3
    cell.build()
    kpts = cell.make_kpts((2,2,2))
    mf=pyscf.pbc.dft.KRKS(cell, kpts)

    mf.xc='lda,vwn'

    mf.kernel()
    return cell, mf


@pytest.fixture(scope='module')
def h_noncubic_sto3g_triplet():
    nk = (1,1,1)
    L = 8
    mol = pyscf.pbc.gto.M(
        atom="""H     {0}      {0}      {0}                
                  H     {1}      {1}      {1}""".format(
            0.0, L / 4
        ),
        basis="sto-3g",
        a=(np.ones((3, 3)) - np.eye(3)) * L / 2,
        spin=2*np.prod(nk),
        unit="bohr",
    )
    kpts = mol.make_kpts(nk)
    mf = pyscf.pbc.scf.KUKS(mol, kpts)
    mf.xc = "pbe"
    #mf = pyscf.pbc.dft.multigrid.multigrid(mf)
    mf = mf.run()
    print(mf.mo_occ)
    return mol, mf
