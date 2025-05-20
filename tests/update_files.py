import pyscf 
import numpy as np
import pyscf.pbc
"""
Set up some pyscf objects that can be reused.
The intention here is to set up a few objects that can be reused across the tests, so we don't have to keep rerunning pyscf for every test.
"""


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
    mf.chkfile = "files/li_cubic_ccecp.hdf5"
    #mf = mf.density_fit()
    #mf = pyscf.pbc.dft.multigrid.multigrid(mf)
    mf = mf.run()
    return cell, mf


def diamond_primitive():
    cell = pyscf.pbc.gto.Cell()
    cell.verbose = 5
    cell.atom=[
        ['C', np.array([0., 0., 0.])], 
        ['C', np.array([0.8917, 0.8917, 0.8917])]
               ]
    cell.a=[[0.0, 1.7834, 1.7834], 
            [1.7834, 0.0, 1.7834], 
            [1.7834, 1.7834, 0.0]]
    cell.basis = 'ccecpccpvdz'
    cell.ecp = 'ccecp'
    cell.exp_to_discard=0.3
    cell.build()
    kpts = cell.make_kpts((2,2,2))
    mf=pyscf.pbc.dft.KRKS(cell, kpts)

    mf.xc='lda,vwn'
    mf.chkfile = "files/diamond_primitive.hdf5"

    mf.kernel()
    return cell, mf



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
    mf.chkfile = "files/h_noncubic_sto3g_triplet.hdf5"
    #mf = pyscf.pbc.dft.multigrid.multigrid(mf)
    mf = mf.run()
    print(mf.mo_occ)
    return mol, mf



def h_pbc_casscf():
    L = 8
    mol = pyscf.pbc.gto.M(
        atom="""H     {0}      {0}      {0}                
                H     {1}      {1}      {1}""".format(
            0.0, L / 4
        ),
        basis="ccpvdz",
        a= np.eye(3) * L,
        spin=0,
        unit="bohr",
        precision=1e-6,
    )
    mf = pyscf.pbc.scf.RKS(mol)
    mf.xc = "pbe"
    #mf = pyscf.pbc.dft.multigrid.multigrid(mf)
    mf.chkfile = "files/h_pbc_casscf.hdf5"
    mf = mf.run()



if __name__ == "__main__":
    print("lithium cubic")
    li_cubic_ccecp()
    print("diamond primitive")
    diamond_primitive()
    print("h_noncubic_sto3g_triplet")
    h_noncubic_sto3g_triplet()
    print("h_pbc_casscf")
    h_pbc_casscf()
