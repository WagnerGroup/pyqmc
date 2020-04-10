import numpy as np
import pyqmc
import pandas as pd
from pyscf.pbc import gto, scf
from pyqmc.reblock import reblock
from pyqmc.slaterpbc import get_supercell
from pyscf.pbc.dft.multigrid import multigrid
from pyscf.scf.addons import remove_linear_dep_
import time


def cubic_with_ecp(kind=0, nk=(1, 1, 1)):
    from pyscf.pbc.dft.multigrid import multigrid

    start = time.time()
    L = 6.63
    mol = gto.Cell(
        atom="""Li     {0}      {0}      {0}                
                  Li     {1}      {1}      {1}""".format(
            0.0, L / 2
        ),
        basis="bfd-vdz",
        ecp="bfd",
        spin=0,
        unit="bohr",
    )
    mol.exp_to_discard = 0.1
    mol.build(a=np.eye(3) * L)
    kpts = mol.make_kpts(nk)
    mf = scf.KUKS(mol, kpts)
    mf.xc = "pbe"
    # mf = mf.density_fit()
    mf = multigrid(mf)
    mf = mf.run()
    supercell = get_supercell(mol, np.diag(nk))
    runtest(supercell, mf, kind=kind)


def multislater(kind=0, nk=(1, 1, 1)):
    L = 3
    mol = gto.Cell(
        atom="""H     {0}      {0}      {0}                
                  H     {1}      {1}      {1}""".format(
            0.0, L / 2
        ),
        basis="cc-pvtz",
        spin=0,
        unit="bohr",
    )
    mol.exp_to_discard = 0.1
    mol.build(a=np.eye(3) * L)
    kpts = mol.make_kpts(nk)
    mf = scf.UKS(mol, (0, 0, 0))
    mf.xc = "pbe"
    mf = multigrid(mf)
    mf = remove_linear_dep_(mf)
    mf.chkfile = "h_bcc.chkfile"
    mf = mf.run()

    supercell = get_supercell(mol, np.diag(nk))
    runtest(supercell, mf, kind=kind, do_mc=True)


def test_RKS(kind=0, nk=(1, 1, 1)):
    L = 2
    mol = gto.M(
        atom="""He     {0}      {0}      {0}""".format(0.0),
        basis="sto-3g",
        a=np.eye(3) * L,
        unit="bohr",
    )
    kpts = mol.make_kpts(nk)
    mf = scf.KRKS(mol, kpts)
    mf.xc = "pbe"
    # mf = mf.density_fit()
    mf = mf.run()

    supercell = get_supercell(mol, np.diag(nk))
    runtest(supercell, mf, kind=kind)


def noncubic(kind=0, nk=(1, 1, 1)):
    L = 3
    mol = gto.M(
        atom="""H     {0}      {0}      {0}                
                  H     {1}      {1}      {1}""".format(
            0.0, L / 4
        ),
        basis="sto-3g",
        a=(np.ones((3, 3)) - np.eye(3)) * L / 2,
        spin=0,
        unit="bohr",
    )
    kpts = mol.make_kpts(nk)
    mf = scf.KUKS(mol, kpts)
    mf.xc = "pbe"
    # mf = mf.density_fit()
    mf = mf.run()
    supercell = get_supercell(mol, np.diag(nk))
    runtest(supercell, mf, kind=kind)


def runtest(mol, mf, kind=0, do_mc=False):
    if do_mc:
        from pyscf import mcscf

        mc = mcscf.CASCI(mf, ncas=4, nelecas=(1, 1))
        mc.kernel()
        wf = pyqmc.default_msj(mol, mf, mc)[0]
        kpt = mf.kpt
        dm = mc.make_rdm1()
        if len(dm.shape) == 4:
            dm = np.sum(dm, axis=0)
    else:
        kpt = mf.kpts[kind]
        wf = pyqmc.PySCFSlaterPBC(mol, mf)
        dm = mf.make_rdm1()
        print("original dm shape", dm.shape)
        if len(dm.shape) == 4:
            dm = np.sum(dm, axis=0)
        dm = dm[kind]

    #####################################
    ## evaluate KE in PySCF
    #####################################
    ke_mat = mol.pbc_intor("int1e_kin", hermi=1, kpts=np.array(kpt))
    print("ke_mat", ke_mat.shape)
    print("dm", dm.shape)
    pyscfke = np.real(np.einsum("ij,ji->", ke_mat, dm))
    print("PySCF kinetic energy: {0}".format(pyscfke))

    #####################################
    ## evaluate KE integral with VMC
    #####################################
    coords = pyqmc.initial_guess(mol, 1200, 0.7)
    warmup = 10
    start = time.time()
    df, coords = pyqmc.vmc(
        wf,
        coords,
        nsteps=100 + warmup,
        tstep=1,
        accumulators={"energy": pyqmc.accumulators.EnergyAccumulator(mol)},
        verbose=False,
    )
    print("VMC time", time.time() - start)
    df = pd.DataFrame(df)
    dfke = reblock(df["energyke"][warmup:], 10)
    dfke /= mol.scale
    vmcke, err = dfke.mean(), dfke.sem()
    print("VMC kinetic energy: {0} +- {1}".format(vmcke, err))

    assert (
        np.abs(vmcke - pyscfke) < 5 * err
    ), "energy diff not within 5 sigma ({0:.6f}): energies \n{1} \n{2}".format(
        5 * err, vmcke, pyscfke
    )


if __name__ == "__main__":
    kind = 0
    nk = [1, 1, 1]
    # multislater(kind, nk)
    cubic_with_ecp(kind, nk)
    test_RKS(kind, nk)
    # noncubic(kind, nk)
