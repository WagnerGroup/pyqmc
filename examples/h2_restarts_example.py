# How to use restarts with PyQMC
from pyscf import gto, scf
import pyqmc.recipes


def run_mf():
    mol = gto.M(
        atom="H 0. 0. 0.; H 0. 0. 1.4", ecp="ccecp", basis="ccecp-ccpvtz", unit="bohr"
    )
    mf = scf.RHF(mol)
    mf.chkfile = "h2.chk"
    mf.kernel()
    return mf.chkfile


def run_restarts(ntimes=5):
    chkfile = run_mf()
    common_args = dict(
        nconfig=100,
        verbose=True,
        jastrow_kws=dict(na=2),
    )
    for i in range(ntimes):
        pyqmc.recipes.OPTIMIZE(
            chkfile,
            f"h2_opt{i}.hdf",
            load_parameters=f"h2_opt{i-1}.hdf" if i > 0 else None,
            slater_kws=dict(optimize_orbitals=True, optimize_zeros=False),
            max_iterations=2 * (i+1),
            **common_args,
        )
    for i in range(ntimes):
        print(i)
        pyqmc.recipes.VMC(
            chkfile,
            f"h2_vmc{i}.hdf",
            load_parameters=f"h2_opt{ntimes-1}.hdf",
            restart_from=f"h2_vmc{i-1}.hdf" if i > 0 else None,
            **common_args,
            nblocks=2 * (i+1),
        )
    for i in range(ntimes):
        print(i)
        pyqmc.recipes.DMC(
            chkfile,
            f"h2_dmc{i}.hdf",
            load_parameters=f"h2_opt{ntimes-1}.hdf",
            restart_from=f"h2_dmc{i-1}.hdf" if i > 0 else None,
            **common_args,
            nblocks=10 * (i+1),
        )


def extend_hdf(f, source):
    for k, it in source.items():
        if k in ["wf", "configs", "weights"]:
            continue
        currshape = f[k].shape
        f[k].resize((currshape[0] + it.shape[0], *currshape[1:]))
        f[k][currshape[0] :] = it


def copy_configs(f, source):
    for k in ["configs", "weights"]:
        if k in source.keys():
            f[k][...] = source[k]


def gather(sourcefile, nfiles=5):
    import h5py
    import shutil

    allfile = sourcefile.format("_all")
    shutil.copyfile(sourcefile.format(0), allfile)
    with h5py.File(allfile, "a") as hdf:
        for i in range(1, nfiles):
            with h5py.File(sourcefile.format(i), "r") as source:
                extend_hdf(hdf, source)
                if i == nfiles - 1:
                    copy_configs(hdf, source)


if __name__ == "__main__":
    n = 5
    run_restarts(ntimes=n)
    gather("h2_opt{0}.hdf", nfiles=n)
    gather("h2_vmc{0}.hdf", nfiles=n)
    gather("h2_dmc{0}.hdf", nfiles=n)
