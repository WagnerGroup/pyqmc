# You must run using if __name__ == "__main__" when using mpi4py
if __name__ == "__main__":
    import pyscf
    import pyqmc
    import pyqmc.recipes
    import mpi4py.futures

    mol = pyscf.gto.M(
        atom="He 0. 0. 0.", basis="ccECP_cc-pVDZ", ecp="ccecp", unit="bohr"
    )

    mf = pyscf.scf.RHF(mol)
    mf.chkfile = "he_dft.hdf5"
    mf.kernel()
    jastrow_kws = {}
    slater_kws = {"optimize_orbitals": True}

    npartitions = 2
    with mpi4py.futures.MPIPoolExecutor(max_workers=npartitions) as client:
        pyqmc.recipes.OPTIMIZE(
            "he_dft.hdf5",
            "he_sj.hdf5",
            jastrow_kws=jastrow_kws,
            slater_kws=slater_kws,
            client=client,
            npartitions=npartitions,
        )

        pyqmc.recipes.VMC(
            "he_dft.hdf5",
            "he_sj_vmc.hdf5",
            load_parameters="he_sj.hdf5",
            accumulators={"rdm1": True},
            jastrow_kws=jastrow_kws,
            slater_kws=slater_kws,
            client=client,
            npartitions=npartitions,
            **{"nblocks": 40},
        )

        pyqmc.recipes.DMC(
            "he_dft.hdf5",
            "he_sj_dmc.hdf5",
            load_parameters="he_sj.hdf5",
            accumulators={"rdm1": True},
            jastrow_kws=jastrow_kws,
            slater_kws=slater_kws,
            client=client,
            npartitions=npartitions,
            **{"nblocks": 4000, "tstep": 0.02},
        )
