if __name__ == "__main__":
    import pyscf
    import pyqmc
    import pyqmc.recipes

    mol = pyscf.gto.M(
        atom="He 0. 0. 0.", basis="ccECP_cc-pVDZ", ecp="ccecp", unit="bohr"
    )

    mf = pyscf.scf.RHF(mol)
    mf.chkfile = "he_dft.hdf5"
    mf.kernel()
    jastrow_kws = {}
    slater_kws = {"optimize_orbitals": True}

    pyqmc.recipes.OPTIMIZE(
        "he_dft.hdf5", "he_sj.hdf5", jastrow_kws=jastrow_kws, slater_kws=slater_kws
    )

    pyqmc.recipes.VMC(
        "he_dft.hdf5",
        "he_sj_vmc.hdf5",
        load_parameters="he_sj.hdf5",
        accumulators={"rdm1": True},
        jastrow_kws=jastrow_kws,
        slater_kws=slater_kws,
        **{"nblocks": 40},
    )

    pyqmc.recipes.DMC(
        "he_dft.hdf5",
        "he_sj_dmc.hdf5",
        load_parameters="he_sj.hdf5",
        accumulators={"rdm1": True},
        jastrow_kws=jastrow_kws,
        slater_kws=slater_kws,
        **{"nblocks": 4000, "tstep": 0.02},
    )
