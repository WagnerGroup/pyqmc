if __name__ == "__main__":
    import pyscf
    import pyqmc.api as pyq

    mol = pyscf.gto.M(atom="He 0. 0. 0.", basis="bfd_vdz", ecp="bfd", unit="bohr")

    mf = pyscf.scf.RHF(mol).run()
    wf, to_opt = pyq.generate_wf(mol, mf)

    nconfig = 1000
    configs = pyq.initial_guess(mol, nconfig)

    acc = pyq.gradient_generator(mol, wf, to_opt)
    pyq.line_minimization(wf, configs, acc,  verbose=True, max_iterations=10)
    quit()
    pyq.rundmc(
        wf,
        configs,
        nblocks=5000,
        accumulators={"energy": pyq.EnergyAccumulator(mol)},
        tstep=0.02,
        hdf_file="he_dmc.hdf5",
        verbose=True,
    )
