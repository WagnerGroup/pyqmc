if __name__ == "__main__":
    import pyscf
    import pyqmc
    import pandas as pd

    mol = pyscf.gto.M(atom="He 0. 0. 0.", basis="bfd_vdz", ecp="bfd", unit="bohr")

    mf = pyscf.scf.RHF(mol).run()
    wf, to_opt = pyqmc.default_sj(mol, mf)

    nconfig = 1000
    configs = pyqmc.initial_guess(mol, nconfig)

    acc = pyqmc.gradient_generator(mol, wf, to_opt)
    pyqmc.line_minimization(wf, configs, acc, hdf_file="he_opt.hdf5", verbose=True)

    pyqmc.rundmc(
        wf,
        configs,
        nsteps=50,  # 5000,
        accumulators={"energy": pyqmc.EnergyAccumulator(mol)},
        tstep=0.02,
        hdf_file="he_dmc.hdf5",
        verbose=True,
    )
