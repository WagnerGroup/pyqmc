if __name__ == "__main__":
    import pyscf
    import pyqmc
    import pandas as pd
    mol = pyscf.gto.M(atom = "He 0. 0. 0.", basis='bfd_vdz', ecp='bfd', unit='bohr')

    mf = pyscf.scf.RHF(mol).run()


    wf = pyqmc.slater_jastrow(mol, mf)

    nconfig = 1000
    configs = pyqmc.initial_guess(mol, nconfig)

    acc = pyqmc.gradient_generator(mol, wf, ['wf2acoeff','wf2bcoeff'])
    wf, dfgrad, dfline = pyqmc.line_minimization(wf, configs, acc)
    pd.DataFrame(dfgrad).to_json("optgrad.json")
    pd.DataFrame(dfline).to_json("optline.json")

    dfdmc, configs, weights = pyqmc.rundmc(wf, configs, nsteps = 5000,
           accumulators={'energy': pyqmc.EnergyAccumulator(mol) }, tstep = 0.02 )
    pd.DataFrame(dfdmc).to_json("dmc.json")

