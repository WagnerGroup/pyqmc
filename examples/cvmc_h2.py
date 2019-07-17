import pyqmc


def setuph2(r, obdm_steps=5):
    from pyscf import gto, scf, lo
    from pyqmc.accumulators import LinearTransform, EnergyAccumulator
    from pyqmc.obdm import OBDMAccumulator
    from pyqmc.cvmc import DescriptorFromOBDM, PGradDescriptor

    import itertools

    # ccECP from A. Annaberdiyev et al. Journal of Chemical Physics 149, 134108 (2018)
    basis = {
        "H": gto.basis.parse(
            """
    H S
23.843185 0.00411490
10.212443 0.01046440
4.374164 0.02801110
1.873529 0.07588620
0.802465 0.18210620
0.343709 0.34852140
0.147217 0.37823130
0.063055 0.11642410
"""
        )
    }
    """
H S
0.040680 1.00000000
H S
0.139013 1.00000000
H P
0.166430 1.00000000
H P
0.740212 1.00000000
"""
    ecp = {
        "H": gto.basis.parse_ecp(
            """
    H nelec 0
H ul
1 21.24359508259891 1.00000000000000
3 21.24359508259891 21.24359508259891
2 21.77696655044365 -10.85192405303825
"""
        )
    }

    mol = gto.M(
        atom=f"H 0. 0. 0.; H 0. 0. {r}", unit="bohr", basis=basis, ecp=ecp, verbose=5
    )
    mf = scf.RHF(mol).run()
    # lowdin = lo.orth_ao(mol, "lowdin")
    mo_occ = mf.mo_coeff[:, mf.mo_occ > 0]
    a = lo.iao.iao(mol, mo_occ)
    a = lo.vec_lowdin(a, mf.get_ovlp())

    obdm_up = OBDMAccumulator(mol=mol, orb_coeff=a, nstep=obdm_steps, spin=0)
    obdm_down = OBDMAccumulator(mol=mol, orb_coeff=a, nstep=obdm_steps, spin=1)

    wf = pyqmc.slater_jastrow(mol, mf)

    descriptors = {
        "t": [[(1.0, (0, 1)), (1.0, (1, 0))], [(1.0, (0, 1)), (1.0, (1, 0))]],
        "trace": [[(1.0, (0, 0)), (1.0, (1, 1))], [(1.0, (0, 0)), (1.0, (1, 1))]],
    }
    for i in [0, 1]:
        descriptors[f"nup{i}"] = [[(1.0, (i, i))], []]
        descriptors[f"ndown{i}"] = [[], [(1.0, (i, i))]]

    acc = PGradDescriptor(
        EnergyAccumulator(mol),
        LinearTransform(wf.parameters),
        [obdm_up, obdm_down],
        DescriptorFromOBDM(descriptors, norm=2.0),
    )

    return {"wf": wf, "acc": acc, "mol": mol, "mf": mf, "descriptors": descriptors}

    # configs = pyqmc.initial_guess(mol, nconf)
    # wf,df=optimize(wf,configs,acc,obj,forcing=forcing,
    #        iters=10,tstep=0.1,datafile=datafile)
    # print('final parameters',wf.parameters)
    # return df,wf.parameters


if __name__ == "__main__":
    import parsl.config
    from parsl.configs.exex_local import config
    from pyqmc.parsltools import distvmc as vmc
    from pyqmc.parsltools import clean_pyscf_objects
    from pyqmc.linemin import line_minimization
    from pyqmc.cvmc import optimize

    # run VMC in parallel
    r = 1.1

    ncore = 4
    sys = setuph2(r)
    sys["mol"], sys["mf"] = clean_pyscf_objects(sys["mol"], sys["mf"])

    config.executors[0].ranks_per_node = ncore
    parsl.load(config)

    # Set up calculation and
    ncore = 2
    nconf = 800
    configs = pyqmc.initial_guess(sys["mol"], nconf)
    df, configs = vmc(
        sys["wf"], configs, accumulators={}, nsteps=40, nsteps_per=40, npartitions=ncore
    )

    wf, df = line_minimization(
        sys["wf"],
        configs,
        pyqmc.gradient_generator(sys["mol"], sys["wf"]),
        maxiters=5,
        vmc=vmc,
        vmcoptions=dict(npartitions=ncore, nsteps=100),
    )

    forcing = {}
    obj = {}
    for k in sys["descriptors"]:
        forcing[k] = 0.0
        obj[k] = 0.0

    forcing["t"] = 0.5
    forcing["trace"] = 0.5
    obj["t"] = 0.0
    obj["trace"] = 2.0

    datafile = "saveh2.json"

    wf, df = optimize(
        sys["wf"],
        configs,
        sys["acc"],
        objective=obj,
        forcing=forcing,
        iters=50,
        tstep=0.1,
        datafile=datafile,
        vmc=vmc,
        vmcoptions=dict(npartitions=ncore, nsteps=100),
    )
