import pyqmc.api as pyq
import copy
import pyqmc.observables.accumulators


def test_transform_consistent_with_wf(H2_casci):
    """Test that evaluate_gradient_threaded works when given states with different numbers of determinants"""
    from pyqmc.method.ensemble_optimization import StochasticReconfigurationWfbyWf
    from pyqmc.method.ensemble_optimization import evaluate_gradients_threaded
    mol, mf, mc = H2_casci
    mcs = [copy.copy(mc) for i in range(2)]
    for i in range(2):
        mcs[i].ci = mc.ci[i]

    energy = pyq.EnergyAccumulator(mol)
    sr_accumulator = []
    tol = 1e-20 # With tol = 1e-20 state 0 has 4 determinants whereas state 1 will have 3
    wfs = []
    for i in range(2):
        wf, to_opt = pyq.generate_slater(mol, mf, mc=mcs[i], optimize_determinants=True, tol = tol)
        wfs.append(wf)
        sr_accumulator.append(
            [
                StochasticReconfigurationWfbyWf(
                    energy,
                    pyqmc.observables.accumulators.LinearTransform(
                        wf.parameters, to_opt
                    ),
                )
            ]
        )
    configs = pyq.initial_guess(mol, 1)
    gradient_configs = [
        [
            {
                "energy": copy.deepcopy(configs),
                "overlap": copy.deepcopy(configs),
            }
            for _ in range(len(sr_accumulator[wfi]))
        ]
        for wfi in range(2)
    ]
    for i,wf in enumerate(wfs):
        print(f"For wf{i} {len(wf.parameters['det_coeff']) = }")
    evaluate_gradients_threaded(wfs, gradient_configs, sr_accumulator, client=None)
