import pyqmc.api as pyq
import copy
import pyqmc.observables.accumulators
from concurrent.futures import ProcessPoolExecutor


def test_transform_consistent_with_wf(H2_casci):
    """Test that evaluate_gradient_threaded works when given states with different numbers of determinants"""
    from pyqmc.method.ensemble_optimization_wfbywf import StochasticReconfigurationWfbyWf
    from pyqmc.method.ensemble_optimization_threaded import evaluate_gradients_threaded
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
    configs_ensemble = [
    [[copy.deepcopy(configs) for _ in range(2)] for _ in range(len(sr_accumulator[wfi]))]
    for wfi in range(2)
]   
    for i,wf in enumerate(wfs):
        print(f"For wf{i} {len(wf.parameters['det_coeff']) = }")
    with ProcessPoolExecutor() as executor:
        _, data_unweighted, configs = pyqmc.method.sample_many.sample_overlap(
                wfs,
                configs_ensemble[0][0][0],
                None,
                client=executor,
                npartitions=1
        )
        evaluate_gradients_threaded(wfs, configs_ensemble, sr_accumulator, client=executor)
    