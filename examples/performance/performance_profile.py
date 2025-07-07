import pyqmc.api as pyq
import h5py
import time 
import sys
import os
import itertools
import pyscf.pbc.tools.k2gamma as k2gamma
from pyqmc.observables.eval_ecp import ecp
from pyqmc.observables.jax_ecp import ECPAccumulator

# here we are desperately trying to avoid multithreading
# to get a good read on the single-threaded performance
# you may or many not want this depending on your use case
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ[
    "XLA_FLAGS"
] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"
os.environ["JAX_NUM_CLIENTS"] = "1"
os.environ["NPROC"]="1"

import jax
# Update any global JAX configurations if necessary.

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)


def evaluate_recompute(configs, wf):
    data = {}
    N = configs.configs.shape[0]
    vals = wf.recompute(configs)
    jax.block_until_ready(vals)
    val_start = time.perf_counter()
    vals = wf.recompute(configs)
    jax.block_until_ready(vals)
    val_end = time.perf_counter()
    data['recompute time per walker'] = (val_end - val_start)/N
    data['recompute_values'] = vals[1]
    return data


def evaluate_laplacian(configs, wf):
    data = {}
    N = configs.configs.shape[0]
    ne = configs.configs.shape[1]
    vals = wf.recompute(configs)
    for e in range(ne):
        vals = wf.gradient_laplacian(e, configs.electron(e))
        jax.block_until_ready(vals)
    val_start = time.perf_counter()
    for e in range(ne):
        vals = wf.gradient_laplacian(e, configs.electron(e))
        jax.block_until_ready(vals)
    jax.block_until_ready(vals)
    val_end = time.perf_counter()
    data['laplacian time per walker'] = (val_end - val_start)/N
    data['laplacian_values'] = vals[1]
    return data


def evaluate_old_ecp(configs, cell, wf, threshold=10):
    data = {}
    N = configs.configs.shape[0]
    res = ecp(cell, configs, wf, threshold=threshold)
    jax.block_until_ready(res)
    
    start_time = time.perf_counter()
    res = ecp(cell, configs, wf, threshold=threshold)
    jax.block_until_ready(res)
    end_time = time.perf_counter()

    data['old_ecp time per walker'] = (end_time - start_time)/N
    data['old_ecp_values'] = res
    return data

def evaluate_new_ecp(configs, cell, wf):
    data = {}
    N = configs.configs.shape[0]
    accumulator = ECPAccumulator(cell)
    res = accumulator(configs, wf)
    jax.block_until_ready(res)

    start_time = time.perf_counter()
    res = accumulator(configs, wf)
    jax.block_until_ready(res)

    end_time = time.perf_counter()
    data['new_ecp time per walker'] = (end_time - start_time)/N
    data['new_ecp_values'] = res
    return data

def evaluate_performance(configs, cell, wf):
    data = {}

    data['nconfigs'] = configs.configs.shape[0]
    # Recompute
    print("Evaluating recompute performance...")
    data.update(evaluate_recompute(configs, wf))

    print("Evaluating old ECP performance...")
    data.update(evaluate_old_ecp(configs, cell, wf))

    print("Evaluating new ECP performance...")
    data.update(evaluate_new_ecp(configs, cell, wf))

    print("Evaluating laplacian performance...")
    data.update(evaluate_laplacian(configs, wf))

    return data


def main():

    mf_chkfile = "NVdiamond_kroks.chk"
    ci_chkfile = "NVdiamond_casci.chk"
    cell, kmf, mc = pyq.recover_pyscf(mf_chkfile, ci_checkfile=ci_chkfile, cancel_outputs=False)
    print(cell.nelec)
    cell.spin = 0
    cell.build()
    mf = k2gamma.k2gamma(kmf)
    mc.ci = mc.ci[0]

    wfs = {
        'jaxnimage2': {"jax": True, "slater_kws": {"tol": 0, 'nimages': 2}},
        'jaxnimage1': {"jax": True, "slater_kws": {"tol": 0, 'nimages': 1}},
        'nonjax': {"jax": False, "slater_kws": {"tol": 0}},
    }


    for N in [5, 10, 20]:
        configs = pyq.initial_guess(cell,N)
        for wf_name, wf_params in wfs.items():
            hdf_file = f'performance_{wf_name}_{N}.hdf5'
            if os.path.exists(hdf_file):
                print(f"Skipping {wf_name} for N={N}, file already exists.")
                continue
            print(f"Generating wavefunction {wf_name} with parameters {wf_params}")
            wf, _ = pyq.generate_wf(cell, mf, mc=mc, **wf_params)

            df = evaluate_performance(configs, cell, wf)
            print(df)
            with h5py.File(hdf_file, 'w') as f:
                for key, value in df.items():
                    f.create_dataset(key, data=value)
                f['N'] = N
                f['wf_name'] = wf_name.encode('utf-8')



if __name__=="__main__":
    main()
