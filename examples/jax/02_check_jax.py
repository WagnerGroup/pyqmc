import os
# here we are desperately trying to avoid multithreading
# to get a good read on the single-threaded performance
# you may or many not want this depending on your use case
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"
)
os.environ["JAX_NUM_CLIENTS"] = "1"
os.environ["NPROC"] = "1"

import pyqmc.api as pyq
import jax
import time
import numpy as np
import pandas as pd

# you may or may not want to set this.
jax.config.update('jax_platform_name', 'cpu')
# you almost always want 64-bit math.
jax.config.update("jax_enable_x64", True)
print(jax.devices())


def check_value(configs, wf_jax, wf_pyscf):
    """
    print out timing and difference in values for recompute()
    """
    vals_jax = wf_jax.recompute(configs)
    jax.block_until_ready(vals_jax)
    start = time.perf_counter()
    vals_jax = wf_jax.recompute(configs)
    jax.block_until_ready(vals_jax)
    jax_time = time.perf_counter()
    vals_pyscf = wf_pyscf.recompute(configs)
    pyscf = time.perf_counter()
    print(f"JAX time {jax_time - start} s PYSCF time {pyscf - jax_time} s " 
          f" Difference {np.mean(np.abs(vals_jax[0] - vals_pyscf[0]))}")


def check_energy(configs, wf_jax, wf_pyscf) -> pd.DataFrame:
    """
    Check the various energies.
    Note that between old and new ECPS they should only agree on average

    """
    enacc = {'old':pyq.EnergyAccumulator(cell, use_old_ecp=True),
             'new': pyq.EnergyAccumulator(cell, use_old_ecp=False)  }
    wfs = {'jax': wf_jax, 'pyscf': wf_pyscf}
    data = []
    for ecp in enacc.keys():
        for wf in wfs.keys():
            if wf == 'jax': #force compile
                enacc[ecp](configs, wfs[wf])
            start = time.perf_counter()
            en = enacc[ecp](configs, wfs[wf])
            end = time.perf_counter()
            data.append({ 'time': end - start,
                          'wf':wf,
                          'ecp':ecp,
                          'ecp_en': np.mean(en['ecp']),
                          'grad2': np.mean(en['grad2']),
                          'ke': np.mean(en['ke']),
                          'total':np.mean(en['total']),
                          })
    return pd.DataFrame(data)


if __name__ == "__main__":
    # we found that etb0.2 was the lowest DFT energy
    cell, mf = pyq.recover_pyscf("etb0.2.hdf5")
    wf_jax, _ = pyq.generate_slater(cell, mf, jax=True, nimages=2)
    wf_pyscf, _ = pyq.generate_slater(cell, mf, jax=False, eval_gto_precision=1e-16 )

    for nconfig in [10, 100, 1000]:
        print(f"###### nconfig = {nconfig}")
        configs = pyq.initial_guess(cell, nconfig)
        check_value(configs, wf_jax, wf_pyscf)
        df = check_energy(configs, wf_jax, wf_pyscf)
        print(df)



