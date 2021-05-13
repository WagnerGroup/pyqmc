Common problems
---------------


### Hanging when using ProcessPoolExecutor 

(Written 05/12/2021)

If you use `concurrent.futures.ProcessPoolExecutor`, then you cannot use openMP within `pyscf`. Fixes: 
* Set the environment variable `export OPENMP_NUM_THREADS=1` before running
* Instead use `loky.get_reusable_executor` (you have to install the loky package)
* Instead use `mpi4py.futures.MPIPoolExecutor`
