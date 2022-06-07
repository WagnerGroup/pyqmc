Installation instructions for HPC machines
--------------------------------------------------------------


### Illinois campus cluster
#### Setting up your conda installation

Make sure your `.bashrc` is not loading another conda environment and that `PYTHONPATH` is not set. If it is, clear those things, log out, and log back in.

```
module load anaconda/3
conda create --name fast-mpi4py python=3.8
module load openmpi/3.1.1-gcc-7.2.0
export MPICC=$(which mpicc)
.  /usr/local/anaconda/5.2.0/python3/etc/profile.d/conda.sh
conda activate fast-mpi4py
pip install -v --no-binary mpi4py mpi4py
pip install -v --no-binary numpy numpy
pip install pyscf
module load git
pip install git+git://github.com/WagnerGroup/pyqmc --upgrade
pip install snakemake
```

#### Submission script

```
#!/usr/bin/env python3
import sys
import uuid
import os
subname = str(uuid.uuid4())
jobname=sys.argv[1]
filename=sys.argv[1]
with open(subname,'w') as f:
    f.write(f"""#!/bin/bash
#SBATCH --time=36:00:00                  # Job run time (hh:mm:ss)
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --cpus-per-task=40             # Number of task (cores/ppn) per node
#SBATCH --job-name={jobname}            # Name of batch job
#SBATCH --partition=qmchamm           # Partition (queue)
#SBATCH --output={jobname}.o
##SBATCH --error={jobname}.e              # Name of batch job error file
##SBATCH --mail-user=NetID@illinois.edu  # Send email notifications
##SBATCH --mail-type=BEGIN,END           # Type of email notifications to send
. ~/bin/activate_conda.sh
export OMP_NUM_THREADS=1
srun -n 1 -c 40 python3  {filename} >& {filename}.stdout
""")

os.system(f"sbatch {subname}")
```



#### snakemake setup

I would suggest starting with the setup here: https://github.com/WagnerGroup/Energy-Entropy

```
module load anaconda/3
module load openmpi/3.1.1-gcc-7.2.0
.  /usr/local/anaconda/5.2.0/python3/etc/profile.d/conda.sh
conda activate fast-mpi4py
pip install cookiecutter
cookiecutter https://github.com/Snakemake-Profiles/slurm.git

```

The profile will be installed in `~/.config/snakemake/slurm`. I have changed my version of `slurm-submit.py` to include the partition:
```
RESOURCE_MAPPING = {
    "time": ("time", "runtime", "walltime"),
    "mem": ("mem", "mem_mb", "ram", "memory"),
    "mem-per-cpu": ("mem-per-cpu", "mem_per_cpu", "mem_per_thread"),
    "nodes": ("nodes", "nnodes"),
    "partition": ("partition",)
}
```

Then your snakemake can look like the following, if you'd like to run on `qmc_threads` cores on one node and the secondary partition. Note the resources section in the rule.
```
rule VMC_MF:
    input: hf="{dir1}/{dir2}/mf.chk", wffile="{dir1}/{dir2}/vmc_mf_{tol}_{orbs}_{nconfig}.chk"
    output: "{dir1}/{dir2}/eval_mf_{tol}_{orbs}.chk"
    threads: qmc_threads
    resources:
        walltime="4:00:00", partition="secondary"
    run:
        with concurrent.futures.ProcessPoolExecutor(max_workers=qmc_threads) as client:
            functions.evaluate_vmc(input.hf, None, input.wffile, output[0], 
                  slater_kws=None, nblocks=1000, client=client, npartitions=qmc_threads)
```
TODO: change this to work with MPI.

You can run by doing something like the following:
```
module load anaconda/3
module load openmpi/3.1.1-gcc-7.2.0
.  /usr/local/anaconda/5.2.0/python3/etc/profile.d/conda.sh
conda activate fast-mpi4py
hostname >> run.out
nohup nice snakemake --jobs=10  --profile=slurm [filenames] > vmc.out & 
```
You can run this on a login node and it will execute your jobs.



### Summit

Some information is available [here](https://www.olcf.ornl.gov/wp-content/uploads/2019/02/STW_Feb_20190211_summit_workshop_python.pdf)

Set up your environment:

```
module load python/3.8-anaconda3
conda create -n pyqmc3.9 python=3.9
conda init
. .bashrc 
conda activate pyqmc3.9
conda install numpy pandas h5py scipy
module load gcc
CC=gcc MPICC=mpicc pip install --no-binary mpi4py install mpi4py

```


Install pyscf and pyqmc. I have had trouble just doing `pip install pyscf` for non-Intel machines. 
```
module load gcc
module load cmake
module load openblas

git clone https://github.com/pyscf/pyscf
cd pyscf/pyscf/lib
mkdir build
cd build
cmake ..
make 
cd [pyscf root directory]
pip install .
pip install pyqmc
```
