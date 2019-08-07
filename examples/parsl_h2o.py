from pyscf import gto, scf
import pyqmc
import parsl
import logging
import pandas as pd
from pyqmc.parsltools import (
    clean_pyscf_objects,
    distvmc,
    line_minimization,
    distdmc_propagate
)
from pyqmc import rundmc, EnergyAccumulator

#############################
# Set up the parallelization
#############################
ncore = 2
nconf = 500  # This must be a multiple of ncore


# parsl for some reason always makes a directory called 'runinfo'
# If you want to run several calculations in a single directory,
# make sure that they get different directory names.
import string
import random
from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from parsl.launchers import SimpleLauncher

from parsl.config import Config
from parsl.executors import ExtremeScaleExecutor
# having one extra rank seems to help with performance a little; we only
# ever run on ncore

config = Config(
    executors=[
        ExtremeScaleExecutor(
            label="Extreme_Local",
            worker_debug=False, #Turn this on to make large files!
            ranks_per_node=ncore+1,
            provider=LocalProvider(
                channel=LocalChannel(),
                init_blocks=1,
                max_blocks=1,
                launcher=SimpleLauncher(),
            )
        )
    ],
    strategy=None,
    run_dir='parsldir' + "".join([random.choice(string.ascii_letters) for i in range(3)])
)


##########################
# The calculation
##########################

mol = gto.M(
    atom="O 0 0 0; H 0 -2.757 2.587; H 0 2.757 2.587", basis="bfd_vtz", ecp="bfd"
)
mf = scf.RHF(mol).run()
# clean_pyscf_objects gets rid of the TextIO objects that can't
# be sent using parsl.
mol, mf = clean_pyscf_objects(mol, mf)

# It's better to load parsl after pyscf has run. Some of the
# executors have timeouts and will kill the job while pyscf is running!
parsl.load(config)
parsl.set_stream_logger(level=logging.WARNING)


# We make a Slater-Jastrow wave function and
# only optimize the Jastrow coefficients.
wf = pyqmc.slater_jastrow(mol, mf)
acc = pyqmc.gradient_generator(mol, wf, ["wf2acoeff", "wf2bcoeff"])


# Generate the initial configurations.
# Here we run VMC for a few steps with no accumulators to equilibrate the
# walkers.
configs = pyqmc.initial_guess(mol, nconf)
df, configs = distvmc(wf, configs, accumulators={}, nsteps=10, npartitions=ncore)

# This uses a stochastic reconfiguration step to generate parameter changes along a line,
# then minimizes the energy along that line.
wf, dfgrad, dfline = line_minimization(
    wf,
    configs,
    acc,
    npartitions=ncore,
    vmcoptions={"nsteps": 30},
    dataprefix="parsl_h2o",
)



dfdmc, configs_, weights_ = rundmc(
    wf,
    configs,
    nsteps=1000,
    branchtime=5,
    accumulators={"energy": EnergyAccumulator(mol)},
    ekey=("energy", "total"),
    tstep=0.01,
    verbose=True,
    propagate=distdmc_propagate,
    npartitions=ncore,
)

dfdmc = pd.DataFrame(dfdmc)
dfdmc.sort_values("step", inplace=True)
dfdmc.to_csv("parsl_h2o_dmc.csv")
warmup = 200
dfprod = dfdmc[dfdmc.step > warmup]

reblock = pyblock.reblock(dfprod[["energytotal", "energyei"]])
print(reblock[1])

