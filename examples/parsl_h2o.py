from pyscf import gto,scf
import pyqmc
import parsl
from parsl.configs.exex_local import config
from pyqmc.parsltools import clean_pyscf_objects,distvmc

#############################
# Set up the parallelization
#############################
ncore=2
nconf=1000   #This must be a multiple of ncore

#having one extra rank seems to help with performance a little; we only  
#ever run on ncore
config.executors[0].ranks_per_node = ncore+1 

#parsl for some reason always makes a directory called 'runinfo'
#If you want to run several calculations in a single directory, 
#make sure that they get different directory names.
import string
import random
def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))
config.run_dir='parsldir'+randomString(4)


##########################
#The calculation
##########################

mol = gto.M(atom='O 0 0 0; H 0 -2.757 2.587; H 0 2.757 2.587',basis='bfd_vtz',ecp='bfd')
mf = scf.RHF(mol).run()
#clean_pyscf_objects gets rid of the TextIO objects that can't 
#be sent using parsl.
mol,mf=clean_pyscf_objects(mol,mf)

#It's better to load parsl after pyscf has run. Some of the 
#executors have timeouts and will kill the job while pyscf is running!
parsl.load(config)


#We make a Slater-Jastrow wave function and 
#only optimize the Jastrow coefficients.
wf=pyqmc.slater_jastrow(mol,mf)
acc=pyqmc.gradient_generator(mol,wf,['wf2acoeff','wf2bcoeff'])


#Generate the initial configurations. 
#Here we run VMC for a few steps with no accumulators to equilibrate the 
#walkers.
configs = pyqmc.initial_guess(mol, nconf)
df,configs = distvmc(wf,configs,accumulators={}, nsteps=10, npartitions=ncore)

#Line minimization costs a lot more per iteration but is more reliable than 
# stochastic gradient descent (gradient_descent). 
wf,df=pyqmc.line_minimization(wf,configs,acc,vmcoptions={'nsteps':30, 'npartitions':ncore},vmc=distvmc,
        dataprefix="parsl_h2o")


