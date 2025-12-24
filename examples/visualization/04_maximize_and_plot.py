import pyqmc.api as pyq
import numpy as np
from scipy.optimize import minimize
import conditional_wf
mol, mf, mc = pyq.recover_pyscf("benzene.hdf5", "benzene_hci.hdf5", cancel_outputs=False)
mc.ci = mc.ci[0] # choose first root
wf, to_opt = pyq.generate_wf(mol, mf,mc= mc)
pyq.read_wf(wf, "optimize_jast.hdf5")

configs = pyq.initial_guess(mol, 1)

def absolute(x):
    configs.configs[0,:,:] = x.reshape(-1, 3)
    val = wf.recompute(configs)
    print(val)
    return -val[1]

def gradient(x):
    configs.configs[0,:,:] = x.reshape(-1, 3)
    val = wf.recompute(configs)
    ne = configs.configs.shape[1]
    grad = np.zeros((ne,3))
    for i in range(ne):
        grad[i,:] = wf.gradient_value(i, configs.electron(i))[0][:,0]
    return -grad.flatten()

x = configs.configs.flatten()[:]
print(absolute(x))
print(gradient(x))

maxres = minimize(absolute, x, method="BFGS", jac=gradient)
configs.configs[0,:,:] = maxres.x.reshape(-1,3)
conditional_wf.plot_conditional_wf(mol, wf, configs, resolution=0.2)