import numpy as np
from numpy.random.mtrand import normal
import pyqmc.mc as mc
import scipy.stats

"""
TODO:

 2) Implement a function that generates d<psi_i|psi_j>/dp, d E_i/dp
    2.5) Write a test for the averaging function with determinants (we should know the exact values for all these)
 3) Parallel implementation of averager, with test

 4) Correlated sampling
    4.5) Correlated sampling test

 5) Optimizer
     5.5) Optimizer test
"""


def collect_overlap_data(wfs, configs, energy, transforms):
    r"""Collect the averages over configs assuming that
    configs are distributed according to

    .. math:: \rho \propto \sum_i |\Psi_i|^2

    The keys 'overlap' and 'overlap_gradient' are

    `overlap` :

    .. math:: \langle \Psi_f | \Psi_i \rangle = \left\langle \frac{\Psi_i^* \Psi_j}{\rho} \right\rangle_{\rho}

    `overlap_gradient`:

    .. math:: \partial_m \langle \Psi_f | \Psi_i \rangle = \left\langle \frac{\partial_{fm} \Psi_i^* \Psi_j}{\rho} \right\rangle_{\rho}

    The function returns two dictionaries: 

    weighted_dat: each element is a list (one item per wf) of quantities that are accumulated as O psi_i^2/rho
    unweighted_dat: Each element is a numpy array that are accumulated just as O (no weight). 
                    This in particular includes 'weight' which is just psi_i^2/rho

    """
    phase, log_vals = [np.nan_to_num(np.array(x)) for x in zip(*[wf.value() for wf in wfs])]
    log_vals = np.real(log_vals)  # should already be real
    ref = np.max(log_vals, axis=0)
    denominator = np.sum(np.exp(2 * (log_vals - ref)), axis=0)
    normalized_values = phase * np.exp(log_vals - ref)

    # Weight for quantities that are evaluated as
    # int( f(X) psi_f^2 dX )
    # since we sampled sum psi_i^2
    weight = np.exp(-2 * (log_vals[:, np.newaxis] - log_vals))
    weight = 1.0 / np.sum(weight, axis=1) # [wf, config]

    energies = [energy(configs, wf) for wf in wfs]

    dppsi = [transform.serialize_gradients(wf.pgradient()) for transform, wf in zip(transforms, wfs)] 

    weighted_dat = {}
    unweighted_dat  = {}

     # normalized_values are [config,wf]
     # we average over configs here and produce [wf,wf]
    unweighted_dat["overlap"] = np.einsum( 
        "ik,jk->ij", normalized_values.conj(), normalized_values / denominator
    ) / len(ref)

    #Weighted average
    for k in energies[0].keys():
        weighted_dat[k]=[]
    for wt, en in zip(weight, energies): 
        for k in en.keys():
            weighted_dat[k].append(np.mean(en[k]*wt,axis=0))

    weighted_dat['dpidpj'] = []
    weighted_dat['dppsi'] = []
    weighted_dat["dpH"] = []
    for wfi, (dp,energy) in enumerate(zip(dppsi,energies)):
        weighted_dat['dppsi'].append(np.einsum(
            "ij,i->j", dp, weight[wfi] , optimize=True
        ))
        weighted_dat['dpidpj'].append(np.einsum(
            "ij,i,ik->jk", dp, weight[wfi] , dp, optimize=True
        ))
        weighted_dat["dpH"].append(np.einsum("i,ij,i->j", energy['total'], dp, weight[wfi]))

    
    ## We have to be careful here because the wave functions may have different numbers of 
    ## parameters
    for wfi, dp in enumerate(dppsi):
        unweighted_dat[("overlap_gradient",wfi)] = \
            np.einsum(
                "km,ik,jk->ijm",  # shape (wf, param) k is config index
                dp,
                normalized_values.conj(),
                normalized_values / denominator,
            )/ len(ref)

    unweighted_dat["weight"] = np.mean(weight, axis=1)
    return weighted_dat, unweighted_dat


def invert_list_of_dicts(A):
    """
    if we have a list [ {'A':1,'B':2}, {'A':3, 'B':5}], invert the structure to 
    {'A':[1,3], 'B':[2,5]}. 
    If not all keys are present in all lists, error.
    """
    final_dict = {}
    for k in A[0].keys():
        final_dict[k] = []
    for a in A:
        for k, v in a.items():
            final_dict[k].append(v)
    return final_dict


def sample_overlap_worker(wfs, configs, energy, transforms, nsteps=10, nblocks=10, tstep=0.5):
    r"""Run nstep Metropolis steps to sample a distribution proportional to
    :math:`\sum_i |\Psi_i|^2`, where :math:`\Psi_i` = wfs[i]
    """
    nconf, nelec, _ = configs.configs.shape
    for wf in wfs:
        wf.recompute(configs)
    weighted = []
    unweighted=[]
    for block in range(nblocks):
        print('-', end="", flush=True)
        weighted_block = {}        
        unweighted_block = {}

        for n in range(nsteps):
            for e in range(nelec):  # a sweep
                # Propose move
                grads = [np.real(wf.gradient(e, configs.electron(e)).T) for wf in wfs]
                grad = mc.limdrift(np.mean(grads, axis=0))
                gauss = np.random.normal(scale=np.sqrt(tstep), size=(nconf, 3))
                newcoorde = configs.configs[:, e, :] + gauss + grad * tstep
                newcoorde = configs.make_irreducible(e, newcoorde)

                # Compute reverse move
                grads, vals = list(zip(*[wf.gradient_value(e, newcoorde) for wf in wfs]))
                grads = [np.real(g.T) for g in grads]
                new_grad = mc.limdrift(np.mean(grads, axis=0))
                forward = np.sum(gauss ** 2, axis=1)
                backward = np.sum((gauss + tstep * (grad + new_grad)) ** 2, axis=1)

                # Acceptance
                t_prob = np.exp(1 / (2 * tstep) * (forward - backward))
                wf_ratios = np.abs(vals) ** 2
                log_values = np.real(np.array([wf.value()[1] for wf in wfs]))
                weights = np.exp(2 * (log_values - log_values[0]))

                ratio = t_prob * np.sum(wf_ratios * weights, axis=0) / weights.sum(axis=0)
                accept = ratio > np.random.rand(nconf)
                #block_avg["acceptance"][n] += accept.mean() / nelec

                # Update wave function
                configs.move(e, newcoorde, accept)
                for wf in wfs:
                    wf.updateinternals(e, newcoorde, configs, mask=accept)

            # Collect rolling average
            weighted_dat, unweighted_dat = collect_overlap_data(wfs, configs, energy, transforms)
            for k, it in unweighted_dat.items():
                if k not in unweighted_block:
                    unweighted_block[k] = np.zeros((*it.shape,), dtype=it.dtype)
                unweighted_block[k] += unweighted_dat[k] / nsteps

            for k, it in weighted_dat.items():
                if k not in weighted_block:
                    weighted_block[k] = [np.zeros((*x.shape,), dtype=x.dtype) for x in it]
                for b, v in zip(weighted_block[k], it):
                    b += v / nsteps
        weighted.append(weighted_block)
        unweighted.append(unweighted_block)


    # here we modify the data so that it's a dictionary of lists of arrays for weighted
    # and a dictionary of arrays for unweighted
    # Access weighted as weighted[quantity][wave function][block, ...]
    # Access unweighted as unweighted[quantity][block,...]
    weighted = invert_list_of_dicts(weighted)
    unweighted = invert_list_of_dicts(unweighted)

    for k in weighted.keys():
        weighted[k] = [np.asarray(x) for x in map(list, zip(*weighted[k]))]
    for k in unweighted.keys():
        unweighted[k] = np.asarray(unweighted[k])
    
    return weighted, unweighted, configs



def average(weighted, unweighted):
    """
    (more or less) correctly average the output from sample_overlap
    Returns the average and error as dictionaries.

    TODO: use a more accurate formula for weighted uncertainties
    """
    avg = {}
    error = {}
    for k,it in weighted.items():
        avg[k] = []
        error[k] = []
        #weight is [block,wf], so we transpose
        for v, w in zip(it,unweighted['weight'].T): 
            print(k,v)
            avg[k].append(np.sum(v, axis=0)/np.sum(w))
            error[k].append(scipy.stats.sem(v,axis=0)/np.mean(w))
    for k,it in unweighted.items():
        print('unweighted',k)
        avg[k] = np.mean(it, axis=0)
        error[k] = scipy.stats.sem(it, axis=0)
    return avg, error


def collect_terms(avg, error):
    """
    Generate 
    """

def run_test():
    from pyscf import lib, gto, scf
    import pyscf.pbc
    import numpy as np
    import pyqmc.api as pyq
    import pyqmc.accumulators
    from rich import print

    def H2_casci():
        mol = gto.M(atom="H 0. 0. 0.0; H 0. 0. 2.4",
                basis=f"ccpvtz",  
                unit="bohr", 
                charge=0, 
                spin=0, 
                verbose=1)  
        mf = scf.ROHF(mol).run()
        mc = pyscf.mcscf.CASCI(mf, 2, 2)
        mc.fcisolver.nroots = 4
        mc.kernel()
        return mol, mf, mc

    mol, mf, mc = H2_casci()

    print('energies', mc.e_tot)
    import copy
    mc1 = copy.copy(mc)
    mc2 = copy.copy(mc)
    mc1.ci = mc.ci[0]
    mc2.ci = mc.ci[1] # (mc.ci[0]+mc.ci[1])/np.sqrt(2)

    wf1, to_opt1 = pyq.generate_slater(mol, mf,mc=mc1, optimize_determinants=True)
    wf2, to_opt2 = pyq.generate_slater(mol, mf, mc=mc2, optimize_determinants=True)
    for to_opt in [to_opt1, to_opt2]:
        to_opt['det_coeff'] = np.ones_like(to_opt['det_coeff'],dtype=bool)

    transform1 = pyqmc.accumulators.LinearTransform(wf1.parameters,to_opt1)
    transform2 = pyqmc.accumulators.LinearTransform(wf2.parameters,to_opt2)
    configs = pyq.initial_guess(mol, 30)
    _, configs = pyq.vmc(wf1, configs)
    energy =pyq.EnergyAccumulator(mol)
    data_weighted, data_unweighted, coords = sample_overlap_worker([wf1,wf2],configs, energy, [transform1,transform2], 200)
    print(average(data_weighted, data_unweighted))
    #print(data_unweighted)

if __name__=="__main__":
    run_test()