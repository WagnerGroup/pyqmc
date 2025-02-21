import jax
import jax.numpy as jnp
from functools import partial
import pyqmc.pyscftools
from pyqmc.jax import gto
import pyscf.gto 
from typing import NamedTuple




class SlaterState(NamedTuple):
    """
    These define the state of a slater determinant wavefunction
    """
    mo_values: jnp.ndarray # Nelec, norbs
    sign : jnp.ndarray  # float
    logabsdet : jnp.ndarray # float
    inverse: jnp.ndarray # nelec_s, nelec_s


class DeterminantParameters(NamedTuple):
    """
    These define the expansion of the wavefunction in terms of determinants.
    We have to separate into up and down variables because they may be different sizes.
    """
    ci_coeff: jnp.ndarray
    mo_coeff_alpha: jnp.ndarray
    mo_coeff_beta: jnp.ndarray

class DeterminantExpansion(NamedTuple):
    """
    This determines the mapping of the determinants. 
    This is kept separate from the parameters because we need to take the derivative with 
    respect to the parameters and JAX doesn't support taking the gradient with respect 
    to only one part of a tuple.
    """
    mapping_up: jnp.ndarray  # mapping of the determinants to the ci_coeff
    mapping_down: jnp.ndarray 
    determinants_up: jnp.ndarray #Orbital occupancy for each up/down determinant
    determinants_down: jnp.ndarray 


def _compute_one_determinant(mos, det):
    return jnp.linalg.slogdet(jnp.take(mos, det, axis = 1))

vmap_determinants = jax.vmap(_compute_one_determinant, in_axes = (None,0), out_axes = 0)

def _compute_inverse(mos, det):
    return jnp.linalg.inv(jnp.take(mos, det, axis = 1))

vmap_inverse = jax.vmap(_compute_inverse, in_axes = (None,0), out_axes = 0)


def compute_determinants(mo_coeff: jnp.ndarray, # nbasis, norb
                        gto_evaluator,  #basis evaluator that returns nelec, nbasis
                        determinants: jnp.ndarray,  # (ndet, norb) what 
                        xyz ) -> SlaterState:
    """
    Compute a set of determinants given by `det_params` at a given position `xyz`
    This is meant to represent all the determinants of one spin
    """
    aos = gto_evaluator(xyz) # Nelec, nbasis
    mos = jnp.dot(aos, mo_coeff) # Nelec, norbs
    dets = vmap_determinants(mos, determinants)
    inverses = vmap_inverse(mos, determinants)
    return SlaterState(mos, dets.sign, dets.logabsdet, inverses)

def evaluate_expansion(gto_evaluator, #function (N,3) -> (N,nbasis)
                       expansion: DeterminantExpansion,
                       nelec: jnp.ndarray, # (2,)
                       det_params: DeterminantParameters,
                       xyz:jnp.ndarray, # (nelec, 3)
                       ):
    dets_up = compute_determinants(det_params.mo_coeff_alpha,
                                   gto_evaluator,
                                   expansion.determinants_up,
                                   xyz[0:nelec[0],:]
                                   )
    dets_down = compute_determinants(det_params.mo_coeff_beta,
                                   gto_evaluator,
                                   expansion.determinants_down,
                                   xyz[nelec[0]:,:]
                                   )
    
    logdets = jnp.take(dets_up.logabsdet, expansion.mapping_up) \
            + jnp.take(dets_down.logabsdet, expansion.mapping_down) # ndet
    signdets = jnp.take(dets_up.sign, expansion.mapping_up) \
             * jnp.take(dets_down.sign, expansion.mapping_down) #ndet
    ref = jnp.max(logdets)
    values = jnp.sum(det_params.ci_coeff *signdets* jnp.exp(logdets - ref)) # scalar
    return jnp.sign(values), jnp.log(jnp.abs(values))+ref, dets_up, dets_down


# Is there a way to do this in a more clean way, while still allowing for a static compile?
# mainly it's just a bit clunky to define an up and down version.

def _determinant_lemma(mos, inverse, det, e):
    """
    Returns the ratio of the determinant with electron e changed to have 
    the new orbitals given by mos
    """
    return  jnp.take(mos, det, axis=-1) @ inverse[:,e]
vmap_lemma = jax.vmap(_determinant_lemma, in_axes = (None,0,0,None), out_axes = 0) # over determinants


def testvalue_up(gto_evaluator, #function (3) -> (nbasis)
              expansion: DeterminantExpansion, 
              det_params: DeterminantParameters,
              dets_up: SlaterState,
              dets_down: SlaterState,
              e, # electron number
              xyz: jnp.ndarray 
):
    aos = gto_evaluator(xyz) #  nbasis
    mos = jnp.dot(aos.T, det_params.mo_coeff_alpha) # shape is (deriv, norbs)
    ratios = vmap_lemma(mos, dets_up.inverse, expansion.determinants_up, e).T # deriv, ndet (the transpose for that last axis is important)

    newsigns = jnp.sign(ratios)*dets_up.sign
    newlogabs = dets_up.logabsdet + jnp.log(jnp.abs(ratios))
    logdets = jnp.take(newlogabs, expansion.mapping_up, axis=-1) \
            + jnp.take(dets_down.logabsdet, expansion.mapping_down) # ndet
    signdets = jnp.take(newsigns, expansion.mapping_up, axis=-1) \
             * jnp.take(dets_down.sign, expansion.mapping_down) #ndet
    
    logdets_old = jnp.take(dets_up.logabsdet, expansion.mapping_up) \
            + jnp.take(dets_down.logabsdet, expansion.mapping_down) # ndet
    signdets_old = jnp.take(dets_up.sign, expansion.mapping_up) \
             * jnp.take(dets_down.sign, expansion.mapping_down) #ndet

    ref = jnp.max(jnp.concat( (logdets.flatten(), logdets_old))) 
    values = jnp.sum(det_params.ci_coeff *signdets* jnp.exp(logdets - ref), axis=-1) # derivatives
    values_old = jnp.sum(det_params.ci_coeff *signdets_old* jnp.exp(logdets_old - ref)) # scalar
    ratio = values/values_old
    return ratio


def testvalue_down(gto_evaluator, #function (3) -> (nbasis)
              expansion: DeterminantExpansion, 
              det_params: DeterminantParameters,
              dets_up: SlaterState,
              dets_down: SlaterState,
              e, # electron number
              xyz: jnp.ndarray 
):
    aos = gto_evaluator(xyz) #  nbasis
    mos = jnp.dot(aos.T, det_params.mo_coeff_beta) # norbs
    ratios = vmap_lemma(mos, dets_down.inverse, expansion.determinants_down, e).T #ndet ratios

    newsigns = jnp.sign(ratios)*dets_down.sign
    newlogabs = dets_down.logabsdet + jnp.log(jnp.abs(ratios))
    logdets = jnp.take(dets_up.logabsdet, expansion.mapping_up) \
            + jnp.take(newlogabs, expansion.mapping_down, axis=-1) # ndet
    signdets = jnp.take(dets_up.sign, expansion.mapping_up) \
             * jnp.take(newsigns, expansion.mapping_down, axis=-1) #ndet
    
    logdets_old = jnp.take(dets_up.logabsdet, expansion.mapping_up) \
            + jnp.take(dets_down.logabsdet, expansion.mapping_down) # ndet
    signdets_old = jnp.take(dets_up.sign, expansion.mapping_up) \
             * jnp.take(dets_down.sign, expansion.mapping_down) #ndet

    ref = jnp.max(jnp.concat( (logdets.flatten(), logdets_old))) 
    values = jnp.sum(det_params.ci_coeff *signdets* jnp.exp(logdets - ref), axis=-1) # derivatives
    values_old = jnp.sum(det_params.ci_coeff *signdets_old* jnp.exp(logdets_old - ref)) # scalar
    ratio = values/values_old
    return ratio



def create_wf_evaluator(mol, mf):
    """
    Create a set of functions that can be used to evaluate the wavefunction.

    """
    # Basis evaluators
    gto_1e = gto.create_gto_evaluator(mol)
    gto_ne = jax.vmap(gto_1e, in_axes=0, out_axes=0)# over electrons

    # determinant expansion
    _determinants = pyqmc.pyscftools.determinants_from_pyscf(mol, mf, mc=None, tol=1e-9)
    ci_coeff, determinants, mapping = pyqmc.determinant_tools.create_packed_objects(_determinants, tol=1e-9)

    ci_coeff = jnp.array(ci_coeff)
    determinants = jnp.array(determinants)
    mapping = jnp.array(mapping)
    mo_coeff = mf.mo_coeff[:,:jnp.max(determinants)+1] # this only works for RHF for now..

    det_params = DeterminantParameters(ci_coeff, mo_coeff, mo_coeff)
    expansion = DeterminantExpansion( mapping[0], mapping[1], determinants[0], determinants[1])
    nelec = tuple(mol.nelec)
    value = partial(evaluate_expansion, gto_ne, expansion, nelec)
    _testvalue_up = partial(testvalue_up, gto_1e, expansion)
    _testvalue_down = partial(testvalue_down, gto_1e, expansion)

    # electron gradient will be testvalue with gradient of gto_1e
    gto_1e_grad = jax.jacobian(gto_1e)
    grad_up = partial(testvalue_up, gto_1e_grad, expansion)
    grad_down = partial(testvalue_down, gto_1e_grad, expansion)


    gto_1e_laplacian = jax.hessian(gto_1e)
    laplacian_up = partial(testvalue_up, gto_1e_laplacian, expansion)
    laplacian_down = partial(testvalue_down, gto_1e_laplacian, expansion)

    # pgradient is derivative of value with respect to ci_coeff and mo_coeff 
    pgradient = jax.jacobian(value, argnums=0)

    # compile all the testvalue functions
    testval_funcs = [_testvalue_up, _testvalue_down, grad_up, grad_down, laplacian_up, laplacian_down]
    testval_funcs = ( jax.jit(
                       jax.vmap(f,
                                    in_axes=(None, #parameters
                                    SlaterState(0,0,0,0), 
                                    SlaterState(0,0,0,0), 
                                    None, #electron number
                                    0), #xyz
                                    out_axes=(0)
                                      )
                    ) 
                     for f in testval_funcs)
    
    
    value_func = [value, pgradient]
    value_func = (jax.jit(
                jax.vmap(f, in_axes=(None, #parameters
                                             0), #xyz
                                             out_axes=(0,0,
                                                       SlaterState(0,0,0,0), 
                                                       SlaterState(0,0,0,0)))
                )
                for f in value_func)

    # The vmaps here are over configurations
    return det_params, value_func, testval_funcs



class JAXSlater:
    def __init__(self, mol, mf):
        self._parameters, (self._recompute, self._pgradient), \
        (_testvalue_up, _testvalue_down, _grad_up, _grad_down, _lap_up, _lap_down) = create_wf_evaluator(mol, mf)
        self._testvalue=(_testvalue_up, _testvalue_down)
        self._grad = (_grad_up, _grad_down)
        self._lap = (_lap_up, _lap_down)
        self._nelec = tuple(mol.nelec)

    def recompute(self, configs):
        xyz = jnp.array(configs.configs)
        self._sign, self._logabs, self._dets_up, self._dets_down = self._recompute(self._parameters, xyz)
        return self._sign, self._logabs
    
    def updateinternals(self, e, epos, configs, mask=None, saved_values=None):
        return self.recompute(configs)
    

    def testvalue(self, e, epos, mask=None):
        xyz = jnp.array(epos.configs)
        spin = int(e >= self._nelec[0] )
        e = e - self._nelec[0]*spin
        newvals = self._testvalue[spin](self._parameters, self._dets_up, self._dets_down, e, xyz)
        return newvals, newvals

    def gradient(self, e, epos, mask=None):
        xyz = jnp.array(epos.configs)
        spin = int(e >= self._nelec[0] )
        e = e - self._nelec[0]*spin
        return self._grad[spin](self._parameters, self._dets_up, self._dets_down, e, xyz).T # pyqmc wants (3, nconfig)
    

    def gradient_value(self, e, epos):
        xyz = jnp.array(epos.configs)
        spin = int(e >= self._nelec[0] )
        e = e - self._nelec[0]*spin
        values = self._testvalue[spin](self._parameters, self._dets_up, self._dets_down, e, xyz)
        derivatives = self._grad[spin](self._parameters, self._dets_up, self._dets_down, e, xyz).T # pyqmc wants (3, nconfig)

        return derivatives, values, None


    def gradient_laplacian(self, e, epos):
        xyz = jnp.array(epos.configs)
        spin = int(e >= self._nelec[0] )
        e = e - self._nelec[0]*spin
        gradient = self._grad[spin](self._parameters, self._dets_up, self._dets_down, e, xyz).T # pyqmc wants (3, nconfig)
        laplacian = jnp.trace(self._lap[spin](self._parameters, self._dets_up, self._dets_down, e, xyz), axis1=1, axis2=2)
        return gradient, laplacian
        

    def laplacian(self, e, epos, mask=None):
        xyz = jnp.array(epos.configs)
        spin = int(e >= self._nelec[0] )
        e = e - self._nelec[0]*spin
        return jnp.trace(self._lap[spin](self._parameters, self._dets_up, self._dets_down, e, xyz), axis1=1, axis2=2)
    
    def pgradient(self, configs):
        xyz = jnp.array(configs.configs)
        grads =  self._pgradient(self._parameters, xyz)[1] # sign, log, dets_up, dets_down
        return {'det_coeff': grads[0], 'mo_coeff_alpha': grads[1], 'mo_coeff_beta': grads[2]}
        




if __name__=="__main__":

    cpu=True
    if cpu:
        jax.config.update('jax_platform_name', 'cpu')
        jax.config.update("jax_enable_x64", True)
    else:
        pass 

    import time
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pyqmc.testwf

    import pyqmc.api as pyq
    mol = {}
    mf = {}

    mol['h2o'] = pyscf.gto.Mole(atom = '''O 0 0 0; H  0 2.0 0; H 0 0 2.0''', basis = 'unc-ccecp-ccpvdz', ecp='ccecp', cart=True)
    mol['h2o'].build()
    mf['h2o'] = pyscf.scf.RHF(mol['h2o']).run()
    
    jax_slater = JAXSlater(mol['h2o'], mf['h2o'])
    slater = pyq.Slater(mol['h2o'], mf['h2o'])

    data = []
    for nconfig in [10]:
        configs = pyq.initial_guess(mol['h2o'], nconfig)
        configs_aux = pyq.initial_guess(mol['h2o'], nconfig)
        wfval = jax_slater.recompute(configs) 
        jax.block_until_ready(wfval)

        jax_start = time.perf_counter()
        wfval = jax_slater.recompute(configs) 

        jax.block_until_ready(wfval)
        jax_end = time.perf_counter()

        newval, _ = jax_slater.testvalue(0, configs.electron(0))
        print("newval", newval)



        slater_start = time.perf_counter()
        values_ref = slater.recompute(configs)
        slater_end = time.perf_counter()

        electron = 3
        newval = jax_slater.gradient(electron, configs.electron(electron))
        print("gradient", newval, newval.shape)
        newval = slater.gradient(electron, configs.electron(electron))
        print("gradient from pyqmc", newval, newval.shape)


        newval = jax_slater.laplacian(electron, configs.electron(electron))
        print("laplacian", newval, newval.shape)
        newval = slater.laplacian(electron, configs.electron(electron))
        print("laplacian from pyqmc", newval, newval.shape)

        newval = jax_slater.pgradient(configs)
        print("parameter gradient", newval['mo_coeff_alpha'][0, 0:10])
        newval = slater.pgradient()
        print(newval.keys())
        print("parameter gradient from pyqmc", newval['mo_coeff_alpha'][0, 0:10])





        print("jax", jax_end-jax_start, "slater", slater_end-slater_start)
        data.append({'N': nconfig, 'time': jax_end-jax_start, 'method': 'jax'})
        data.append({'N': nconfig, 'time': slater_end-slater_start, 'method': 'pyqmc'})
        print('MAD', jnp.mean(jnp.abs(values_ref[1] - wfval[1])))
        print("jax values", wfval[1])
        print("pyqmc values", values_ref[1])
    sns.lineplot(data = pd.DataFrame(data), x='N', y='time', hue='method')
    plt.ylim(0)
    plt.savefig("jax_vs_pyqmc.png")
