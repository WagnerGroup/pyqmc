import jax
import jax.numpy as jnp
from functools import partial
import pyqmc.pyscftools
from pyqmc.wf.jax import gto
import pyscf.gto 
from typing import NamedTuple
import numpy as np
import time 

##############################
# Data storage tuples
##############################

class SlaterState(NamedTuple):
    """
    These define the state of a slater determinant wavefunction as 
    evaluated on a given position
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


##############################
# Evaluating the determinant
#############################

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

def compute_values_from_determinants(expansion: DeterminantExpansion,
                                     det_params: DeterminantParameters,
                                     dets_up: SlaterState,
                                     dets_down: SlaterState,):
    
    logdets = jnp.take(dets_up.logabsdet, expansion.mapping_up) \
            + jnp.take(dets_down.logabsdet, expansion.mapping_down) # ndet
    signdets = jnp.take(dets_up.sign, expansion.mapping_up) \
             * jnp.take(dets_down.sign, expansion.mapping_down) #ndet
    ref = jnp.max(logdets)
    values = jnp.sum(det_params.ci_coeff *signdets* jnp.exp(logdets - ref)) # scalar
    return jnp.sign(values), jnp.log(jnp.abs(values))+ref

compute_values_from_determinants = jax.vmap(compute_values_from_determinants, in_axes = (None,None,0,0), out_axes = (0,0))


def evaluate_expansion(gto_evaluator, #function (N,3) -> (N,nbasis)
                       expansion: DeterminantExpansion,
                       nelec: jnp.ndarray, # (2,)
                       det_params: DeterminantParameters,
                       xyz:jnp.ndarray, # (nelec, 3)
                       ):
    """
    This is the main function which collects the up and 
    down determinants, and combines them with the CI coefficients
    to create 
    """
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



##############################
# Single-electron update formulas
#############################

# Is there a way to do this in a more clean way, while still allowing for a static compile?
# It's just a bit clunky to define an up and down version.

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
    values_old = jnp.sum(det_params.ci_coeff *signdets_old* jnp.exp(logdets_old - ref), axis=-1) # scalar
    ratio = values/values_old
    return ratio, mos


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
    return ratio, mos

##############################
# Sherman-Morrison update formulas
#############################

def _sherman_morrison_row(e: int, 
                          inverse: jnp.ndarray, # nelec_s, nelec_s
                          mos: jnp.ndarray, # nelec_s
                          determinant: jnp.ndarray # nelec_s
                          ): 
    """
    """
    vec = jnp.take(mos, determinant, axis=-1)
    tmp = jnp.einsum("k,kj->j", vec, inverse)
    ratio = tmp[e]
    inv_ratio = inverse[:,e]/ratio

    invnew = inverse -  jnp.outer(inv_ratio, tmp) 
    invnew = invnew.at[:,e].set(inv_ratio)
    return ratio, invnew

# outer vmap is over configurations
# inner vmap is over determinants
sherman_morrison_row = jax.vmap(
                  jax.vmap(_sherman_morrison_row, in_axes = (None, 0, None, 0), out_axes = (0,0)),
             in_axes=(None, 0, 0, None), out_axes=(0,0))
sherman_morrison_row = jax.jit(sherman_morrison_row)


##############################
# Generating objects
#############################

def create_wf_evaluator(mol, mf, det_tol=1e-9, nimages=2, mc = None):
    """
    Create a set of functions that can be used to evaluate the wavefunction.

    """
    # Basis evaluators 
    gto_1e, gto_1e_vg, gto_1e_vgl = gto.create_gto_evaluator(mol, nimages=nimages)
    gto_ne = jax.vmap(gto_1e, in_axes=0, out_axes=0)# over electrons

    # determinant expansion
    # Use interpret_ci because we expect this to be gamma only.
    #
    if mc is not None:
        _determinants = pyqmc.pyscftools.interpret_ci(mc, tol=det_tol)
    else:
        _determinants = pyqmc.pyscftools.determinants_from_pyscf(mol, mf)
    ci_coeff, determinants, mapping = pyqmc.wf.determinant_tools.create_packed_objects(_determinants, tol=det_tol)

    ci_coeff = jnp.array(ci_coeff)
    #determinants = jnp.array(determinants)
    mapping = jnp.array(mapping)
    mo_coeff_up = mf.mo_coeff[:,:np.max(determinants[0])+1] 
    mo_coeff_down = mf.mo_coeff[:,:np.max(determinants[1])+1] 

    det_params = DeterminantParameters(ci_coeff, mo_coeff_up, mo_coeff_down)
    expansion = DeterminantExpansion( mapping[0], mapping[1], 
                                     jnp.array(determinants[0]), 
                                     jnp.array(determinants[1]))
    nelec = tuple(mol.nelec)

    assert(expansion.determinants_up.shape[1] == nelec[0])
    assert(expansion.determinants_down.shape[1] == nelec[1])

    value = partial(evaluate_expansion, gto_ne, expansion, nelec)
    _testvalue_up = partial(testvalue_up, gto_1e, expansion)
    _testvalue_down = partial(testvalue_down, gto_1e, expansion)

    # electron gradient will be testvalue with gradient of gto_1e
    #gto_1e_grad = jax.jacfwd(gto_1e)
    grad_up = partial(testvalue_up, gto_1e_vg, expansion)
    grad_down = partial(testvalue_down, gto_1e_vg, expansion)

    #gto_1e_laplacian = jax.jacfwd(gto_1e_grad)
    laplacian_up = partial(testvalue_up, gto_1e_vgl, expansion)
    laplacian_down = partial(testvalue_down, gto_1e_vgl, expansion)

    # pgradient is derivative of value with respect to ci_coeff and mo_coeff 
    def value_logdet(params, xyz):
        return value(params, xyz)[1]
    pgradient = jax.jacobian(value_logdet, argnums=0)

    # these are for batches in which we want to evaluate the wavefunction at multiple positions
    _testval_up_batch = jax.vmap(_testvalue_up, in_axes = (None, None, None, None, 0), out_axes = 0)
    _testval_down_batch = jax.vmap(_testvalue_down, in_axes = (None, None, None, None, 0), out_axes = 0)


    # compile all the testvalue functions
    testval_funcs = [_testvalue_up, _testvalue_down, grad_up, grad_down, laplacian_up, laplacian_down, _testval_up_batch, _testval_down_batch]
    testval_funcs = ( jax.jit(
                       jax.vmap(f,
                                    in_axes=(None, #parameters
                                    SlaterState(0,0,0,0), 
                                    SlaterState(0,0,0,0), 
                                    None, #electron number
                                    0), #xyz
                                    out_axes=(0,0)
                                      )
                    ) 
                     for f in testval_funcs)
    
    
    value_func = [value]
    value_func = [jax.jit(
                jax.vmap(f, in_axes=(None, #parameters
                                             0), #xyz
                                             out_axes=(0,0,
                                                       SlaterState(0,0,0,0), 
                                                       SlaterState(0,0,0,0)))
                )
                for f in value_func]
    value_func.append(jax.jit(
                jax.vmap(pgradient, in_axes=(None, #parameters
                                             0), #xyz
                                             out_axes=0)
                        )
                )

    # The vmaps here are over configurations
    return det_params, expansion, value_func, testval_funcs


class _parameterMap:
    """
    This class wraps the parameters so that we only need to transfer them to 
    the GPU if they are changed. 

    """
    def __init__(self, jax_parameters):
        self.jax_parameters = jax_parameters
        self.parameters= {'det_coeff': np.asarray(self.jax_parameters[0]), 
                          'mo_coeff_alpha': np.asarray(self.jax_parameters[1]), 
                          'mo_coeff_beta': np.asarray(self.jax_parameters[2]) }

    def __setitem__(self, key, item):
        self.parameters[key] = item
        self.jax_parameters = DeterminantParameters(jnp.array(self.parameters['det_coeff']),
                                                jnp.array(self.parameters['mo_coeff_alpha']),
                                                jnp.array(self.parameters['mo_coeff_beta']))

    def __getitem__(self, key):
        return self.parameters[key]

    def __repr__(self):
        return repr(self.parameters)

    def __len__(self):
        return len(self.parameters)

    def __delitem__(self, key):
        raise NotImplementedError("Cannot delete parameters")

    def clear(self):
        raise NotImplementedError("Cannot clear parameters")

    def copy(self):
        return self.parameters.copy()

    def has_key(self, k):
        return k in self.parameters

    def update(self, *args, **kwargs):
        raise NotImplementedError("Cannot update parameters")

    def keys(self):
        return self.parameters.keys()

    def values(self):
        return self.parameters.values()


class JAXSlater:
    def __init__(self, mol, mf, nimages=2, tol=1e-9, mc = None):
        _parameters, self.expansion, (self._recompute, self._pgradient), \
        (_testvalue_up, _testvalue_down, _grad_up, _grad_down, _lap_up, _lap_down, _batch_up, _batch_down),\
              = create_wf_evaluator(mol, mf, det_tol=tol, nimages=nimages, mc = mc)
        self._testvalue=(_testvalue_up, _testvalue_down)
        self._testvalue_batch = (_batch_up, _batch_down)
        self._grad = (_grad_up, _grad_down)
        self._lap = (_lap_up, _lap_down)
        self._nelec = tuple(mol.nelec)
        self.parameters = _parameterMap(_parameters)
        self.dtype = float
        self._cached_pad_ecp = 0

    def recompute(self, configs):        
        self.xyz = configs.configs.copy()
        xyz = jnp.array(configs.configs)
        self._sign, self._logabs, self._dets_up, self._dets_down = self._recompute(self.parameters.jax_parameters, xyz)
        return self._sign, self._logabs
    
    def value(self):
        """
        Return the value of the wavefunction at the current position
        """
        return self._sign, self._logabs
    
    def updateinternals(self, e, epos, configs, mask=None, saved_values=None):
        """
        Update the internal state of the wavefunction given that electron e has been moved to epos
        """
        spin = int(e >= self._nelec[0] )
        e = e - self._nelec[0]*spin
        if mask is None:
            mask = np.ones((configs.configs.shape[0],), dtype=bool)
        self.xyz[mask, e, :] = epos.configs[mask]

        if saved_values is None:
            self.recompute(configs)
            return
        
        
        if mask is None: 
            mask = jnp.ones((configs.configs.shape[0],), dtype=bool)

        if spin ==0:
            ratio, inverse = sherman_morrison_row(e, self._dets_up.inverse, saved_values, self.expansion.determinants_up)
            newsigns = jnp.sign(ratio)*self._dets_up.sign
            newlogabs = self._dets_up.logabsdet + jnp.log(jnp.abs(ratio))
            inverse = jnp.where(mask[:,jnp.newaxis, jnp.newaxis, jnp.newaxis], inverse, self._dets_up.inverse)
            
            newsigns = jnp.where(mask[:,jnp.newaxis], newsigns, self._dets_up.sign)
            newlogabs = jnp.where(mask[:,jnp.newaxis], newlogabs, self._dets_up.logabsdet)
            self._dets_up = SlaterState(None, newsigns, newlogabs, inverse)

        else:
            ratio, inverse = sherman_morrison_row(e, self._dets_down.inverse, saved_values, self.expansion.determinants_down)
            newsigns = jnp.sign(ratio)*self._dets_down.sign
            newlogabs = self._dets_down.logabsdet + jnp.log(jnp.abs(ratio))
            inverse = jnp.where(mask[:,jnp.newaxis, jnp.newaxis, jnp.newaxis], inverse, self._dets_down.inverse)
            newsigns = jnp.where(mask[:,jnp.newaxis], newsigns, self._dets_down.sign)  # (nconf, ndet)
            newlogabs = jnp.where(mask[:,jnp.newaxis], newlogabs, self._dets_down.logabsdet)
            self._dets_down = SlaterState(None, newsigns, newlogabs, inverse)

        self._sign, self._logabs =  compute_values_from_determinants(self.expansion,
                                                                    self.parameters.jax_parameters,
                                                                    self._dets_up, 
                                                                    self._dets_down)

    

    def testvalue(self, e, epos, mask=None):
        xyz = jnp.array(epos.configs)
        spin = int(e >= self._nelec[0] )
        e = e - self._nelec[0]*spin
        if len(xyz.shape) ==3 and mask is not None: # This is an optimization for ECPs. 
            if True:
                nmask = np.sum(mask)
                if self._cached_pad_ecp < nmask:
                    self._cached_pad_ecp = int(nmask*1.1)
                #print(nmask, self._cached_pad_ecp)
                xyz = jnp.pad(xyz[mask], ((0, self._cached_pad_ecp-nmask), (0,0), (0,0)), mode='empty')
                #print(xyz.shape)
                det_up = SlaterState(None, 
                                     jnp.pad(self._dets_up.sign[mask], ((0, self._cached_pad_ecp-nmask), (0,0)), mode='constant'),
                                     jnp.pad(self._dets_up.logabsdet[mask], ((0, self._cached_pad_ecp-nmask), (0,0)), mode='constant'),
                                     jnp.pad(self._dets_up.inverse[mask], ((0, self._cached_pad_ecp-nmask), (0,0), (0,0), (0,0)), mode='constant'))
                det_down = SlaterState(None,    
                                        jnp.pad(self._dets_down.sign[mask], ((0, self._cached_pad_ecp-nmask), (0,0)), mode='constant'),
                                        jnp.pad(self._dets_down.logabsdet[mask], ((0, self._cached_pad_ecp-nmask), (0,0)), mode='constant'),
                                        jnp.pad(self._dets_down.inverse[mask], ((0, self._cached_pad_ecp-nmask), (0,0), (0,0), (0,0)), mode='constant'))
                newvals, saved = self._testvalue_batch[spin](self.parameters.jax_parameters, det_up, det_down, e, xyz)
                # jax.block_until_ready(newvals)
                # for some reason JAX's mask is very slow.
                ret = np.array(newvals)[:nmask,:], None
                return ret
            else:
                newvals, saved = self._testvalue_batch[spin](self.parameters.jax_parameters, self._dets_up, self._dets_down, e, xyz)
                return np.array(newvals)[mask], saved
        elif len(xyz.shape) == 3:  # for a batch of configurations where there is no mask
            newvals, saved = self._testvalue_batch[spin](self.parameters.jax_parameters, self._dets_up, self._dets_down, e, xyz)
            return newvals, saved
        else: 
            newvals, saved = self._testvalue[spin](self.parameters.jax_parameters, self._dets_up, self._dets_down, e, xyz)
            if mask is not None:
                return np.array(newvals)[mask], saved
            return newvals, saved

    def gradient(self, e, epos, mask=None):
        xyz = jnp.array(epos.configs)
        spin = int(e >= self._nelec[0] )
        e = e - self._nelec[0]*spin
        grad, saved = self._grad[spin](self.parameters.jax_parameters, self._dets_up, self._dets_down, e, xyz) # pyqmc wants (3, nconfig)
        return np.asarray(grad[:,1:].T/grad[:,0])
    

    def gradient_value(self, e, epos):
        xyz = jnp.array(epos.configs)
        spin = int(e >= self._nelec[0] )
        e = e - self._nelec[0]*spin
        derivatives, saved = self._grad[spin](self.parameters.jax_parameters, self._dets_up, self._dets_down, e, xyz) # pyqmc wants (3, nconfig)
        #print("saved", saved.shape)
        convert = derivatives[:,1:].T/derivatives[:,0], derivatives[:,0], saved[:,0, :] 
        return convert


    def gradient_laplacian(self, e, epos):
        xyz = jnp.array(epos.configs)
        spin = int(e >= self._nelec[0] )
        e = e - self._nelec[0]*spin
        laplacian, saved = self._lap[spin](self.parameters.jax_parameters, self._dets_up, self._dets_down, e, xyz)
        return np.array(laplacian[:,1:4].T/laplacian[:,0]), np.asarray(laplacian[:,4]/laplacian[:,0])

    def laplacian(self, e, epos, mask=None):
        xyz = jnp.array(epos.configs)
        spin = int(e >= self._nelec[0] )
        e = e - self._nelec[0]*spin
        lap, saved = self._lap[spin](self.parameters.jax_parameters, self._dets_up, self._dets_down, e, xyz)
        return lap[:,4].T/lap[:,0]
    
    def pgradient(self):
        xyz = jnp.array(self.xyz)
        grads =  self._pgradient(self.parameters.jax_parameters, xyz)  #[1] # sign, log, dets_up, dets_down
        return {'det_coeff': np.array(grads[0]), 
                'mo_coeff_alpha': np.array(grads[1]), 
                'mo_coeff_beta': np.array(grads[2])}
        


