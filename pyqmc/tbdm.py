""" Evaluate the TBDM for a wave function object. """
import numpy as np
from copy import copy
from pyqmc.mc import initial_guess
from sys import stdout

class TBDMAccumulator:
    """ Return the tbdm as an array with indices rho[spin][i][j][k][l] = < c^+_{spin,i} c^+_{spin,k} c_{spin,l} c_{spin,j} > .
    This is in pyscf's notation. ATTENTION: The OBDM is not in pyscf's notation!
        NOTE: We will assume that the localized basis in which the tbdm is written is the same for up and down electrons.

    Args:

      mol (Mole): PySCF Mole object.

      configs (array): electron positions.

      wf (pyqmc wave function object): wave function to evaluate on.

      orb_coeff (array): coefficients with size (nbasis,norb) relating mol basis to basis 
                         of 2-RDM desired.
      
      tstep (float): width of the Gaussian to update two walkers positions for the 
                     two extra coordinates.

      spin: [0,0], [0,1], [1,0] or [1,1] for up-up, up-down, down-up or down-down. Defaults to all electrons.
    """

    def __init__(
        self,
        mol,
        orb_coeff,
        nstep=10,
        tstep=0.50,
        warmup=100,
        naux=500,
        spin=None,
        electrons=None,
        ijkl=None,
    ):
        assert (
            len(orb_coeff.shape) == 2
        ), "orb_coeff should be a list of orbital coefficients."

        if not (spin is None):
            if spin == [0,0]:
                self._electrons1 = np.arange(0, mol.nelec[0])
                self._electrons2 = np.arange(0, mol.nelec[0])
            elif spin == [0,1]:
                self._electrons1 = np.arange(0, mol.nelec[0])
                self._electrons2 = np.arange(mol.nelec[0], np.sum(mol.nelec))
            elif spin == [1,0]:
                self._electrons1 = np.arange(mol.nelec[0], np.sum(mol.nelec))
                self._electrons2 = np.arange(0, mol.nelec[0])
            elif spin == [1,1]:
                self._electrons1 = np.arange(mol.nelec[0], np.sum(mol.nelec))
                self._electrons2 = np.arange(mol.nelec[0], np.sum(mol.nelec))
            else:
                raise ValueError("Spin not equal to [0,0], [0,1], [1,0] or [1,1]")
        elif not (electrons is None):
            self._electrons1 = electrons[0]
            self._electrons2 = electrons[1]
        else:
            self._electrons1 = np.arange(0, np.sum(mol.nelec))
            self._electrons2 = np.arange(0, np.sum(mol.nelec))

        self._orb_coeff = orb_coeff
        self._tstep = tstep
        self._mol = mol
        # self._extra_config = np.random.normal(scale=tstep,size=3) # not zero to avoid sitting on top of atom.
        nelec = sum(self._mol.nelec)
        self._extra_config = initial_guess(mol, 2 * (int(naux / nelec) + 1) ).reshape(-1, 3) # 2 because we have two extra positions

        self._nstep = nstep

        if not (ijkl is None):
            self._ijkl=ijkl
        else:
            aux=np.arange(0,self._orb_coeff.shape[0])
            self._ijkl=np.array(np.meshgrid(aux,aux,aux,aux)).reshape(-1,4) # All entries of the 2rdm  
        
        for i in range(warmup):
            accept, self._extra_config = sample_onebody(
                mol, orb_coeff, self._extra_config, tstep
            )

            
    def __call__(self, configs, wf):
        """ Quantities from equation (10) of DOI:10.1063/1.4793531"""

        results = {
            "value": np.zeros(
                (configs.shape[0], self._orb_coeff.shape[1], self._orb_coeff.shape[1], self._orb_coeff.shape[1], self._orb_coeff.shape[1])
            ),
            "norm": np.zeros((configs.shape[0], self._orb_coeff.shape[1])),
            "acceptance": np.zeros(configs.shape[0]),
        }
        acceptance = 0
        naux = self._extra_config.shape[0]
        nelec1 = len(self._electrons1)
        nelec2 = len(self._electrons2)

        for step in range(self._nstep):
            e1 = np.random.choice(self._electrons1)
            e2 = np.random.choice(self._electrons2[self._electrons2!=e1]) # Cannot repeat electron
            #print('step=%d; e1=%d; e2=%d;'%(step,e1,e2))
            
            points = np.concatenate([self._extra_config, configs[:, e1, :], configs[:, e2, :]])
            ao = self._mol.eval_gto("GTOval_sph", points)
            borb = ao.dot(self._orb_coeff)

            # Orbital evaluations at extra coordinates.
            borb_aux = borb[0:naux, :]
            fsum = np.sum(borb_aux[0:naux,:] * borb_aux[0:naux,:], axis=1)
            norm = borb_aux[0:naux,:] * borb_aux[0:naux,:] / fsum[:, np.newaxis]
            borb_configs1 = borb[naux:(naux+configs.shape[0]), :]
            borb_configs2 = borb[(naux+configs.shape[0]):, :]

            #print(fsum.shape)
            
            # It would be faster to implement a wf.testvalue_2body()
            auxassignments1 = np.random.randint(0, int(naux/2), size=configs.shape[0])
            auxassignments2 = np.random.randint(int(naux/2), naux, size=configs.shape[0])
            #print('ass1:',auxassignments1)
            #print('ass2:',auxassignments2)
            #print('_extra_config[ass1]:',self._extra_config[auxassignments1, :])
            #print('_extra_config[ass2]:',self._extra_config[auxassignments2, :])
            #stdout.flush()
            wfratio1 = wf.testvalue(e1, self._extra_config[auxassignments1, :])
            wf_aux = copy(wf)
            #print('Warning: Shallow copy.')
            wf_aux.updateinternals(e1, self._extra_config[auxassignments1, :])
            wfratio2 = wf_aux.testvalue(e2, self._extra_config[auxassignments2, :])
            wfratio = np.nan_to_num(wfratio1) * np.nan_to_num(wfratio2)
            
            #print('wfratio1:\n',wfratio1)
            #print('wfratio2:\n',wfratio2)
            #print('wfratio:\n',wfratio)
            #stdout.flush()
            #print((borb_aux[auxassignments1, :] / fsum[auxassignments1, np.newaxis]).shape)
            #print((borb_aux[auxassignments2, :] / fsum[auxassignments2, np.newaxis]).shape)
            #print(borb_configs1.shape)
            #print(borb_configs2.shape)
            
            orbratio = np.einsum(
                "ij,ik,il,im->ijklm",
                borb_aux[auxassignments1, :] / fsum[auxassignments1, np.newaxis],
                borb_aux[auxassignments2, :] / fsum[auxassignments2, np.newaxis],
                borb_configs1, borb_configs2
            )


            #print(orbratio.shape)

            #print('Warning: Check if should multiply nelec1 * nelec2.')
            results["value"] += nelec1 * nelec2 * np.einsum("i,ijklm->ijklm", wfratio, orbratio)
            results["norm"] += (norm[auxassignments1] + norm[auxassignments2])/2
            
            accept, self._extra_config = sample_onebody(
                self._mol, self._orb_coeff, self._extra_config, tstep=self._tstep
            )

            results["acceptance"] += np.mean(accept)

        results["value"] /= self._nstep
        results["norm"] = results["norm"] / self._nstep
        results["acceptance"] /= self._nstep

        #print('value:\n',results["value"][0])
        #print('norm:\n',results["norm"])
        
        return results

    def avg(self, configs, wf):
        d = self(configs, wf)
        davg = {}
        for k, v in d.items():
            # print(k, v.shape)
            davg[k] = np.mean(v, axis=0)
        return davg


def sample_onebody(mol, orb_coeff, configs, tstep=2.0):
    """ For a set of orbitals defined by orb_coeff, return samples from f(r) = \sum_i phi_i(r)^2. """
    config_pack = np.concatenate(
        [configs, configs + np.sqrt(tstep) * np.random.randn(*configs.shape)], axis=0
    )

    ao = mol.eval_gto("GTOval_sph", config_pack)
    borb = ao.dot(orb_coeff)
    fsum = (borb ** 2).sum(axis=1)

    n = configs.shape[0]
    accept = fsum[n:] / fsum[0:n] > np.random.rand(n) 
    newconf = config_pack[n:, :]
    configs[accept, :] = newconf[accept, :]
    return accept, configs


def normalize_tbdm(tbdm, norm):
    return tbdm / (norm[np.newaxis, np.newaxis, np.newaxis, :] * norm[np.newaxis, np.newaxis, :, np.newaxis] * norm[np.newaxis, :, np.newaxis, np.newaxis] * norm[:, np.newaxis, np.newaxis, np.newaxis]) ** 0.5 









