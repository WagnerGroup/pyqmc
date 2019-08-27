""" Evaluate the TBDM for a wave function object. """
import numpy as np
from copy import deepcopy
from pyqmc.mc import initial_guess

class TBDMAccumulator:
    """ Return the tbdm as an array with indices rho[spin][i,j,k,l] = <c^+_{spina,i}c^+_{spinb,j}c_{spina,k}c_{spinb,l}>
        where spina, spinb from up, dn
    Args:

    mol (Mole): PySCF Mole object.

    configs (array): electron positions.

    wf (pyqmc wave function object): wave function to evaluate on.

    orb_coeff (array): coefficients with size (nbasis,norb) relating mol basis to basis 
      of 1-RDM desired.
      
    tstep (float): width of the Gaussian to update a walker position for the 
      extra coordinate.

    spin: 0, 1, 2, 3 for upup, updn, dnup, dndn respectively
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
    ):  
        assert (
            len(orb_coeff.shape) == 2
        ), "orb_coeff should be a list of orbital coefficients."
        if not (spin is None):
            if spin == 0:
                self._electrons = (np.arange(0, mol.nelec[0]), np.arange(0, mol.nelec[0]))
            elif spin == 1:
                self._electrons = (np.arange(0, mol.nelec[0]), np.arange(mol.nelec[0], np.sum(mol.nelec)))
            elif spin == 2:
                self._electrons = (np.arange(mol.nelec[0], np.sum(mol.nelec)), np.arange(0, mol.nelec[0]))
            elif spin == 3:
                self._electrons = (np.arange(mol.nelec[0], np.sum(mol.nelec)), np.arange(mol.nelec[0], np.sum(mol.nelec)))
            else:
                raise ValueError("Spin not equal to 0, 1, 2, or 3")
        elif not (electrons is None):
            self._electrons = electrons
        else:
            self._electrons = (np.arange(0, np.sum(mol.nelec)),np.arange(0, np.sum(mol.nelec)))
        self._orb_coeff = orb_coeff
        self._tstep = tstep
        self._mol = mol
        nelec = sum(self._mol.nelec)
        self._extra_config = np.asarray(
          [initial_guess(mol, int(naux / nelec) + 1).reshape(-1, 3),
          initial_guess(mol, int(naux / nelec) + 1).reshape(-1, 3)])
        
        self._nstep = nstep
        for i in range(warmup):
            accept, self._extra_config = sample_twobody(
                mol, orb_coeff, self._extra_config, tstep
            )

    def __call__(self, configs, wf):
        """ Quantities from equation (9) of DOI:10.1063/1.4793531"""
        
        results = {
            "value": np.zeros(
                (configs.shape[0], self._orb_coeff.shape[1], self._orb_coeff.shape[1],
                self._orb_coeff.shape[1], self._orb_coeff.shape[1])
            ),
            "norm": np.zeros((configs.shape[0], self._orb_coeff.shape[1])),
            "acceptance": np.zeros(configs.shape[0]),
        }
        naux = self._extra_config.shape[1]
        nelec = len(self._electrons[0]) + len(self._electrons[1])
        for step in range(self._nstep):
            e1 = np.random.choice(self._electrons[0])
            e2 = np.random.choice(self._electrons[1])

            points1 = np.concatenate([self._extra_config[0], configs[:, e1, :]])
            points2 = np.concatenate([self._extra_config[1], configs[:, e2, :]])
            
            ao1 = self._mol.eval_gto("GTOval_sph", points1)
            ao2 = self._mol.eval_gto("GTOval_sph", points2)
            
            borb1 = ao1.dot(self._orb_coeff)
            borb2 = ao2.dot(self._orb_coeff)

            # Orbital evaluations at extra coordinates.
            borb_aux1 = borb1[0:naux, :]
            borb_aux2 = borb2[0:naux, :]

            fsum1 = np.sum(borb_aux1 * borb_aux1, axis=1)
            norm1 = borb_aux1 * borb_aux1 / fsum1[:, np.newaxis]

            fsum2 = np.sum(borb_aux2 * borb_aux2, axis=1)
            norm2 = borb_aux2 * borb_aux2 / fsum2[:, np.newaxis]
            
            borb_configs1 = borb1[naux:, :]
            borb_configs2 = borb2[naux:, :]

            auxassignments = np.random.randint(0, naux, size=configs.shape[0])

            wfratio1 = wf.testvalue(e1, self._extra_config[0][auxassignments, :])
            wf.updateinternals(e1, self._extra_config[0][auxassignments,  :])
            wfratio2 = wf.testvalue(e2, self._extra_config[1][auxassignments, :])
            wf.updateinternals(e1, configs[:,e1,:])
            wfratio = wfratio2*wfratio1

            #Quantity evaluation
            orbratio1 = np.einsum(
                "ci,ck->cik",
                borb_aux1[auxassignments, :] / fsum1[auxassignments, np.newaxis],
                borb_configs1,
            )
            orbratio2 = np.einsum(
                "cj,cl->cjl",
                borb_aux2[auxassignments, :] / fsum2[auxassignments, np.newaxis],
                borb_configs2,
            )
            orbratio = np.einsum(
                "cik,cjl -> cijkl",orbratio1,orbratio2
            )

            results["value"] += (nelec - 1) * nelec * np.einsum("c,cijkl->cijkl", wfratio, orbratio)/2
            results["norm"] += (norm1[auxassignments] + norm2[auxassignments])/2

            accept, self._extra_config = sample_twobody(
                self._mol, self._orb_coeff, self._extra_config, tstep=self._tstep
            )

            results["acceptance"] += np.mean(accept)

        results["value"] /= self._nstep
        results["norm"] = results["norm"] / self._nstep
        results["acceptance"] /= self._nstep

        return results

    def avg(self, configs, wf):
        d = self(configs, wf)
        davg = {}
        for k, v in d.items():
            # print(k, v.shape)
            davg[k] = np.mean(v, axis=0)
        return davg

def sample_twobody(mol, orb_coeff, double_configs, tstep=2.0):
    from pyqmc.obdm import sample_onebody
    """ For a set of orbitals defined by orb_coeff, return samples from f(r) = \sum_i phi_i(r)^2. """
    return_configs = []
    return_accept = []
    
    for configs in double_configs:
      accept, new_configs = sample_onebody(mol,orb_coeff,configs,tstep)
      return_configs.append(configs)
      return_accept.append(accept)

    return np.concatenate(return_accept), np.array(return_configs)

def normalize_tbdm(tbdm, norm):
    denom = norm[np.newaxis, :] * norm[:, np.newaxis]
    denom = denom[np.newaxis, :] * norm[:, np.newaxis]
    denom = denom[np.newaxis, :] * norm[:, np.newaxis]
    return tbdm/(denom) ** 0.5
