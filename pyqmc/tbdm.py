""" Evaluate the TBDM for a wave function object. """
import numpy as np
from copy import copy, deepcopy
from pyqmc.mc import initial_guess
from pyqmc.obdm import sample_onebody
from sys import stdout
from pyqmc.orbitals import MoleculeOrbitalEvaluator, PBCOrbitalEvaluatorKpoints


class TBDMAccumulator:
    """ Returns one spin sector of the tbdm[s1,s2] as an array (norb_s1,norb_s1,norb_s2,norb_s2) with indices (using pySCF's 
    convention): tbdm[s1,s2][i,j,k,l] = < c^+_{s1,i} c^+_{s2,k} c_{s2,l} c_{s1,j} > = \phi*_{s1,j} \phi*_{s2,l} \phi_{s2,k} \phi_{s1,i}.


    We use pySCF's index convention (while Eq. 10 in DOI:10.1063/1.4793531 uses QWalk's)
    QWalk -> tbdm[s1,s2,i,j,k,l] = < c^+_{s1,i} c^+_{s2,j} c_{s2,l} c_{s1,k} > = \phi*_{s1,k} \phi*_{s2,l} \phi_{s2,j} \phi_{s1,i}
    pySCF -> tbdm[s1,s2,i,j,k,l] = < c^+_{s1,i} c^+_{s2,k} c_{s2,l} c_{s1,j} > = \phi*_{s1,j} \phi*_{s2,l} \phi_{s2,k} \phi_{s1,i}

    Args:

      mol (Mole): PySCF Mole object.

      orb_coeff (array): coefficients with size (2,nbasis,norb) relating mol basis to basis
        of the 2-RDM.

      nsweeps (int): number of sweeps over all the electron pairs.

      tstep (float): width of the Gaussian to update a walker position for each extra coordinate.

      warmup (int): number of warmup steps for single-particle local orbital sampling.

      naux (int): number of auxiliary configurations for extra moves sampling the local
         orbitals.  

      spin (tuple): size 2 indicates spin sector to be computed, either (0,0), (0,1), (1,0) or (1,1)
         for up-up, up-down, down-up or down-down. 

      ijkl (array): contains M tbdm matrix elements to calculate with dim (M,4). 
    """

    def __init__(
        self,
        mol,
        orb_coeff,
        spin,
        nsweeps=4,
        tstep=0.50,
        warmup=200,
        naux=500,
        ijkl=None,
        kpts=None
    ):
        assert (
            len(orb_coeff.shape) == 3
        ), "orb_coeff should be a list of orbital coefficients with size (2,num_mobasis,num_orb)."

        self._mol = mol
        self._tstep = tstep
        self._nsweeps = nsweeps
        self._spin = spin

        if kpts is None:
            self.orbitals = MoleculeOrbitalEvaluator(mol, orb_coeff)
        else: 
            self.orbitals = PBCOrbitalEvaluatorKpoints(mol, orb_coeff, kpts)

        self._spin_sector = spin
        self._electrons=[np.arange(spin[s] * mol.nelec[0], mol.nelec[0] + spin[s] * mol.nelec[1]) for s in [0,1]]
        self._pairs = np.array(
            np.meshgrid(self._electrons[0], self._electrons[1])
        ).T.reshape(-1, 2)
        self._pairs = self._pairs[
            self._pairs[:, 0] != self._pairs[:, 1]
        ]  # Removes repeated electron pairs

        # Initialization and warmup of configurations
        nwalkers = int(naux / sum(self._mol.nelec))
        self._aux_configs=[]
        for spin in [0,1]:
            self._aux_configs.append(initial_guess(mol, nwalkers))
            self._aux_configs[spin].reshape((-1,1, 3))
            _, self._aux_configs[spin], _ = \
                sample_onebody(self._aux_configs[spin], self.orbitals, 0, nsamples=warmup)
            self._aux_configs[spin] = self._aux_configs[spin][-1]


        # Default to full 2rdm if ijkl not specified
        if ijkl is None:
            norb_up = orb_coeff[0].shape[1]
            norb_down = orb_coeff[1].shape[1]
            ijkl = [
                [i, j, k, l]
                for i in range(norb_up)
                for j in range(norb_up)
                for k in range(norb_down)
                for l in range(norb_down)
            ]
        self._ijkl = np.array(ijkl).T

    def get_configurations(self, nconf):
        """
        Obtain a sequence of auxilliary configurations with assignments.
        Changes internal state: self._aux_configs is updated to the last sampled location. 

        returns a dictionary with the following elements, separated by spin:
            assignments: [nsweeps, nconf]: assign configurations for each sweep to an auxilliary walker.
            orbs: [nsweeps, naux, norb]: orbital values
            configs: [nsweeps] Configuration object with naux configurations of 1 electron
            acceptance: [nsweeps, naux] acceptance probability for each 
        """
        configs=[]
        assignments = []
        orbs = []
        acceptance = []
        for spin in [0,1]:
            accept, tmp_config, tmp_orbs = sample_onebody(self._aux_configs[spin], self.orbitals, 0, self._nsweeps, tstep=self._tstep)
            acceptance.append(accept)
            configs.append(tmp_config)
            orbs.append(tmp_orbs)
            self._aux_configs[spin] = configs[spin][-1]
            configs[spin] = np.array(configs[spin])
            naux = self._aux_configs[spin].configs.shape[0]
            assignments.append(np.random.randint(0, naux, size=(self._nsweeps, nconf)))

        return {'acceptance':acceptance, 'orbs':orbs, 'configs':configs, 'assignments':assignments}


    def __call__(self, configs, wf, extra_configs=None, auxassignments=None):
        """Gathers quantities from equation (10) of DOI:10.1063/1.4793531."""

        nconf = configs.configs.shape[0]
        if extra_configs is not None:
            raise NotImplementedError("have not implemented extra_configs")

        aux = self.get_configurations(nconf)

        # Evaluate orbital values for the primary samples
        ao_configs = self.orbitals.aos("GTOval_sph", configs)
        ao_configs = ao_configs.reshape((ao_configs.shape[0], nconf, -1, ao_configs.shape[-1]))
        orb_configs = [self.orbitals.mos(ao_configs[...,self._electrons[spin], :], spin) for spin in [0,1]]
        results = {"value": np.zeros((nconf, self._ijkl.shape[1])),
            'norm_a': np.zeros((nconf, orb_configs[0].shape[-1])),
            'norm_b': np.zeros((nconf, orb_configs[1].shape[-1]))}

        orb_configs = [orb_configs[s][:,:, self._ijkl[2*s]] for s in [0,1]]

        # Sweeps over electron pairs
        down_start = [np.min(self._electrons[s]) for s in [0,1]]
        for sweep in range(self._nsweeps):
            fsum = [np.sum(np.abs(aux['orbs'][spin][sweep])**2, axis=1) for spin in [0,1]]
            norm = [np.abs(aux['orbs'][spin][sweep])**2/fsum[spin][:,np.newaxis] for spin in [0,1]]

            """
            orbratio collects 
            phi_i(r1) phi_j(r1') phi_k(r2) phi_l(r2')/rho(r1') rho(r2')
            """
            phi_j_r1p = aux['orbs'][0][sweep][aux['assignments'][0][sweep]][:, self._ijkl[1]]
            phi_l_r2p = aux['orbs'][1][sweep][aux['assignments'][1][sweep]][:, self._ijkl[3]]
            rho1rho2 = fsum[0][aux['assignments'][0][sweep]]*fsum[1][aux['assignments'][1][sweep]]

            wfratio = []
            electrons_a_ind = []
            electrons_b_ind = []
            for ea in self._electrons[0]:
                # Don't move the same electron twice
                electrons_b = self._electrons[1][self._electrons[1] != ea]
                epos_a = aux['configs'][0][sweep].electron(0)
                epos_b = aux['configs'][1][sweep].electron(0)
                wfratio_a = wf.testvalue(ea, epos_a)
                wf.updateinternals(ea, epos_a)
                wfratio_b = wf.testvalue_many(electrons_b, epos_b)
                wf.updateinternals(ea, configs.electron(ea))
                wfratio.append(wfratio_a[:, np.newaxis] * wfratio_b)
                electrons_a_ind.extend([ea-down_start[0]]*len(electrons_b))
                electrons_b_ind.extend(electrons_b-down_start[1])


            wfratio = np.concatenate(wfratio, axis=1)
            orbratio = np.einsum("nio,nio,no,no ->nio", 
                                 orb_configs[0][:,electrons_a_ind,:], 
                                 orb_configs[1][:,electrons_b_ind,:],
                                 phi_j_r1p,
                                 phi_l_r2p)/rho1rho2[:,np.newaxis, np.newaxis]

            # Adding to results
            results["value"] += np.einsum("in,inj->ij", wfratio, orbratio)
            results["norm_a"] += norm[0]
            results["norm_b"] += norm[1]

        # Average over sweeps and pairs
        results["value"] /= self._nsweeps
        for e in ["a", "b"]:
            results["norm_%s" % e] /= self._nsweeps

        return results

    def keys(self):
        return set(
            ["value", "norm_a", "norm_b", "acceptance_a", "acceptance_b", "ijkl"]
        )

    def shapes(self):
        d = {"value": (self._ijkl.shape[1],), "ijkl": self._ijkl.T.shape}
        for e, s in zip(["a", "b"], self._spin_sector):
            d["norm_%s" % e] = (self._orb_coeff[s].shape[1],)
            d["acceptance_%s" % e] = ()
        return d

    def avg(self, configs, wf):
        d = {k: np.mean(it, axis=0) for k, it in self(configs, wf).items()}
        d["ijkl"] = self._ijkl.T
        return d

    def get_extra_configs(self, configs):
        """ Returns an nstep length array of configurations
        starting from self._extra_config """
        nconf = configs.configs.shape[0]

        aux_configs_a = []
        aux_configs_b = []
        for step in range(self._nsweeps):
            aux_configs_a.append(np.copy(self._aux_configs_a))
            accept_a, self._aux_configs_a = sample_onebody(
                self._mol,
                self._orb_coeff[self._spin_sector[0]],
                self._aux_configs_a,
                tstep=self._tstep,
            )
            aux_configs_b.append(np.copy(self._aux_configs_b))
            accept_b, self._aux_configs_b = sample_onebody(
                self._mol,
                self._orb_coeff[self._spin_sector[1]],
                self._aux_configs_b,
                tstep=self._tstep,
            )
        aux_configs_a = np.array(aux_configs_a)
        aux_configs_b = np.array(aux_configs_b)

        # Generates random choice of aux_config_a and aux_config_b for moving electron_a and electron_b
        naux_a = self._aux_configs_a.shape[0]
        naux_b = self._aux_configs_b.shape[0]
        auxassignments_a = np.random.randint(0, naux_a, size=(self._nsweeps, nconf))
        auxassignments_b = np.random.randint(0, naux_b, size=(self._nsweeps, nconf))
        return [aux_configs_a, aux_configs_b], [auxassignments_a, auxassignments_b]


def normalize_tbdm(tbdm, norm_a, norm_b):
    """Returns tbdm by taking the ratio of the averages in Eq. (10) of DOI:10.1063/1.4793531."""
    # We are using pySCF's notation:
    #  tbdm[s1,s2,i,j,k,l] = < c^+_{s1,i} c^+_{s2,k} c_{s2,l} c_{s1,j} > = \phi*_{s1,j} \phi*_{s2,l} \phi_{s2,k} \phi_{s1,i}
    return tbdm / np.einsum("i,j,k,l->ijkl", norm_a, norm_a, norm_b, norm_b) ** 0.5
