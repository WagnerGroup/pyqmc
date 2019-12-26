""" Evaluate the TBDM for a wave function object. """
import numpy as np
from copy import copy, deepcopy
from pyqmc.mc import initial_guess
from pyqmc.obdm import sample_onebody
from sys import stdout


class TBDMAccumulator:
    """ Returns one spin sector of the tbdm[s1,s2] as an array (norb_s1,norb_s1,norb_s2,norb_s2) with indices (using pySCF's 
    convention): tbdm[s1,s2][i,j,k,l] = < c^+_{s1,i} c^+_{s2,k} c_{s2,l} c_{s1,j} > = \phi*_{s1,j} \phi*_{s2,l} \phi_{s2,k} \phi_{s1,i}.

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
    ):
        assert (
            len(orb_coeff.shape) == 3
        ), "orb_coeff should be a list of orbital coefficients with size (2,num_mobasis,num_orb)."

        self._mol = mol
        self._orb_coeff = orb_coeff
        self._tstep = tstep
        self._nsweeps = nsweeps
        self._spin = spin

        self._spin_sector = spin
        self._electrons_a = np.arange(
            spin[0] * mol.nelec[0], mol.nelec[0] + spin[0] * mol.nelec[1]
        )
        self._electrons_b = np.arange(
            spin[1] * mol.nelec[0], mol.nelec[0] + spin[1] * mol.nelec[1]
        )
        self._pairs = np.array(
            np.meshgrid(self._electrons_a, self._electrons_b)
        ).T.reshape(-1, 2)
        self._pairs = self._pairs[
            self._pairs[:, 0] != self._pairs[:, 1]
        ]  # Removes repeated electron pairs

        # Initialization and warmup of aux_configs_a
        self._aux_configs_a = initial_guess(
            mol, int(naux / sum(self._mol.nelec))
        ).configs.reshape(-1, 3)
        for i in range(warmup):
            accept_a, self._aux_configs_a = sample_onebody(
                mol, orb_coeff[self._spin_sector[0]], self._aux_configs_a, tstep
            )
        # Initialization and warmup of aux_configs_b
        self._aux_configs_b = initial_guess(
            mol, int(naux / sum(self._mol.nelec))
        ).configs.reshape(-1, 3)
        for i in range(warmup):
            accept_b, self._aux_configs_b = sample_onebody(
                mol, orb_coeff[self._spin_sector[1]], self._aux_configs_b, tstep
            )
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

    def __call__(self, configs, wf, extra_configs=None, auxassignments=None):
        """Gathers quantities from equation (10) of DOI:10.1063/1.4793531."""

        # Constructs results dictionary
        nconf = configs.configs.shape[0]
        results = {}
        orb_a_size = self._orb_coeff[self._spin_sector[0]].shape[1]
        orb_b_size = self._orb_coeff[self._spin_sector[1]].shape[1]
        results["value"] = np.zeros((nconf, self._ijkl.shape[1]))
        for i, e in enumerate(["a", "b"]):
            results["norm_%s" % e] = np.zeros(
                (nconf, self._orb_coeff[self._spin_sector[i]].shape[1])
            )
            results["acceptance_%s" % e] = np.zeros(nconf)

        # Returns empty arrays if no electron pairs
        if len(self._pairs) == 0:
            return results

        if extra_configs is None:
            # Generates aux_configs_a and aux_configs_b
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
                results["acceptance_a"] += np.mean(accept_a)
                results["acceptance_b"] += np.mean(accept_b)
            results["acceptance_a"] /= self._nsweeps
            results["acceptance_b"] /= self._nsweeps
            aux_configs_a = np.array(aux_configs_a)
            aux_configs_b = np.array(aux_configs_b)
            # Generates random choice of aux_config_a and aux_config_b for moving electron_a and electron_b
            naux_a = self._aux_configs_a.shape[0]
            naux_b = self._aux_configs_b.shape[0]
            auxassignments_a = np.random.randint(0, naux_a, size=(self._nsweeps, nconf))
            auxassignments_b = np.random.randint(0, naux_b, size=(self._nsweeps, nconf))
        else:
            assert auxassignments is not None
            aux_configs_a = extra_configs[0]
            aux_configs_b = extra_configs[1]
            naux_a = self._aux_configs_a.shape[0]
            naux_b = self._aux_configs_b.shape[0]
            auxassignments_a = auxassignments[0]
            auxassignments_b = auxassignments[1]

        # Evaluate VMC configurations
        coords = configs.configs.reshape(
            (configs.configs.shape[0] * configs.configs.shape[1], -1)
        )
        ao_configs = self._mol.eval_gto("GTOval_sph", coords)
        orb_a_configs = ao_configs.dot(self._orb_coeff[self._spin_sector[0]]).reshape(
            (configs.configs.shape[0], configs.configs.shape[1], -1)
        )
        orb_b_configs = ao_configs.dot(self._orb_coeff[self._spin_sector[1]]).reshape(
            (configs.configs.shape[0], configs.configs.shape[1], -1)
        )
        orb_a_configs = orb_a_configs[:, self._pairs[:, 0], :]
        orb_b_configs = orb_b_configs[:, self._pairs[:, 1], :]

        # Sweeps over electron pairs
        for sweep in range(self._nsweeps):
            ao_a_aux = self._mol.eval_gto("GTOval_sph", aux_configs_a[sweep])
            ao_b_aux = self._mol.eval_gto("GTOval_sph", aux_configs_b[sweep])
            orb_a_aux = ao_a_aux.dot(self._orb_coeff[self._spin_sector[0]])
            orb_b_aux = ao_b_aux.dot(self._orb_coeff[self._spin_sector[1]])
            fsum_a = np.sum(orb_a_aux * orb_a_aux, axis=1)
            fsum_b = np.sum(orb_b_aux * orb_b_aux, axis=1)
            norm_a = orb_a_aux * orb_a_aux / fsum_a[:, np.newaxis]
            norm_b = orb_b_aux * orb_b_aux / fsum_b[:, np.newaxis]

            # We use pySCF's index convention (while Eq. 10 in DOI:10.1063/1.4793531 uses QWalk's)
            # QWalk -> tbdm[s1,s2,i,j,k,l] = < c^+_{s1,i} c^+_{s2,j} c_{s2,l} c_{s1,k} > = \phi*_{s1,k} \phi*_{s2,l} \phi_{s2,j} \phi_{s1,i}
            # pySCF -> tbdm[s1,s2,i,j,k,l] = < c^+_{s1,i} c^+_{s2,k} c_{s2,l} c_{s1,j} > = \phi*_{s1,j} \phi*_{s2,l} \phi_{s2,k} \phi_{s1,i}
            orbratio = (
                (
                    orb_a_aux[auxassignments_a[sweep]][:, self._ijkl[1]]
                    / fsum_a[auxassignments_a[sweep], np.newaxis]
                )[:, np.newaxis, :]
                * (
                    orb_b_aux[auxassignments_b[sweep]][:, self._ijkl[3]]
                    / fsum_b[auxassignments_b[sweep], np.newaxis]
                )[:, np.newaxis, :]
                * orb_a_configs[..., self._ijkl[0]]
                * orb_b_configs[..., self._ijkl[2]]
            )

            # Calculation of wf ratio (no McMillan trick yet)
            epos_a = configs.make_irreducible(
                -1, aux_configs_a[sweep][auxassignments_a[sweep]]
            )
            epos_b = configs.make_irreducible(
                -1, aux_configs_b[sweep][auxassignments_b[sweep]]
            )

            wfratio = []
            for ea in self._electrons_a:
                electrons_b = self._electrons_b[self._electrons_b != ea]
                wfratio_a = wf.testvalue(ea, epos_a)
                wf.updateinternals(ea, epos_a)
                wfratio_b = wf.testvalue_many(electrons_b, epos_b)
                wf.updateinternals(ea, configs.electron(ea))
                wfratio.append(wfratio_a[:, np.newaxis] * wfratio_b)
            wfratio = np.concatenate(wfratio, axis=1)

            # Adding to results
            results["value"] += np.einsum("in,inj->ij", wfratio, orbratio)
            results["norm_a"] += norm_a[auxassignments_a[sweep]]
            results["norm_b"] += norm_b[auxassignments_b[sweep]]

        # Average over sweeps and pairs
        results["value"] /= self._nsweeps
        for e in ["a", "b"]:
            results["norm_%s" % e] /= self._nsweeps

        return results

    def avg(self, configs, wf):
        d = self(configs, wf)
        davg = {}
        for k, v in d.items():
            # print(k, v.shape)
            davg[k] = np.mean(v, axis=0)
        davg["ijkl"] = self._ijkl.T
        return davg

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
