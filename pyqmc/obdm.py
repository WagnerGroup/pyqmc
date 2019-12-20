""" Evaluate the OBDM for a wave function object. """
import numpy as np
from copy import deepcopy
from pyqmc.mc import initial_guess


class OBDMAccumulator:
    """ Return the obdm as an array with indices rho[spin][i][k] = <c_{spin,i}c^+_{spin,j}>
  Args:

    mol (Mole): PySCF Mole object.

    configs (array): electron positions.

    wf (pyqmc wave function object): wave function to evaluate on.

    orb_coeff (array): coefficients with size (nbasis,norb) relating mol basis to basis 
      of 1-RDM desired.
      
    tstep (float): width of the Gaussian to update a walker position for the 
      extra coordinate.

    spin: 0 or 1 for up or down. Defaults to all electrons.
  """

    def __init__(
        self,
        mol,
        orb_coeff,
        nsweeps=5,
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
                self._electrons = np.arange(0, mol.nelec[0])
            elif spin == 1:
                self._electrons = np.arange(mol.nelec[0], np.sum(mol.nelec))
            else:
                raise ValueError("Spin not equal to 0 or 1")
        elif not (electrons is None):
            self._electrons = electrons
        else:
            self._electrons = np.arange(0, np.sum(mol.nelec))

        self._orb_coeff = orb_coeff
        self._tstep = tstep
        self._mol = mol

        nelec = len(self._electrons)
        self._extra_config = initial_guess(mol, int(naux / nelec) + 1).configs.reshape(
            -1, 3
        )

        self._nsweeps = nsweeps
        self._nstep = nsweeps * nelec

        for i in range(warmup):
            accept, self._extra_config = sample_onebody(
                mol, orb_coeff, self._extra_config, tstep
            )

    def __call__(self, configs, wf, extra_configs=None, auxassignments=None):
        """ Quantities from equation (9) of DOI:10.1063/1.4793531"""

        nconf = configs.configs.shape[0]
        results = {
            "value": np.zeros(
                (nconf, self._orb_coeff.shape[1], self._orb_coeff.shape[1])
            ),
            "norm": np.zeros((nconf, self._orb_coeff.shape[1])),
            "acceptance": np.zeros(nconf),
        }
        acceptance = 0
        naux = self._extra_config.shape[0]
        nelec = len(self._electrons)
        e = np.repeat(self._electrons, self._nsweeps)

        if extra_configs is None:
            auxassignments = np.random.randint(0, naux, size=(self._nstep, nconf))
            extra_configs = []
            for step in range(self._nstep):
                extra_configs.append(np.copy(self._extra_config))
                accept, self._extra_config = sample_onebody(
                    self._mol, self._orb_coeff, self._extra_config, tstep=self._tstep
                )
                results["acceptance"] += np.mean(accept)
        else:
            assert auxassignments is not None

        for step in range(self._nstep):
            points = np.concatenate(
                [extra_configs[step], configs.configs[:, e[step], :]]
            )
            ao = self._mol.eval_gto("GTOval_sph", points)
            borb = ao.dot(self._orb_coeff)

            # Orbital evaluations at extra coordinate.
            borb_aux = borb[0:naux, :]
            fsum = np.sum(borb_aux * borb_aux, axis=1)
            norm = borb_aux * borb_aux / fsum[:, np.newaxis]
            borb_configs = borb[naux:, :]

            epos = configs.make_irreducible(
                e[step], extra_configs[step][auxassignments[step]]
            )
            wfratio = wf.testvalue(e[step], epos)

            orbratio = np.einsum(
                "ij,ik->ijk",
                borb_aux[auxassignments[step], :]
                / fsum[auxassignments[step], np.newaxis],
                borb_configs,
            )

            results["value"] += nelec * np.einsum("i,ijk->ijk", wfratio, orbratio)
            results["norm"] += norm[auxassignments[step]]

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

    def get_extra_configs(self, configs):
        """ Returns an nstep length array of configurations
            starting from self._extra_config """
        nconf = configs.configs.shape[0]
        naux = self._extra_config.shape[0]
        extra_configs = []
        auxassignments = np.random.randint(0, naux, size=(self._nstep, nconf))
        for step in range(self._nstep):
            extra_configs.append(np.copy(self._extra_config))
            accept, self._extra_config = sample_onebody(
                self._mol, self._orb_coeff, self._extra_config, tstep=self._tstep
            )
        return extra_configs, auxassignments


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


def normalize_obdm(obdm, norm):
    return obdm / (norm[np.newaxis, :] * norm[:, np.newaxis]) ** 0.5
