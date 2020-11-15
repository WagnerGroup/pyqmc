""" Evaluate the OBDM for a wave function object. """
import numpy as np
from copy import deepcopy
from pyqmc.mc import initial_guess
from pyqmc import pbc, slater


class OBDMAccumulator:
    r""" Return the obdm as an array with indices rho[spin][i][j] = <c_{spin,i}c^+_{spin,j}>

  .. math:: \rho^\sigma_{ij} = \langle c_{\sigma, i} c^\dagger_{\sigma, j} \rangle

  We are measuring the amplitude of moving one electron (e.g. the first one) from orbital :math:`\phi_i` to orbital :math:`\phi_j` (Eq (7) of DOI:10.1063/1.4793531)

  .. math:: \rho_{i,k} = \int dR dr' \Psi^*(R') \phi_k(r') \phi_i^*(r) \Psi(R)

  (The complex conjugate is on the wrong orbital in Eq (7) in the paper.) Sampling :math:`R` from :math:`|\Psi(R)^2|` and :math:`r'` from :math:`f(r) = \sum_i |\phi(r)|^2`

  .. math:: \rho_{i,k} = \int dR dr' \frac{\Psi^*(R')}{\Psi^*(R)} \left[\Psi^*(R) \Psi(R)\right] \frac{\phi_k(r') \phi_i^*(r)}{f(r)} \left[f(r)\right]

  The distributions (in square brackets) are accounted for by the Monte Carlo integration

  .. math:: \rho_{i,k} = \left\langle \frac{\Psi^*(R')}{\Psi^*(R)} \frac{\phi_k(r') \phi_i^*(r)}{f(r)} \right\rangle

  Eq (9) in the paper is the complex conjugate of this

  .. math:: \rho_{i,k}^* = \left\langle \frac{\Psi(R')}{\Psi(R)} \frac{\phi_k^*(r') \phi_i(r)}{f(r)} \right\rangle


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
        kpts=None,
    ):

        if spin is not None:
            if spin == 0:
                self._electrons = np.arange(0, mol.nelec[0])
            elif spin == 1:
                self._electrons = np.arange(mol.nelec[0], np.sum(mol.nelec))
            else:
                raise ValueError("Spin not equal to 0 or 1")
        elif electrons is not None:
            self._electrons = electrons
        else:
            self._electrons = np.arange(0, np.sum(mol.nelec))

        self.iscomplex = bool(sum(map(np.iscomplexobj, orb_coeff)))
        if hasattr(mol, "lattice_vectors"):
            assert kpts is not None
            assert len(orb_coeff) == len(kpts)
            self.iscomplex = self.iscomplex or np.linalg.norm(kpts) > 1e-12
            self._kpts = kpts
            self.evaluate_orbitals = self._evaluate_orbitals_pbc
            if self.iscomplex:
                self.get_wrapphase = slater.get_wrapphase_complex
            else:
                self.get_wrapphase = slater.get_wrapphase_real
            for attribute in ["original_cell", "S"]:
                if not hasattr(mol, attribute):
                    from pyqmc.supercell import get_supercell

                    mol = get_supercell(mol, np.eye(3))
            self.supercell = mol
            self._mol = mol.original_cell
        else:
            self._mol = mol
            self.evaluate_orbitals = self._evaluate_orbitals
            assert (
                len(orb_coeff.shape) == 2
            ), "orb_coeff should be a list of orbital coefficients."

        self._orb_coeff = orb_coeff
        if hasattr(self._orb_coeff, "shape"):
            self.norb = self._orb_coeff.shape[1]
        else:
            self.norb = sum(c.shape[1] for c in self._orb_coeff)
        self._tstep = tstep
        self.nelec = len(self._electrons)
        self._nsweeps = nsweeps
        self._nstep = nsweeps * self.nelec

        self._extra_config = initial_guess(mol, int(naux / self.nelec) + 1)
        self._extra_config.reshape((-1, 3))

        accept, extra_configs = self.sample_onebody(self._extra_config, warmup)
        self._extra_config = extra_configs[-1]

    def __call__(self, configs, wf, extra_configs=None, auxassignments=None):
        """ 
        """

        nconf = configs.configs.shape[0]
        dtype = complex if self.iscomplex else float
        results = {
            "value": np.zeros((nconf, self.norb, self.norb), dtype=dtype),
            "norm": np.zeros((nconf, self.norb)),
            "acceptance": np.zeros(nconf),
        }
        acceptance = 0
        naux = self._extra_config.configs.shape[0]

        if extra_configs is None:
            auxassignments = np.random.randint(0, naux, size=(self._nsweeps, nconf))
            accept, extra_configs = self.sample_onebody(
                self._extra_config, self._nsweeps
            )
            self._extra_config = extra_configs[-1]
            results["acceptance"] += np.sum(accept) / naux
        else:
            assert auxassignments is not None

        borb_configs = self.evaluate_orbitals(configs.electron(self._electrons))
        borb_configs = borb_configs.reshape(nconf, self.nelec, -1)
        # Orbital evaluations at extra coordinate.
        all_extra_configs = extra_configs[0].mask(
            np.zeros(naux * self._nsweeps, dtype=int)
        )
        all_extra_configs.join(extra_configs)
        borb_aux = self.evaluate_orbitals(all_extra_configs)
        borb_aux = borb_aux.reshape(self._nsweeps, naux, -1)
        bauxsquared = np.abs(borb_aux) ** 2
        fsum = np.sum(bauxsquared, axis=-1, keepdims=True)
        norm = bauxsquared / fsum
        baux_f = borb_aux / fsum

        for sweep, aux in enumerate(auxassignments):
            epos = extra_configs[sweep].configs[aux]
            newconfigs = configs.make_irreducible(0, epos)
            wfratio = wf.testvalue_many(self._electrons, newconfigs)

            ratio = np.einsum(
                "ie,ij,iek->ijk",
                wfratio.conj(),
                baux_f[sweep, aux],
                borb_configs.conj(),
                optimize=True,
            )

            results["value"] += ratio
            results["norm"] += norm[sweep, aux]

        results["value"] /= self._nstep
        results["norm"] = results["norm"] / self._nstep
        results["acceptance"] /= self._nstep

        return results

    def avg(self, configs, wf):
        return {k: np.mean(it, axis=0) for k, it in self(configs, wf).items()}

    def get_extra_configs(self, configs):
        """ Returns an nstep length array of configurations
            starting from self._extra_config """
        nconf = configs.configs.shape[0]
        naux = self._extra_config.configs.shape[0]
        auxassignments = np.random.randint(0, naux, size=(self._nsweeps, nconf))
        accept, extra_configs = self.sample_onebody(self._extra_config, self._nsweeps)
        self._extra_config = extra_configs[-1]
        return extra_configs, auxassignments

    def sample_onebody(self, configs, nsamples=1):
        r""" For a set of orbitals defined by orb_coeff, return samples from :math:`f(r) = \sum_i \phi_i(r)^2`. """
        n = configs.configs.shape[0]
        borb = self.evaluate_orbitals(configs)
        fsum = (np.abs(borb) ** 2).sum(axis=1)

        allaccept = np.zeros((nsamples, n))
        allconfigs = []
        for s in range(nsamples):
            shift = np.sqrt(self._tstep) * np.random.randn(*configs.configs.shape)
            newconfigs = configs.make_irreducible(None, configs.configs + shift)
            borbnew = self.evaluate_orbitals(newconfigs)
            fsumnew = (np.abs(borbnew) ** 2).sum(axis=1)
            accept = fsumnew / fsum > np.random.rand(n)
            configs.move_all(newconfigs, accept)
            borb[accept] = borbnew[accept]
            fsum[accept] = fsumnew[accept]
            allconfigs.append(configs.copy())
            allaccept[s] = accept

        # config_pack = configs.mask(np.tile(np.arange(n), 2)) # two copies of configs
        # config_pack.join([configs, newconfigs])

        # borb = self.evaluate_orbitals(config_pack)

        return allaccept, allconfigs

    def _evaluate_orbitals(self, configs):
        ao = self._mol.eval_gto("GTOval_sph", configs.configs.reshape(-1, 3))
        return ao.dot(self._orb_coeff)

    def _evaluate_orbitals_pbc(self, configs):
        # wrap supercell positions into primitive cell
        lvecs = self._mol.lattice_vectors()
        prim_coords, prim_wrap = pbc.enforce_pbc(lvecs, configs.configs)
        wrap = prim_wrap + np.dot(configs.wrap, self.supercell.S)
        wrap = wrap.reshape(-1, 3)
        prim_coords = prim_coords.reshape(-1, 3)
        kdotR = np.linalg.multi_dot((self._kpts, lvecs.T, wrap.T))
        wrap_phase = self.get_wrapphase(kdotR)
        # evaluate AOs for all electron positions
        ao = self._mol.eval_gto("PBCGTOval_sph", prim_coords, kpts=self._kpts)
        ao = [aok * wrap_phase[k][:, np.newaxis] for k, aok in enumerate(ao)]
        borb = [aok.dot(ock) for aok, ock in zip(ao, self._orb_coeff)]
        return np.concatenate(borb, axis=1)

    def keys(self):
        return set(["value", "norm", "acceptance"])

    def shapes(self):
        norb = self.norb
        return {"value": (norb, norb), "norm": (norb,), "acceptance": ()}


def sample_onebody(mol, orb_coeff, configs, tstep=2.0):
    r""" For a set of orbitals defined by orb_coeff, return samples from :math:`f(r) = \sum_i \phi_i(r)^2`. """
    n = configs.shape[0]
    config_pack = np.concatenate([configs, configs], axis=0)
    config_pack[n:] += np.sqrt(tstep) * np.random.randn(*configs.shape)

    ao = mol.eval_gto("GTOval_sph", config_pack)
    borb = ao.dot(orb_coeff)
    fsum = (np.abs(borb) ** 2).sum(axis=1)

    accept = fsum[n:] / fsum[0:n] > np.random.rand(n)
    configs[accept] = config_pack[n:][accept]
    return accept, configs


def normalize_obdm(obdm, norm):
    return obdm / (norm[np.newaxis, :] * norm[:, np.newaxis]) ** 0.5
