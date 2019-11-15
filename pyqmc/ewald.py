import numpy as np
import pyqmc
from scipy.special import erfc
from pyqmc.slaterpbc import get_supercell_copies


def get_supercell_atoms(mol, supercell):
    """
    Calculate atom coordinates and charges for a supercell of the mol object
    """
    atom_coords = []
    atom_charges = []
    Rpts = get_supercell_copies(mol.lattice_vectors(), supercell)
    for (xyz, charge) in zip(mol.atom_coords(), mol.atom_charges()):
        atom_coords.extend([xyz + R for R in Rpts])
        atom_charges.extend([charge for R in Rpts])
    return np.asarray(atom_coords), np.asarray(atom_charges)


class Ewald:
    def __init__(self, mol, supercell, ewald_gmax=200, nlatvec=2):
        """
        Class for computing Ewald sums
        Inputs:
            mol: pyscf Cell object
            supercell: (3, 3) array to scale mol.lattice_vectors() up to the QMC calculation cell (i.e. qmc_cell = np.dot(supercell, mol.lattice_vectors()))
            ewald_gmax: int, how far to take reciprocal sum; probably never needs to be changed.
            nlatvec: int, how far to take real space sum; probably never needs to be changed.
        """
        if supercell is None:
            supercell = np.eye(3)
        self.nelec = np.array(mol.nelec) * np.linalg.det(supercell)
        self.atom_coords, self.atom_charges = get_supercell_atoms(mol, supercell)
        self.latvec = np.dot(supercell, mol.lattice_vectors())
        self.set_lattice_displacements(nlatvec)
        self.set_up_reciprocal_ewald_sum(ewald_gmax)

    def set_lattice_displacements(self, nlatvec):
        """
        Generates list of lattice-vector displacements to add together for real space sum
        """
        XYZ = np.meshgrid(*[np.arange(-nlatvec, nlatvec + 1)] * 3, indexing="ij")
        xyz = np.stack(XYZ, axis=-1).reshape((-1, 3))
        self.lattice_displacements = np.dot(xyz, self.latvec)

    def set_up_reciprocal_ewald_sum(self, ewald_gmax):
        """
        Determine parameters for Ewald sums
        Inputs:
            latvec: (3, 3) array of lattice vectors; latvec[0] is the first
            ewald_gmax: int, max number of reciprocal lattice vectors to check away from 0
        """
        cellvolume = np.linalg.det(self.latvec)
        recvec = np.linalg.inv(self.latvec)
        crossproduct = recvec * cellvolume

        # Determine alpha
        tmpheight_i = np.einsum("ij,ij->i", crossproduct, self.latvec)
        length_i = np.linalg.norm(crossproduct, axis=1)
        smallestheight = np.amin(np.abs(tmpheight_i) / length_i)
        self.alpha = 5.0 / smallestheight

        # Determine G points to include in reciprocal Ewald sum
        XYZ = np.meshgrid(*[np.arange(-ewald_gmax, ewald_gmax + 1)] * 3, indexing="ij")
        X, Y, Z = [x.ravel() for x in XYZ]
        positive_octants = X + 1e-6 * Y + 1e-12 * Z > 0  # assume ewald_gmax < 1e5
        gpoints = np.stack((X, Y, Z), axis=-1)[positive_octants]
        gpoints = np.dot(gpoints, recvec) * 2 * np.pi
        gsquared = np.sum(gpoints ** 2, axis=1)
        gweight = 4 * np.pi * np.exp(-gsquared / (4 * self.alpha ** 2))
        gweight /= cellvolume * gsquared
        bigweight = gweight > 1e-10
        self.gpoints = gpoints[bigweight]
        self.gweight = gweight[bigweight]

        # def const_ewald():
        # Compute Ewald constants (sums)
        # ntot = np.sum(self.nelec)
        # ee_sum2 = ntot
        ii_sum2 = np.sum(self.atom_charges ** 2)

        # ee_sum = ntot * (ntot - 1) / 2
        i_sum = np.sum(self.atom_charges)
        ii_sum = (np.sum(self.atom_charges) ** 2 - ii_sum2) / 2

        ijconst = -np.pi / (cellvolume * self.alpha ** 2)
        self.ijconst = ijconst
        squareconst = -self.alpha / np.sqrt(np.pi) + ijconst / 2

        self.ii_const = ii_sum * ijconst + ii_sum2 * squareconst
        self.ee_const = lambda ne: ne * (ne - 1) / 2 * ijconst + ne * squareconst
        self.ei_const = lambda ne: -ne * i_sum * ijconst

        self.e_single = lambda ne: (ne - 1) * ijconst - i_sum * ijconst + squareconst
        self.e_single_test = -i_sum * ijconst + squareconst
        self.ion_ion = self.ewald_ion()

        # XC correction not used, so we can compare to other codes
        rs = lambda ne: (3 / (4 * np.pi) / (ne * cellvolume)) ** (1 / 3)
        cexc = 0.36
        xc_correction = lambda ne: cexc / rs(ne)

    def ewald_ion(self):
        """
        Compute ion contribution to Ewald sums
        Returns:
            ion_ion: float, ion-ion component of Ewald sum
        """
        # Real space part
        if len(self.atom_charges) == 1:
            ion_ion_real = 0
        else:
            dist = pyqmc.distance.MinimalImageDistance(self.latvec)
            ion_distances, ion_inds = dist.dist_matrix(self.atom_coords[np.newaxis])
            rvec = ion_distances[:, :, np.newaxis, :] + self.lattice_displacements
            r = np.linalg.norm(rvec, axis=-1)
            charge_ij = np.prod(self.atom_charges[np.asarray(ion_inds)], axis=1)
            ion_ion_real = np.einsum("j,ijk->", charge_ij, erfc(self.alpha * r) / r)

        # Reciprocal space part
        GdotR = np.dot(self.gpoints, self.atom_coords.T)
        self.ion_exp = np.dot(np.exp(1j * GdotR), self.atom_charges)
        ion_ion_rec = np.dot(self.gweight, np.abs(self.ion_exp) ** 2)

        ion_ion = ion_ion_real + ion_ion_rec
        return ion_ion

    def ewald_electron(self, configs):
        """
        Compute the Ewald sum for e-e and e-ion
        Inputs:
            configs: pyqmc PeriodicConfigs object of shape (nconf, nelec, ndim)
        Returns:
            ee: electron-electron part
            ei: electron-ion part
        """
        nconf, nelec, ndim = configs.configs.shape

        # Real space electron-ion part
        # ei_distances shape (elec, conf, atom, dim)
        ei_distances = configs.dist.dist_i(self.atom_coords, configs.configs)
        rvec = ei_distances[:, :, :, np.newaxis, :] + self.lattice_displacements
        r = np.linalg.norm(rvec, axis=-1)
        ei_real_separated = np.einsum(
            "k,ijkl->ji", -self.atom_charges, erfc(self.alpha * r) / r
        )

        # Real space electron-electron part
        if nelec > 1:
            ee_distances, ee_inds = configs.dist.dist_matrix(configs.configs)
            rvec = ee_distances[:, :, np.newaxis, :] + self.lattice_displacements
            r = np.linalg.norm(rvec, axis=-1)
            ee_cij = np.sum(erfc(self.alpha * r) / r, axis=-1)

            ee_matrix = np.zeros((nconf, nelec, nelec))
            # ee_matrix[:, ee_inds] = ee_cij
            for ((i, j), val) in zip(ee_inds, ee_cij.T):
                ee_matrix[:, i, j] = val
                ee_matrix[:, j, i] = val
            ee_real_separated = ee_matrix.sum(axis=-1) / 2
        else:
            ee_real_separated = np.zeros(nelec)

        # Reciprocal space electron-electron part
        e_GdotR = np.dot(configs.configs, self.gpoints.T)
        e_expGdotR = np.exp(1j * e_GdotR)
        sum_e_exp = np.sum(e_expGdotR, axis=1, keepdims=True)
        coscos_sinsin = np.real(sum_e_exp.conj() * e_expGdotR)
        ee_recip_separated = np.dot(coscos_sinsin - 0.5, self.gweight)

        # Reciprocal space electron-ion part
        coscos_sinsin = np.real(-self.ion_exp.conj() * e_expGdotR)
        ei_recip_separated = np.dot(coscos_sinsin, self.gweight)

        # Combine parts
        self.ei_separated = ei_real_separated + 2 * ei_recip_separated
        self.ee_separated = ee_real_separated + 1 * ee_recip_separated
        self.ewalde_separated = self.ei_separated + self.ee_separated
        nelec = ee_recip_separated.shape[1]
        ee = self.ee_separated.sum(axis=1) + nelec / 2 * self.gweight.sum()
        ei = self.ei_separated.sum(axis=1)
        return ee, ei

    def energy(self, configs):
        """
        Compute Coulomb energy for a set of configs
        Inputs:
            configs: pyqmc PeriodicConfigs object of shape (nconf, nelec, ndim)
        Returns: 
            ee: electron-electron part
            ei: electron-ion part
            ii: ion-ion part
        """
        nelec = configs.configs.shape[1]
        ee, ei = self.ewald_electron(configs)
        ee += self.ee_const(nelec)
        ei += self.ei_const(nelec)
        ii = self.ion_ion + self.ii_const
        return ee, ei, ii

    def energy_separated(self, configs):
        """
        Compute Coulomb energy separated by electron in a set of configs. 
        NOTE: energy() needs to be called first to update the separated energy values
        Inputs:
            configs: pyqmc PeriodicConfigs object of shape (nconf, nelec, ndim)
        Returns: 
            (nelec,) energies
        """
        nelec = configs.configs.shape[1]
        return self.e_single(nelec) + self.ewalde_separated

    def energy_with_test_pos(self, configs, epos):
        """
        Compute Coulomb energy of an additional test electron with a set of configs
        Inputs:
            configs: pyqmc PeriodicConfigs object of shape (nconf, nelec, ndim)
            epos: pyqmc PeriodicConfigs object of shape (nconf, ndim)
        Returns: 
            Vtest: (nconf, nelec+1) array. The first nelec columns are Coulomb energies between the test electron and each electron; the last column is the contribution from all the ions.
        """
        nconf, nelec, ndim = configs.configs.shape
        Vtest = np.zeros((nconf, nelec + 1)) + self.ijconst
        Vtest[:, -1] = self.e_single_test

        # Real space electron-ion part
        # ei_distances shape (conf, atom, dim)
        ei_distances = configs.dist.dist_i(self.atom_coords, epos.configs)
        rvec = ei_distances[:, :, np.newaxis, :] + self.lattice_displacements
        r = np.linalg.norm(rvec, axis=-1)
        Vtest[:, -1] += np.einsum(
            "k,jkl->j", -self.atom_charges, erfc(self.alpha * r) / r
        )

        # Real space electron-electron part
        ee_distances = configs.dist.dist_i(configs.configs, epos.configs)
        rvec = ee_distances[:, :, np.newaxis, :] + self.lattice_displacements
        r = np.linalg.norm(rvec, axis=-1)
        Vtest[:, :-1] += np.sum(erfc(self.alpha * r) / r, axis=-1)

        # Reciprocal space electron-electron part
        e_expGdotR = np.exp(1j * np.dot(configs.configs, self.gpoints.T))
        test_exp = np.exp(1j * np.dot(epos.configs, self.gpoints.T))
        ee_recip_separated = np.dot(np.real(test_exp.conj() * e_expGdotR), self.gweight)
        Vtest[:, :-1] += 2 * ee_recip_separated

        # Reciprocal space electrin-ion part
        coscos_sinsin = np.real(-self.ion_exp.conj() * test_exp)
        ei_recip_separated = np.dot(coscos_sinsin + 0.5, self.gweight)
        Vtest[:, -1] += 2 * ei_recip_separated

        return Vtest
