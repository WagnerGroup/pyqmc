import numpy as np
import pyqmc
from pyqmc.coord import PeriodicConfigs
import pyqmc.energy
import pyqmc.gpu as gpu
from pyscf.pbc.gto.cell import Cell
from typing import Tuple

class Ewald:
    '''
    Evaluate the Ewald summation using the 2D formula
    [Yeh and Berkowitz. J. Chem. Phys. 111, 3155–3162 (1999)]
    https://doi.org/10.1063/1.479595
    '''

    def __init__(self, cell: Cell, gmax: int = 200, nlatvec: int = 1, alpha_scaling: float = 5.0):
        '''
        :parameter pyscf.pbc.gto.cell.Cell cell: PySCF Cell object
        :parameter int gmax: max number of reciprocal lattice vectors to check away from 0
        :parameter int nlatvec: sum goes from `-nlatvec` to `nlatvec` in each lattice direction.
        :parameter float alpha_scaling: scaling factor for partitioning the real-space and reciprocal-space parts.
        '''
        self.latvec = cell.lattice_vectors()
        self.atom_coords = cell.atom_coords()[np.newaxis]
        self.nelec = gpu.cp.array(cell.nelec)
        self.atom_charges = gpu.cp.asarray(cell.atom_charges())
        self.dist = pyqmc.distance.MinimalImageDistance(self.latvec)
        self.cell_area = gpu.cp.linalg.det(self.latvec[:2, :2])
        self.recvec = gpu.cp.linalg.inv(self.latvec).T
        self.alpha_scaling = alpha_scaling
        self.set_alpha()
        self.set_lattice_displacements(nlatvec)
        self.set_gpoints(gmax)

    def set_alpha(self):
        '''
        Define the partitioning of the real and reciprocal-space parts.
        '''
        smallest_height = gpu.cp.amin(1 / gpu.cp.linalg.norm(self.recvec[:2, :2], axis=1))
        self.alpha = self.alpha_scaling / smallest_height

    def set_lattice_displacements(self, nlatvec: int):
        '''
        Define a list of lattice-vector displacements to add together for real-space sum.

        :parameter int nlatvec: sum goes from `-nlatvec` to `nlatvec` in each lattice direction.
        '''
        space = [gpu.cp.arange(-nlatvec, nlatvec + 1)] * 2
        XYZ = gpu.cp.meshgrid(*space, indexing='ij')
        xyz = gpu.cp.stack(XYZ, axis=-1).reshape((-1, 2))
        z_zeros = gpu.cp.zeros((xyz.shape[0], 1))
        xyz = gpu.cp.concatenate([xyz, z_zeros], axis=1)
        self.lattice_displacements = gpu.cp.asarray(gpu.cp.dot(xyz, self.latvec))

    def generate_positive_gpoints(self, gmax: int) -> gpu.cp.ndarray:
        '''
        Generate a list of points in the reciprocal space to add together for reciprocal-space sum.

        :parameter gmax: max number of reciprocal lattice vectors to check away from 0
        :return: reciprocal-space points (nk, 3)
        '''
        gXpos = gpu.cp.mgrid[1 : gmax + 1, -gmax: gmax + 1, 0:1].reshape(3, -1)
        gX0Ypos = gpu.cp.mgrid[0:1, 1: gmax + 1, 0:1].reshape(3, -1)
        gpts = gpu.cp.concatenate([gXpos, gX0Ypos], axis=1)
        gpoints = gpu.cp.einsum("ji,jk->ik", gpts, gpu.cp.asarray(self.recvec) * 2 * gpu.cp.pi)
        return gpoints

    def set_gpoints(self, gmax: int, tol: float = 1e-10):
        '''
        Define reciprocal-lattice points with large contributions according to `tol`.

        :parameter gmax: max number of reciprocal lattice vectors to check away from 0
        :parameter tol: tolerance for the cutoff weight
        '''
        candidate_gpoints = self.generate_positive_gpoints(gmax)
        gsquared = gpu.cp.einsum('jk,jk->j', candidate_gpoints, candidate_gpoints)
        gnorm = gsquared**0.5
        gweight = gpu.cp.pi * gpu.erfc(gnorm/(2*self.alpha)) * 2
        gweight /= self.cell_area * gnorm
        mask_bigweight = gweight > tol
        self.gpoints = candidate_gpoints[mask_bigweight]

    def ewald_real_weight(self, dist: gpu.cp.ndarray, lattice_displacements: gpu.cp.ndarray) -> gpu.cp.ndarray:
        r'''
        Compute the weight for real-space sum

        .. math:: W_{\textrm{real}}(\mathbf{r}_{mn}) = \sum_{\mathbf{n}} {}^{\prime} \frac{\textrm{erfc}(\alpha |\mathbf{r}_{mn} + \mathbf{n} L|)}{|\mathbf{r}_{mn} + \mathbf{n} L|}.

        `m` and `n` denote either electrons or ions.

        :parameter dist: distance matrix. Shape: (num_particles_m, num_particles_n, 1, 3)
        :parameter lattice_displacements: a list of lattice-vector displacements. Shape: (1, 1, num_lattice_vectors, 3)
        :return: weight for real-space sum. Shape: (num_particles_m, num_particles_n)
        '''
        rvec = dist + lattice_displacements
        r = gpu.cp.linalg.norm(rvec, axis=-1)
        weight = gpu.cp.sum(gpu.erfc(self.alpha * r) / r, axis=-1) # sum over the lattice vectors
        return weight

    def ewald_real_ion_ion_cross(self) -> float:
        r'''
        Compute ion-ion contributions to the cross terms of real space sum.

        .. math:: E_{\textrm{real,cross}}^{\textrm{ion-ion}} = \sum_{I=1}^{N_{\textrm{ion}}} \sum_{J > I}^{N_{\textrm{ion}}} q_I q_J W_{\textrm{real}}(\mathbf{r}_{IJ}).

        :returns: ion-ion real-space cross-term component of Ewald sum
        '''
        if len(self.atom_charges) == 1:
            ion_ion_real_cross = 0
        else:
            # input to dist_matrix has the shape (nconf, natoms, ndim)
            ion_ion_dist, ion_ion_idxs = self.dist.dist_matrix(self.atom_coords)
            ion_ion_cij = self.ewald_real_weight(ion_ion_dist[:, :, None, :], self.lattice_displacements[None, None, :, :]) # (nconf, npairs=choose(natoms, 2))
            ion_ion_charge_ij = gpu.cp.prod(self.atom_charges[gpu.cp.asarray(ion_ion_idxs)], axis=1) # (npairs,)
            ion_ion_real_cross = gpu.cp.einsum('j,ij->i', ion_ion_charge_ij, ion_ion_cij) # (nconf,)
        return ion_ion_real_cross

    def ewald_real_elec_ion_cross(self, configs: PeriodicConfigs) -> float:
        r'''
        Compute electron-ion contributions to the cross terms of real space sum.

        .. math:: E_{\textrm{real,cross}}^{\textrm{e-ion}} = \sum_{i=1}^{N_{\textrm{e}}} \sum_{I=1}^{N_{\textrm{ion}}} (-1) q_I W_{\textrm{real}}(\mathbf{r}_{iI}).

        :parameter configs: Shape: (nconf, nelec, 3)
        :returns: electron-ion real-space cross-term component of Ewald sum
        '''
        elec_ion_dist = configs.dist.pairwise(self.atom_coords, configs.configs) # (nelec, nconf, natoms, ndim)
        elec_ion_cij = self.ewald_real_weight(elec_ion_dist[:, :, :, None, :], self.lattice_displacements[None, None, None, :, :])  # (nconf, nelec, natoms)
        elec_ion_real = gpu.cp.einsum('k,ijk->i', -self.atom_charges, elec_ion_cij)  # (nconf,)
        return elec_ion_real

    def ewald_real_elec_elec_cross(self, configs: PeriodicConfigs) -> float:
        r'''
        Compute electron-electron contributions to the cross terms of real space sum.

        .. math:: E_{\textrm{real,cross}}^{\textrm{e-e}} = \sum_{i=1}^{N_{\textrm{e}}} \sum_{j > i}^{N_{\textrm{e}}} W_{\textrm{real}}(\mathbf{r}_{ij}).

        :parameter configs: Shape: (nconf, nelec, 3)
        :returns: electron-electron real-space cross-term component of Ewald sum
        '''
        nconf, nelec, ndim = configs.configs.shape
        if nelec == 0:
            elec_elec_real = 0
        else:
            elec_elec_dist, elec_elec_idxs = configs.dist.dist_matrix(configs.configs) # (nconf, npairs, ndim)
            elec_elec_cij = self.ewald_real_weight(elec_elec_dist[:, :, None, :], self.lattice_displacements[None, None, :, :])
            elec_elec_real = gpu.cp.sum(elec_elec_cij, axis=-1)
        return elec_elec_real

    def ewald_recip_weight(self, dist: gpu.cp.ndarray) -> gpu.cp.ndarray:
        r'''
        Compute the weight for the reciprocal-space sum

        .. math:: W_{\textrm{recip},k>0}(k, z_{mn}) = \frac{\pi}{A k}
                \left[
                    \mathrm{e}^{k z_{mn}} \textrm{erfc} \left(\alpha z_{mn} + \frac{k}{2 \alpha} \right)
                    + \mathrm{e}^{-k z_{mn}} \textrm{erfc} \left(-\alpha z_{mn} + \frac{k}{2 \alpha} \right)
                \right].

        :parameter dist: distance matrix.
            Shape: (num_particles_m, num_particles_n, 3) or (nconf, num_particles_m, num_particles_n, 3)
        :return: weight for reciprocal-space sum when k > 0. Shape: (nk, num_particles_m, num_particles_n) or (nconf, nk, num_particles_m, num_particles_n)
        '''
        z = dist[..., 2][None, ...]
        gsquared = gpu.cp.einsum('jk,jk->j', self.gpoints, self.gpoints)
        gnorm = gsquared**0.5
        gnorm = gnorm[:, None, None]
        w1 = gpu.cp.exp(gnorm * z) * gpu.erfc(self.alpha * z + gnorm / (2 * self.alpha))
        w2 = gpu.cp.exp(-gnorm * z) * gpu.erfc(-self.alpha * z + gnorm / (2 * self.alpha))
        gweight = gpu.cp.pi / (self.cell_area * gnorm) * (w1 + w2)
        return gweight

    def ewald_recip_ion_ion(self) -> float:
        r'''
        Compute ion-ion contributions to the reciprocal-space sum.

        .. math:: E_{\textrm{recip},k > 0}^{\textrm{ion-ion}}
            = \sum_{\mathbf{k} > 0} \sum_{I=1}^{N_{\textrm{ion}}} \sum_{J=1}^{N_{\textrm{ion}}} q_I q_J \mathrm{e}^{i \mathbf{k} \cdot \mathbf{r}_{IJ}} W_{\textrm{recip},k>0}(k, z_{IJ}).

        :return: ion-ion reciprocal-space k>0 component of Ewald sum
        '''
        ii_dist = self.dist.pairwise(self.atom_coords, self.atom_coords)[0] # (natoms, natoms, 3)
        g_dot_r = gpu.cp.einsum('kd,ijd->kij', self.gpoints, ii_dist) # (nk, natoms, natoms)
        gweight = self.ewald_recip_weight(ii_dist) # (nk, natoms, natoms)
        ion_ion_recip = gpu.cp.einsum('i,j,kij,kij->', self.atom_charges, self.atom_charges, gpu.cp.exp(1j * g_dot_r), gweight).real
        return ion_ion_recip

    def ewald_recip_elec_ion(self, configs: PeriodicConfigs) -> float:
        r'''
        Compute electron-ion contributions to the reciprocal-space sum.

        .. math:: E_{\textrm{recip},k > 0}^{\textrm{e-ion}}
            = \sum_{\mathbf{k} > 0} \sum_{i=1}^{N_{\textrm{e}}} \sum_{I=1}^{N_{\textrm{ion}}} (-2 q_I) \mathrm{e}^{i \mathbf{k} \cdot \mathbf{r}_{iI}} W_{\textrm{recip},k>0}(k, z_{iI}).

        :parameter configs: Shape: (nconf, nelec, 3)
        :return: electron-ion reciprocal-space k>0 component of Ewald sum
        '''
        ei_dist = self.dist.pairwise(self.atom_coords, configs.configs) # (nconf, natoms, nelec, 3)
        g_dot_r = gpu.cp.einsum('kd,cijd->ckij', self.gpoints, ei_dist) # (nconf, nk, natoms, nelec)
        gweight = self.ewald_recip_weight(ei_dist) # (nconf, nk, natoms, nelec)
        elec_ion_recip = -2 * gpu.cp.einsum('i,ckij,ckij->c', self.atom_charges, gpu.cp.cos(g_dot_r), gweight)
        return elec_ion_recip

    def ewald_recip_elec_elec(self, configs: PeriodicConfigs) -> float:
        r'''
        Compute electron-electron contributions to the reciprocal-space sum.

        .. math:: E_{\textrm{recip},k > 0}^{\textrm{e-e}}
            = \sum_{\mathbf{k} > 0} \sum_{i=1}^{N_{\textrm{e}}} \sum_{j=1}^{N_{\textrm{e}}} \mathrm{e}^{i \mathbf{k} \cdot \mathbf{r}_{ij}} W_{\textrm{recip},k>0}(k, z_{ij}).

        :parameter configs: Shape: (nconf, nelec, 3)
        :return: electron-electron reciprocal-space k>0 component of Ewald sum
        '''
        ee_dist = self.dist.pairwise(configs.configs, configs.configs) # (nconf, nelec, nelec, 3)
        g_dot_r = gpu.cp.einsum('kd,cijd->ckij', self.gpoints, ee_dist) # (nconf, nk, nelec, nelec)
        gweight = self.ewald_recip_weight(ee_dist) # (nconf, nk, nelec, nelec)
        elec_elec_recip = gpu.cp.einsum('ckij,ckij->c', gpu.cp.exp(1j * g_dot_r), gweight).real
        return elec_elec_recip

    def ewald_real_ion_ion_self(self) -> float:
        r'''
        Compute ion-ion contributions to the self terms of real-space sum

        .. math:: E_{\textrm{real,self}}^{\textrm{ion-ion}} = - \frac{\alpha}{\sqrt{\pi}} \sum_{I=1}^{N_{\textrm{ion}}} q_I^2. \\

        :return: ion-ion real-space self energy
        '''
        ion_ion_real_self = -self.alpha / gpu.cp.sqrt(gpu.cp.pi) * gpu.cp.sum(self.atom_charges**2)
        return ion_ion_real_self

    def ewald_real_elec_elec_self(self, nelec: int) -> float:
        r'''
        Compute electron-electron contributions to the self terms of real-space sum

        .. math:: E_{\textrm{real,self}}^{\textrm{e-e}} = - \frac{\alpha}{\sqrt{\pi}} N_{\textrm{e}}. \\

        :parameter int nelec: number of electrons
        :return: electron-electron real-space self energy
        '''
        elec_elec_real_self = -self.alpha / gpu.cp.sqrt(gpu.cp.pi) * nelec
        return elec_elec_real_self

    def ewald_recip_weight_charge(self, dist: gpu.cp.ndarray) -> gpu.cp.ndarray:
        r'''
        Compute the weight for the charge terms (k = 0 terms) in reciprocal space sum

        .. math:: W_{\textrm{recip,k=0}}(z_{mn}) = - \frac{\pi}{A} \left[z_{mn} \textrm{erf}(\alpha z_{mn}) + \frac{1}{\alpha \sqrt{\pi}} \exp(-\alpha^2 z_{mn}^2) \right].

        :parameter dist: distance matrix.
            Shape: (num_particles_m, num_particles_n, 3) or (nconf, num_particles_m, num_particles_n, 3)
        :return: weight for reciprocal-space sum when k = 0. Shape: (num_particles_m, num_particles_n) or (nconf, num_particles_m, num_particles_n)
        '''
        z = dist[..., 2] # (natoms, natoms)
        w1 = z * gpu.erf(self.alpha * z)
        w2 = 1/(self.alpha * gpu.cp.sqrt(gpu.cp.pi)) * gpu.cp.exp(-self.alpha**2 * z**2)
        w = -gpu.cp.pi / self.cell_area * (w1 + w2)
        return w

    def ewald_recip_ion_ion_charge(self) -> float:
        r'''
        Compute ion-ion contributions to sum of the charge terms.

        .. math:: E_{\textrm{recip},k = 0}^{\textrm{ion-ion}}
            = \sum_{I=1}^{N_{\textrm{ion}}} \sum_{J=1}^{N_{\textrm{ion}}} q_I q_J W_{\textrm{recip},k=0}(z_{IJ}).

        :return: ion-ion charge term
        '''
        ii_dist = self.dist.pairwise(self.atom_coords, self.atom_coords)[0] # (natoms, natoms, 3)
        weight = self.ewald_recip_weight_charge(ii_dist)
        ion_ion_charge = gpu.cp.einsum('i,j,ij->', self.atom_charges, self.atom_charges, weight)
        return ion_ion_charge

    def ewald_recip_elec_ion_charge(self, configs: PeriodicConfigs) -> float:
        r'''
        Compute electron-ion contributions to sum of the charge terms.

        .. math:: E_{\textrm{recip},k = 0}^{\textrm{e-ion}}
            = \sum_{i=1}^{N_{\textrm{e}}} \sum_{I=1}^{N_{\textrm{ion}}} (-2 q_I) W_{\textrm{recip},k=0}(z_{iI}).

        :parameter configs: Shape: (nconf, nelec, 3)
        :return: electron-ion charge term
        '''
        ei_dist = self.dist.pairwise(self.atom_coords, configs.configs) # (nconf, natoms, nelec, 3)
        weight = self.ewald_recip_weight_charge(ei_dist)
        elec_ion_charge = -2 * gpu.cp.einsum('i,cij->c', self.atom_charges, weight)
        return elec_ion_charge

    def ewald_recip_elec_elec_charge(self, configs: PeriodicConfigs) -> float:
        r'''
        Compute electron-electron contributions to sum of the charge terms.

        .. math:: E_{\textrm{recip},k = 0}^{\textrm{e-e}}
            = \sum_{i=1}^{N_{\textrm{e}}} \sum_{j=1}^{N_{\textrm{e}}} W_{\textrm{recip},k=0}(z_{ij}).

        :parameter configs: Shape: (nconf, nelec, 3)
        :return: electron-electron charge term
        '''
        ee_dist = self.dist.pairwise(configs.configs, configs.configs) # (nconf, nelec, nelec, 3)
        weight = self.ewald_recip_weight_charge(ee_dist)
        elec_elec_charge = gpu.cp.einsum('cij->c', weight)
        return elec_elec_charge

    def energy(self, configs: PeriodicConfigs) -> Tuple[float, float, float]:
        r'''
        Compute Coulomb energy for a set of configs.

        .. math:: E &= E^{\textrm{e-e}} + E^{\textrm{e-ion}} + E^{\textrm{ion-ion}} \\
            E^{\textrm{e-e}} &= E_{\textrm{real,cross}}^{\textrm{e-e}} + E_{\textrm{real,self}}^{\textrm{e-e}}
                + E_{\textrm{recip},k>0}^{\textrm{e-e}} + E_{\textrm{recip},k=0}^{\textrm{e-e}} \\
            E^{\textrm{e-ion}} &= E_{\textrm{real,cross}}^{\textrm{e-ion}}
                + E_{\textrm{recip},k>0}^{\textrm{e-ion}} + E_{\textrm{recip},k=0}^{\textrm{e-ion}} \\
            E^{\textrm{ion-ion}} &= E_{\textrm{real,cross}}^{\textrm{ion-ion}} + E_{\textrm{real,self}}^{\textrm{ion-ion}}
                + E_{\textrm{recip},k>0}^{\textrm{ion-ion}} + E_{\textrm{recip},k=0}^{\textrm{ion-ion}}

        :parameter configs: Shape: (nconf, nelec, 3)
        :return:
            * ee: electron-electron part
            * ei: electron-ion part
            * ii: ion-ion part
        '''
        nelec = configs.configs.shape[1]
        ii_const = self.ewald_recip_ion_ion_charge() + self.ewald_real_ion_ion_self()
        ii_real_cross = self.ewald_real_ion_ion_cross()
        ii_recip = self.ewald_recip_ion_ion()
        ii = ii_real_cross + ii_recip + ii_const

        ee_const = self.ewald_recip_elec_elec_charge(configs) + self.ewald_real_elec_elec_self(nelec)
        ee_real_cross = self.ewald_real_elec_elec_cross(configs)
        ee_recip = self.ewald_recip_elec_elec(configs)
        ee = ee_real_cross + ee_recip + ee_const

        ei_const = self.ewald_recip_elec_ion_charge(configs)
        ei_real_cross = self.ewald_real_elec_ion_cross(configs)
        ei_recip = self.ewald_recip_elec_ion(configs)
        ei = ei_real_cross + ei_recip + ei_const
        return ee, ei, ii
