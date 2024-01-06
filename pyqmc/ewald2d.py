import numpy as np
import pyqmc
import pyqmc.energy
import pyqmc.gpu as gpu


class Ewald:

    def __init__(self, cell, gmax=200, nlatvec=1):
        self.latvec = cell.lattice_vectors()
        self.atom_coords = cell.atom_coords()
        self.nelec = np.array(cell.nelec)
        self.atom_charges = gpu.cp.asarray(cell.atom_charges())
        self.dist = pyqmc.distance.MinimalImageDistance(self.latvec)
        self.cell_area = gpu.cp.linalg.det(self.latvec[:2, :2])
        self.recvec = gpu.cp.linalg.inv(self.latvec).T
        self.set_alpha()
        self.set_lattice_displacements(nlatvec)
        self.gpoints = self.generate_positive_gpoints(gmax)
        # self.set_gpoints_gweight(gmax)

    def set_alpha(self):
        smallest_height = gpu.cp.amin(1 / gpu.cp.linalg.norm(self.recvec[:2, :2], axis=1))
        self.alpha = 5.0 / smallest_height

    def set_lattice_displacements(self, nlatvec):
        space = [gpu.cp.arange(-nlatvec, nlatvec + 1)] * 2
        XYZ = gpu.cp.meshgrid(*space, indexing='ij')
        xyz = gpu.cp.stack(XYZ, axis=-1).reshape((-1, 2))
        z_zeros = np.zeros((xyz.shape[0], 1))
        xyz = gpu.cp.concatenate([xyz, z_zeros], axis=1)
        self.lattice_displacements = gpu.cp.asarray(np.dot(xyz, self.latvec))

    def generate_positive_gpoints(self, gmax):
        gXpos = gpu.cp.mgrid[1 : gmax + 1, -gmax: gmax + 1, 0:1].reshape(3, -1)
        gX0Ypos = gpu.cp.mgrid[0:1, 1: gmax + 1, 0:1].reshape(3, -1)
        gpts = gpu.cp.concatenate([gXpos, gX0Ypos], axis=1)
        gpoints = gpu.cp.einsum("ji,jk->ik", gpts, gpu.cp.asarray(self.recvec) * 2 * gpu.cp.pi)
        return gpoints

    # def set_gpoints_gweight(self, gmax, tol=1e-10):
    #     candidate_gpoints = self.generate_positive_gpoints(gmax)
    #     gsquared = gpu.cp.einsum('jk,jk->j', candidate_gpoints, candidate_gpoints)
    #     gnorm = gsquared**0.5
    #     gweight = gpu.cp.pi * gpu.erfc(gnorm/(2*self.alpha)) * 2
    #     gweight /= self.cell_area * gnorm
    #     self.gpoints = candidate_gpoints
        # self.gweight = gweight
        # mask_bigweight = gweight > tol
        # self.gpoints = candidate_gpoints[mask_bigweight]
        # self.gweight = gweight[mask_bigweight]
        # print()

    def eval_real_cij(self, dist, lattice_displacements):
        rvec = dist + lattice_displacements
        r = gpu.cp.linalg.norm(rvec, axis=-1)
        cij = gpu.cp.sum(gpu.erfc(self.alpha * r) / r, axis=-1)
        return cij

    def ewald_ion_ion_real_cross(self):
        if len(self.atom_charges) == 1:
            ion_ion_real_cross = 0
        else:
            # input to dist_matrix has the shape (nconf, nparticles, ndim)
            ion_ion_distances, ion_ion_idxs = self.dist.dist_matrix(self.atom_coords[None])
            ion_ion_cij = self.eval_real_cij(ion_ion_distances[:, :, None, :], self.lattice_displacements[None, None, :, :]) # (nconf=1, npairs=choose(4, 2))
            ion_ion_charge_ij = gpu.cp.prod(self.atom_charges[ion_ion_idxs], axis=1) # (npairs=6,)
            ion_ion_real_cross = gpu.cp.einsum('j,ij->i', ion_ion_charge_ij, ion_ion_cij) # (nconf=1,)
        return ion_ion_real_cross

    def ewald_elec_ion_real_cross(self, configs):
        elec_ion_dist = configs.dist.dist_i(self.atom_coords, configs.configs)  # (nelec=4, nconf=1, natoms=4, ndim=3)
        elec_ion_cij = self.eval_real_cij(elec_ion_dist[:, :, :, None, :], self.lattice_displacements[None, None, None, :, :])  # (nelec=4, nconf=1, natoms=4)
        elec_ion_real = gpu.cp.einsum('k,ijk->j', -self.atom_charges, elec_ion_cij)  # (nconf,)
        return elec_ion_real

    def ewald_elec_elec_real_cross(self, configs):
        nconf, nelec, ndim = configs.configs.shape
        if nelec == 0:
            elec_elec_real = 0
        else:
            elec_elec_dist, elec_elec_idxs = configs.dist.dist_matrix(configs.configs) # (nconf=1, npairs=6, ndim=3)
            elec_elec_cij = self.eval_real_cij(elec_elec_dist[:, :, None, :], self.lattice_displacements[None, None, :, :])
            elec_elec_real = gpu.cp.sum(elec_elec_cij, axis=-1)
        return elec_elec_real

    def eval_weight(self, dist):
        z = dist[..., 2][None, ...]
        gsquared = gpu.cp.einsum('jk,jk->j', self.gpoints, self.gpoints)
        gnorm = gsquared**0.5
        gnorm = gnorm[:, None, None]
        bracket = (
            gpu.cp.exp(gnorm * z) * gpu.erfc(self.alpha * z + gnorm / (2 * self.alpha)) +
            gpu.cp.exp(-gnorm * z) * gpu.erfc(-self.alpha * z + gnorm / (2 * self.alpha))
            )
        gweight = gpu.cp.pi / (self.cell_area * gnorm) * bracket
        return gweight

    # def ewald_ion_ion_recip_self(self):
    #     '''
    #     This is not for real use, but for testing the special case when i == j
    #     '''
    #     ii_sum2 = np.sum(self.atom_charges**2)
    #     ion_ion_recip_z_zero = ii_sum2 * np.sum(self.gweight)
    #     return ion_ion_recip_z_zero

    def ewald_ion_ion_recip(self):
        ii_dist = self.dist.dist_i(self.atom_coords, self.atom_coords) # (natoms, natoms, 3)
        g_dot_r = gpu.cp.einsum('kd,ijd->kij', self.gpoints, ii_dist) # (nk, natoms, natoms)
        gweight = self.eval_weight(ii_dist) # (nk, natoms, natoms)
        ion_ion_recip = gpu.cp.einsum('i,j,kij,kij->', self.atom_charges, self.atom_charges, gpu.cp.exp(1j * g_dot_r), gweight).real
        return ion_ion_recip # tested

    def ewald_elec_ion_recip(self, configs):
        ei_dist = self.dist.dist_i(self.atom_coords, configs.configs) # (nconf, natoms, nelec, 3)
        g_dot_r = gpu.cp.einsum('kd,cijd->ckij', self.gpoints, ei_dist) # (nconf, nk, natoms, nelec)
        gweight = self.eval_weight(ei_dist) # (nconf, nk, natoms, nelec)
        elec_ion_recip = -2 * gpu.cp.einsum('i,ckij,ckij->c', self.atom_charges, gpu.cp.exp(1j * g_dot_r), gweight).real
        return elec_ion_recip # tested

    def energy(self, configs):
        nelec = configs.configs.shape[1]
        # self.ewald_ion_ion_real_cross()
        # self.ewald_elec_ion_real_cross(configs)
        # self.ewald_elec_elec_real_cross(configs)
        # self.ewald_ion_ion_recip()
        self.ewald_elec_ion_recip(configs)
