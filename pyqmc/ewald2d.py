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
        self.set_constants()
        self.set_lattice_displacements(nlatvec)
        self.set_gpoints_gweight(gmax)

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

    def set_gpoints_gweight(self, gmax, tol=1e-10):
        candidate_gpoints = self.generate_positive_gpoints(gmax)
        gsquared = gpu.cp.einsum('jk,jk->j', candidate_gpoints, candidate_gpoints)
        gnorm = gsquared**0.5
        gweight = gpu.cp.pi * gpu.erfc(gnorm/(2*self.alpha)) * 2
        gweight /= self.cell_area * gnorm
        mask_bigweight = gweight > tol
        self.gpoints = candidate_gpoints[mask_bigweight]
        self.gweight = gweight[mask_bigweight]

    def eval_real_cij(self, dist, lattice_displacements):
        rvec = dist + lattice_displacements
        r = gpu.cp.linalg.norm(rvec, axis=-1)
        cij = gpu.cp.sum(gpu.erfc(self.alpha * r) / r, axis=-1)
        return cij

    def ewald_ion_ion_real_cross(self):
        if len(self.atom_charges) == 1:
            ion_ion_real_cross = 0
        else:
            # input to dist_matrix has the shape (nconf, natoms, ndim)
            ion_ion_distances, ion_ion_idxs = self.dist.dist_matrix(self.atom_coords[None])
            ion_ion_cij = self.eval_real_cij(ion_ion_distances[:, :, None, :], self.lattice_displacements[None, None, :, :]) # (nconf, npairs=choose(natoms, 2))
            ion_ion_charge_ij = gpu.cp.prod(self.atom_charges[ion_ion_idxs], axis=1) # (npairs,)
            ion_ion_real_cross = gpu.cp.einsum('j,ij->i', ion_ion_charge_ij, ion_ion_cij) # (nconf,)
        return ion_ion_real_cross

    def ewald_elec_ion_real_cross(self, configs):
        elec_ion_dist = configs.dist.dist_i(self.atom_coords, configs.configs)  # (nelec, nconf, natoms, ndim)
        elec_ion_cij = self.eval_real_cij(elec_ion_dist[:, :, :, None, :], self.lattice_displacements[None, None, None, :, :])  # (nelec, nconf, natoms)
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
        w1 = gpu.cp.exp(gnorm * z) * gpu.erfc(self.alpha * z + gnorm / (2 * self.alpha))
        w2 = gpu.cp.exp(-gnorm * z) * gpu.erfc(-self.alpha * z + gnorm / (2 * self.alpha))
        gweight = gpu.cp.pi / (self.cell_area * gnorm) * (w1 + w2)
        return gweight

    def ewald_ion_ion_recip(self):
        ii_dist = self.dist.dist_i(self.atom_coords, self.atom_coords) # (natoms, natoms, 3)
        g_dot_r = gpu.cp.einsum('kd,ijd->kij', self.gpoints, ii_dist) # (nk, natoms, natoms)
        gweight = self.eval_weight(ii_dist) # (nk, natoms, natoms)
        ion_ion_recip = gpu.cp.einsum('i,j,kij,kij->', self.atom_charges, self.atom_charges, gpu.cp.exp(1j * g_dot_r), gweight).real
        return ion_ion_recip

    def ewald_elec_ion_recip(self, configs):
        ei_dist = self.dist.dist_i(self.atom_coords, configs.configs).transpose((1, 0, 2, 3)) # (nconf, natoms, nelec, 3)
        g_dot_r = gpu.cp.einsum('kd,cijd->ckij', self.gpoints, ei_dist) # (nconf, nk, natoms, nelec)
        gweight = self.eval_weight(ei_dist) # (nconf, nk, natoms, nelec)
        elec_ion_recip = -2 * gpu.cp.einsum('i,ckij,ckij->c', self.atom_charges, gpu.cp.cos(g_dot_r), gweight)
        return elec_ion_recip

    def ewald_elec_elec_recip(self, configs):
        ee_dist = self.dist.dist_i(configs.configs, configs.configs).transpose((1, 0, 2, 3)) # (nconf, nelec, nelec, 3)
        g_dot_r = gpu.cp.einsum('kd,cijd->ckij', self.gpoints, ee_dist) # (nconf, nk, nelec, nelec)
        gweight = self.eval_weight(ee_dist) # (nconf, nk, nelec, nelec)
        elec_elec_recip = gpu.cp.einsum('ckij,ckij->c', gpu.cp.exp(1j * g_dot_r), gweight).real
        return elec_elec_recip

    def set_constants(self):
        self.const_self = self.alpha / gpu.cp.sqrt(gpu.cp.pi)

    def ewald_ion_ion_real_self(self):
        ion_ion_real_self = -self.const_self * gpu.cp.sum(self.atom_charges**2)
        return ion_ion_real_self

    def ewald_elec_elec_real_self(self, nelec):
        elec_elec_real_self = -self.const_self * nelec
        return elec_elec_real_self

    def eval_weight_charge(self, dist):
        z = dist[..., 2] # (natoms, natoms)
        w1 = z * gpu.erf(self.alpha * z)
        w2 = 1/(self.alpha * gpu.cp.sqrt(gpu.cp.pi)) * gpu.cp.exp(-self.alpha**2 * z**2)
        w = -gpu.cp.pi / self.cell_area * (w1 + w2)
        return w

    def ewald_ion_ion_charge(self):
        ii_dist = self.dist.dist_i(self.atom_coords, self.atom_coords) # (natoms, natoms, 3)
        weight = self.eval_weight_charge(ii_dist)
        ion_ion_charge = gpu.cp.einsum('i,j,ij->', self.atom_charges, self.atom_charges, weight)
        return ion_ion_charge

    def ewald_elec_ion_charge(self, configs):
        ei_dist = self.dist.dist_i(self.atom_coords, configs.configs).transpose((1, 0, 2, 3)) # (nconf, natoms, nelec, 3)
        weight = self.eval_weight_charge(ei_dist)
        elec_ion_charge = -2 * gpu.cp.einsum('i,cij->c', self.atom_charges, weight)
        return elec_ion_charge

    def ewald_elec_elec_charge(self, configs):
        ee_dist = self.dist.dist_i(configs.configs, configs.configs).transpose((1, 0, 2, 3)) # (nconf, nelec, nelec, 3)
        weight = self.eval_weight_charge(ee_dist)
        elec_elec_charge = gpu.cp.einsum('cij->c', weight)
        return elec_elec_charge

    def energy(self, configs):
        nelec = configs.configs.shape[1]
        ii_const = self.ewald_ion_ion_charge() + self.ewald_ion_ion_real_self()
        ii_real_cross = self.ewald_ion_ion_real_cross()
        ii_recip = self.ewald_ion_ion_recip()
        ii = ii_real_cross + ii_recip + ii_const

        ee_const = self.ewald_elec_elec_charge(configs) + self.ewald_elec_elec_real_self(nelec)
        ee_real_cross = self.ewald_elec_elec_real_cross(configs)
        ee_recip = self.ewald_elec_elec_recip(configs)
        ee = ee_real_cross + ee_recip + ee_const

        ei_const = self.ewald_elec_ion_charge(configs)
        ei_real_cross = self.ewald_elec_ion_real_cross(configs)
        ei_recip = self.ewald_elec_ion_recip(configs)
        ei = ei_real_cross + ei_recip + ei_const
        return ee, ei, ii
