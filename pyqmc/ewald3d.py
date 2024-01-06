import pyqmc
import pyqmc.energy
import pyqmc.gpu as gpu

class Ewald:

    def __init__(self, cell, gmax=200, nlatvec=1):
        self.latvec = cell.lattice_vectors()
        self.atom_coords = cell.atom_coords()
        self.nelec = gpu.cp.array(cell.nelec)
        self.atom_charges = gpu.cp.asarray(cell.atom_charges())
        self.dist = pyqmc.distance.MinimalImageDistance(self.latvec)
        self.cell_volume = gpu.cp.linalg.det(self.latvec)
        self.recvec = gpu.cp.linalg.inv(self.latvec).T
        self.set_alpha()
        self.set_lattice_displacements(nlatvec)
        self.set_gpoints_gweight(gmax)
        self.set_constants()

    def set_alpha(self):
        smallest_height = gpu.cp.amin(1 / gpu.cp.linalg.norm(self.recvec, axis=1))
        self.alpha = 5.0 / smallest_height

    def set_lattice_displacements(self, nlatvec):
        space = [gpu.cp.arange(-nlatvec, nlatvec + 1)] * 3
        XYZ = gpu.cp.meshgrid(*space, indexing='ij')
        xyz = gpu.cp.stack(XYZ, axis=-1).reshape((-1, 3))
        self.lattice_displacements = gpu.cp.asarray(gpu.cp.dot(xyz, self.latvec)) # (27, 3)

    def generate_positive_gpoints(self, gmax):
        gXpos = gpu.cp.mgrid[1 : gmax + 1, -gmax : gmax + 1, -gmax : gmax + 1].reshape(3, -1)
        gX0Ypos = gpu.cp.mgrid[0:1, 1 : gmax + 1, -gmax : gmax + 1].reshape(3, -1)
        gX0Y0Zpos = gpu.cp.mgrid[0:1, 0:1, 1 : gmax + 1].reshape(3, -1)
        gpts = gpu.cp.concatenate([gXpos, gX0Ypos, gX0Y0Zpos], axis=1)
        gpoints = gpu.cp.einsum('ji,jk->ik', gpts, gpu.cp.asarray(self.recvec) * 2 * gpu.cp.pi)
        return gpoints

    def set_gpoints_gweight(self, gmax, tol=1e-10):
        candidate_gpoints = self.generate_positive_gpoints(gmax)
        gsquared = gpu.cp.einsum('jk,jk->j', candidate_gpoints, candidate_gpoints)
        gweight = 4 * gpu.cp.pi * gpu.cp.exp(-gsquared / (4 * self.alpha**2))
        gweight /= self.cell_volume * gsquared
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
            # input to dist_matrix has the shape (nconf, nparticles, ndim)
            ion_ion_distances, ion_ion_idxs = self.dist.dist_matrix(self.atom_coords[None])
            ion_ion_cij = self.eval_real_cij(ion_ion_distances[:, :, None, :], self.lattice_displacements[None, None, :, :]) # (nconf=1, npairs=choose(4, 2))
            ion_ion_charge_ij = gpu.cp.prod(self.atom_charges[ion_ion_idxs], axis=1) # (npairs=6,)
            ion_ion_real_cross = gpu.cp.einsum('j,ij->i', ion_ion_charge_ij, ion_ion_cij) # (nconf=1,)
        return ion_ion_real_cross

    def set_constants(self):
        self.const_self = self.alpha / gpu.cp.sqrt(gpu.cp.pi)
        self.const_charge = gpu.cp.pi / (2 * self.cell_volume * self.alpha**2)

    def ewald_ion_ion_real_self(self):
        ion_ion_real_self = -self.const_self * gpu.cp.sum(self.atom_charges**2)
        return ion_ion_real_self

    def ewald_elec_elec_real_self(self, nelec):
        elec_elec_real_self = -self.const_self * nelec
        return elec_elec_real_self

    def ewald_ion_ion_charge(self):
        ion_ion_charge = -self.const_charge * gpu.cp.sum(self.atom_charges)**2
        return ion_ion_charge

    def ewald_elec_ion_charge(self, nelec):
        elec_ion_charge = self.const_charge * gpu.cp.sum(self.atom_charges)*nelec*2
        return elec_ion_charge

    def ewald_elec_elec_charge(self, nelec):
        elec_elec_charge = -self.const_charge * nelec**2
        return elec_elec_charge

    def ewald_elec_ion_real_cross(self, configs):
        elec_ion_dist = configs.dist.dist_i(self.atom_coords, configs.configs) # (nelec=4, nconf=1, natoms=4, ndim=3)
        elec_ion_cij = self.eval_real_cij(elec_ion_dist[:, :, :, None, :], self.lattice_displacements[None, None, None, :, :]) # (nelec=4, nconf=1, natoms=4)
        elec_ion_real = gpu.cp.einsum('k,ijk->j', -self.atom_charges, elec_ion_cij) # (nconf,)
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

    def ewald_ion_ion_recip(self):
        ion_g_dot_r = gpu.cp.dot(self.gpoints, gpu.cp.asarray(self.atom_coords.T))
        self.ion_exp = gpu.cp.dot(gpu.cp.exp(1j * ion_g_dot_r), self.atom_charges)
        ion_ion_recip = gpu.cp.dot(gpu.cp.abs(self.ion_exp) ** 2, self.gweight)
        return ion_ion_recip

    def ewald_elec_elec_recip(self, configs):
        self.elec_g_dot_r = gpu.cp.einsum('kh,ijh->ijk', self.gpoints, gpu.cp.asarray(configs.configs)) # (nconf, nelec, nk)
        self.elec_exp = gpu.cp.sum(gpu.cp.exp(1j * self.elec_g_dot_r), axis=1)
        elec_elec_recip = gpu.cp.dot(gpu.cp.abs(self.elec_exp) ** 2, self.gweight)
        return elec_elec_recip

    def ewald_elec_ion_recip(self, configs):
        elec_exp = gpu.cp.sum(gpu.cp.exp(-1j * self.elec_g_dot_r), axis=1)
        elec_ion_recip = -2 * gpu.cp.dot(self.ion_exp * elec_exp, self.gweight).real
        return elec_ion_recip

    def energy(self, configs):
        nelec = configs.configs.shape[1]
        ii_const = self.ewald_ion_ion_charge() + self.ewald_ion_ion_real_self()
        ei_const = self.ewald_elec_ion_charge(nelec)
        ee_const = self.ewald_elec_elec_charge(nelec) + self.ewald_elec_elec_real_self(nelec)
        ii = self.ewald_ion_ion_real_cross() + self.ewald_ion_ion_recip() + ii_const
        ee = self.ewald_elec_elec_real_cross(configs) + self.ewald_elec_elec_recip(configs) + ee_const
        ei = self.ewald_elec_ion_real_cross(configs) + self.ewald_elec_ion_recip(configs) + ei_const
        return ee, ei, ii
