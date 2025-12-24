import numpy as np
import pyqmc.api as pyq

def compute_wf_grid(wf, coords, electron, X, Y, Z, verbose=True):
    nconf=coords.configs.shape[0]
    wfval = np.zeros((nconf,np.prod(X.shape)))
    wf.recompute(coords)
    if verbose:
        npts = len(X.flatten())
        print(f"Computing {npts} points", flush=True)
        printouts = [int(npts*i/10) for i in range(10)]

    for i, (x,y, z) in enumerate(zip(X.flatten(), Y.flatten(), Z.flatten())):
        epos = np.tile(np.array([x,y,z]), (coords.configs.shape[0],1))
        epos = coords.make_irreducible(electron, epos)
        wfval[:,i] = wf.testvalue(electron, epos)[0]
        if verbose and i in printouts:
            print("Fraction done", i/npts)
    return wfval.reshape( (nconf, *X.shape))


def compute_elocal_grid(wf, coords, electron, X, Y, Z, enacc):
    coords = coords.copy()
    nconf=coords.configs.shape[0]
    wfval = np.zeros((nconf,np.prod(X.shape)))
    wf.recompute(coords)
    enavg = enacc.avg(coords, wf)['total']
    for i, (x,y, z) in enumerate(zip(X.flatten(), Y.flatten(), Z.flatten())):
        coords.configs[:,electron,:] = np.array([x,y,z])[np.newaxis,:]
        wfval[:,i] = enacc(coords, wf)['total']-enavg
    return wfval.reshape( (nconf, *X.shape))


def write_atoms_to_xsf(mol,  output, electron_positions=None):
    output.write("ATOMS\n")
    for i, pos in enumerate(mol.atom_coords()):
        output.write(f"{mol.atom_symbol(i)} {pos[0]} {pos[1]} {pos[2]}\n")
    if electron_positions is not None:
        for pos in electron_positions:
            output.write(f"E {pos[0]} {pos[1]} {pos[2]}\n")


def write_grid_to_xsf(lattice_vectors, vals, output, origin=(0.,0.,0.)):
    """Vals should be in x,y,z. Resolution should be a """
    output.write("BEGIN_BLOCK_DATAGRID_3D\n comment \n")
    output.write("BEGIN_DATAGRID_3D_this_is_3Dgrid\n")
    output.write(f"{vals.shape[0]} {vals.shape[1]} {vals.shape[2]} \n")
    output.write(f"{origin[0]} {origin[1]} {origin[2]} \n")
    for vec in lattice_vectors:
        output.write(f"{vec[0]} {vec[1]} {vec[2]} \n")
    for k in range(vals.shape[2]):
        for j in range(vals.shape[1]):
            for i in range(vals.shape[0]):
                output.write(f"{vals[i,j,k]} \n")

    output.write("END_DATAGRID_3D\n")
    output.write("END_BLOCK_DATAGRID_3D\n")


def generate_grid(origin, latvec, resolution):
    for i,j in [(0,1),(0,2),(1,2), (2,1), (0,2),(2,0)]:
        if abs(latvec[i,j]) > 1e-8:
            raise Exception("Can't generate grid for non-orthogonal boxes right now.")
    size_x = latvec[0,0]
    size_y = latvec[1,1]
    size_z = latvec[2,2]
    x = np.linspace(origin[0], origin[0]+size_x, int(size_x/resolution))
    y = np.linspace(origin[1], origin[1]+size_y, int(size_y/resolution))
    z = np.linspace(origin[2], origin[2]+size_z, int(size_z/resolution))
    return np.meshgrid(x,y, z)


def bbox_from_mol(mol, buffer = 4.0):
    min_pos = np.amin(mol.atom_coords(), axis=0)
    max_pos = np.amax(mol.atom_coords(), axis=0)

    print(min_pos, max_pos)
    origin = min_pos-buffer
    latvec = np.zeros((3,3))
    for i in range(3):
        latvec[i,i]=max_pos[i]+buffer - origin[i]
    return origin, latvec

def plot_conditional_wf(mol, wf, coords, outputroot="conditional", resolution=0.2, buffer=4.0, electron=0, quantity = 'wavefunction_value'):
    """
    quantity is one of wavefunction_value, wavefunction_squared, elocal
    """
    origin, latvec = bbox_from_mol(mol, buffer=buffer)
    X,Y, Z = generate_grid(origin, latvec, resolution)
    if quantity=='elocal':
        enacc = pyq.EnergyAccumulator(mol)
        val = compute_elocal_grid(wf, coords, electron, X,Y,Z, enacc)
    else:
        val = compute_wf_grid(wf, coords, electron, X,Y,Z)
        if quantity=='wavefunction_squared':
            val = np.abs(val)**2
    nconf = coords.configs.shape[0]
    for w in range(nconf):
        f = open(f"{outputroot}_{w}.xsf", 'w')
        other_electrons = np.arange(coords.configs.shape[1]) != electron
        write_atoms_to_xsf(mol, f, electron_positions=coords.configs[w,other_electrons,:])
        write_grid_to_xsf(latvec, val[w],f , origin=origin)



if __name__=="__main__":

    import pyscf
    import pyqmc.api as pyq
    import numpy as np
    mol, mf = pyq.recover_pyscf("mf.hdf5", cancel_outputs=False)
    mc = pyscf.mcscf.CASSCF(mf, 2, 2).run()
    wf, _ = pyq.generate_wf(mol,mf, mc=mc)
    pyq.read_wf(wf, "opt.hdf5")
    nconf=10
    coords = pyq.initial_guess(mol, nconf)
    data, coords = pyq.vmc(wf, coords, nblocks=1)
    plot_conditional_wf(mol, wf, coords, resolution = 0.2)
