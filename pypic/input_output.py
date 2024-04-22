import numpy as np
import os
import h5py
import pypic
from pypic.fields import *
from pypic.particles import *
import pandas as pd
import datetime

def read_particles_from_selection_cycles(cycles, selection):
    """ Reach particle data from selection files split up by cycle """
    df_data = [[] for i in range(13)]

    # fname1 = 'df_S0_185k_pt2.h5'
    # df1 = pd.read_hdf(fname1, key='Block')
    # q1 = df1['q'].unique()
    # print(f'Number of q1 = {len(q1)}')
    # print(f'Min cycle = {df1["cycle"].min()}')
    # print(f'Max cycle = {df1["cycle"].max()}')
    # print('----------------------------------')

    # fname2 = 'df_S0_185k.h5'
    # df2 = pd.read_hdf(fname2, key='Block')
    # df2 = df2[(df2['cycle'] > 185500)]
    # df2 = df2[df2['q'].isin(q1)]
    # q2 = df2['q'].unique()
    # print(f'Number of q2 = {len(q2)}')
    # print(f'Min cycle 2 = {df2["cycle"].min()}')
    # print(f'Max cycle 2 = {df2["cycle"].max()}')
    # df = pd.concat([df1, df2])
    # df.to_hdf(f'df_S0_186k.h5', key='Block', mode='w', index=False)
    # print(f'Min cycle all = {df["cycle"].min()}')
    # print(f'Max cycle all = {df["cycle"].max()}')
    # exit()

    # load reference dataframe
    # fname_ref = f'df_ref_S{selection.species}.h5'
    fname_ref = f'/Users/lrusaiti/DATA/selection_S1/df_S1_186000.h5'
    # fname_ref = f'/Users/leo/DATA/selection_S0_185k/df_S0_170500.h5'
    df_ref = pd.read_hdf(fname_ref, key='Block')
    print(df_ref)
    q_ref = df_ref['q'].unique()
    print(f'Number of q_ref = {len(q_ref)}')
    # print(f'Min cycle = {df_ref["cycle"].min()}')
    # print(f'Max cycle = {df_ref["cycle"].max()}')
    # exit()

    for cycle in cycles:
        s = selection
        s.cycle = cycle
        s.name = f'S{s.species}'
        print(f'Processing cycle {cycle}')
        selection_folder = f'selection_S{s.species}'
        input_file = os.path.join(s.data_dir, selection_folder,
                                  s.selection_filenames["particles"])
        read_from_original_file = False
        if read_from_original_file:
            particles, attr = load_hdf5_array(input_file,
                                              keys = selection.particle_data_keys,
                                              split_by_species = True,
                                              selection  = None,
                                              incr_write = False,
                                              incr_read  = False,
                                              # fast_search = True,
                                              dtype      = selection.dtype,
                                              write_filename = None,
                                              verbose    = False,
                                              # every_nth = 100,
                                              )
            particles.name = selection.name
            particles.to_phys(selection.sim)
            particles.average_movement()
            particles.check_unique()
            particle = particles.__dict__[particles.names[selection.species]]
            print(f'Number of particles = {particle.N}')

            cycles = np.full(particle.N, cycle)
            species = np.full(particle.N, selection.species)
            speeds = np.linalg.norm(particle.v, axis=0) * selection.sim.c_phys / 1000.
            rs = np.linalg.norm(particle.r, axis=0)
            if selection.species in [0,2]:
                energies = selection.sim.me * (speeds*1000)**2 / 2. / selection.sim.e / 1000
            else:
                energies = selection.sim.mp * (speeds*1000)**2 / 2. / selection.sim.e / 1000
            angles =  np.arctan2(particle.v[2], np.sqrt(particle.v[0]**2 + particle.v[1]**2)) * 180. / np.pi

            data_new = [cycles, species, *particle.r, *particle.v, particle.q, speeds, energies, angles, rs]
            data_new = np.asarray(data_new)
            df = pd.DataFrame(data_new.T, columns=['cycle', 'species', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'q', 'speed', 'energy', 'angle', 'r'], dtype=selection.dtype)
            df = df.drop_duplicates(keep='last')
            # # df = df[(df['x'] > 0) & (df['z'] > 0)]
            df.to_hdf(f'df_S{selection.species}_{cycle:06d}.h5', key='Block', mode='w', index=False)
            # df = df[df['q'].isin(q_ref)]


        fname = f'/Users/lrusaiti/DATA/selection_S1/df_S1_{cycle:06d}.h5'
        df = pd.read_hdf(fname, key='Block',
                         # columns=['cycle', 'species', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'q', 'speed', 'energy', 'angle', 'r'],
                         )

        # combine df_ref and df
        df_ref = pd.concat([df_ref, df])
        # np.save(f'q_unique_el_S0.npy', q_unique)
        # q_unique = np.load(f'q_unique_el_S0.npy')
        # df_data = np.hstack((df_data, df_data_new))

    # print(f'Finished processing cycles.')
    # exit()

    # Saving the dataframe to a single file
    # print(f'Creating a dataframe...')
    # df = pd.DataFrame(df_data.T, columns=['cycle', 'species', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'q', 'speed', 'energy', 'angle', 'r'], dtype=selection.dtype)

    filename = f'df_S{selection.species}_{int(cycles[-1]/1000):d}k_pt0'
    print(f'Saving HDF5 file... ')
    df_ref.to_hdf(f'{filename}.h5', key='Block', mode='w', index=False)
    print(f'Saving npy file... ')
    np.save(f'{filename}.npy', df_ref.to_numpy())

def read_particle_dataframe(filename,
                                  read_fields=False,
                                  drop_duplicates=False,
                                  save_in_place=False,
                                  dtype=np.float64,
                                  ):
    """ Read particles from HDF5 or npy file and return a dataframe """
    # filename = 'df_ions_152k_2.npy'
    # filename = 'df_ions_152k_clean.h5'
    # filename1 = 'df_ions_152k_pt1_16.npy'
    # filename2 = 'df_ions_152k_pt2_16.npy'
    # df1 = pd.DataFrame(np.load(filename1), columns=['cycle', 'species', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'q', 'speed', 'energy', 'angle', 'r'])
    # df2 = pd.DataFrame(np.load(filename2), columns=['cycle', 'species', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'q', 'speed', 'energy', 'angle', 'r'])
    # df = pd.concat([df1, df2])
    # save df to an h5 file
    # df.to_hdf('df_ions_all_16.h5', key='Block', mode='w', index=False)
    # np.save('df_ions_all_64.npy', df.to_numpy(dtype=np.float64))

    if read_fields:
        columns = ['cycle', 'species', 'x', 'y', 'z',
                          'vx', 'vy', 'vz', 'q',
                          'speed', 'energy', 'angle', 'r',
                          'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz',
                          'Jx', 'Jy', 'Jz', 'Vx', 'Vy', 'Vz',
                          'vexbx', 'vexby', 'vexbz', 'rho', 'N']
    else:
        columns = ['cycle', 'species', 'x', 'y', 'z',
                   'vx', 'vy', 'vz', 'q', 'speed', 'energy', 'angle', 'r']

    if filename[-2:] == 'h5':
        # df = pd.read_hdf(filename, key='Block')
        # df = pd.read_hdf(filename, key='q_1')
        # keys = ['q_1', 'u_1', 'v_1', 'w_1', 'x_1', 'y_1', 'z_1']
        # df = load_hdf5_array(filename,
        #             keys             = keys,
        #             split_by_species = False,
        #             selection        = None,
        #             incr_write       = False,
        #             incr_read        = False,
        #             dtype            = 'f8',
        #             fast_search      = True,
        #             fast_search_ind  = [0, 1, 2],
        #             write_filename   = 'dataset.h5',
        #             )
        # df = np.asarray(df[0]).T
        # print(f'shape of df = {df.shape}')
        # print('Done.')
        # df = pd.DataFrame(df, columns=columns)
        df = pd.read_hdf(filename)
        # df = pd.read_hdf(filename, columns = ['cycle', 'species', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'q'])
        # df = pd.read_hdf(filename, columns = )
    else:
        df = pd.DataFrame(np.load(filename), columns=columns)
    dtypes = {'cycle': 'int32',
               'species': 'int8',
               'x': 'float32',
               'y': 'float32',
               'z': 'float32',
               'vx': 'float32',
               'vy': 'float32',
               'vz': 'float32',
               'q': 'float64',
               'speed': 'float32',
               'energy': 'float32',
               'angle': 'float32',
               'r': 'float32',
               }
    # df.astype(dtypes).dtypes
    # df.to_hdf('df_ions_all_32m.h5', key='Block', mode='w', index=False)
    # np.save('df_ions_all_32m.npy', df.to_numpy(dtype=dtypes))


    if drop_duplicates:
        print('Dropping duplicates.')
        print(df)
        df = df.drop_duplicates(keep='last')
        # df.drop_duplicates(inplace=True)
        print(df)
        exit()
    if save_in_place:
        print(f'Saving dataframe in place.')
        np.save(filename, df.to_numpy(dtype=np.float32))
        # df.to_hdf(filename, key='Block', mode='w', index=False)
    return df

def read_particle_fields(filename,
                         selection,
                         q=None,
                         fixed_cycle=None,
                         smooth_length_phys=2.,
                         smooth_std = 2):
    """ Read particle data from a file and add field data """
    print(f'Reading particle dataframe.')
    df = read_particle_dataframe(filename)
    if q is None:
        q = df['q'].unique()[0]
    # Select only one particle
    dfpick = df[df['q'] == q]
    dfpick.drop_duplicates(subset=['cycle', 'q'], inplace=True)
    nselected = len(dfpick["q"].unique())
    print(f'Number of selected particles: {nselected:,d}')
    if nselected == 0:
        print(f'No particles found in file {filename}.')
        return None
    df = dfpick
    # sort df by cycle
    df = df.sort_values(by=['cycle'])
    df['vx'] = -df['vx'] # flip the x velocity (FIX)
    cycles0 = np.arange(2000, 131000, 1000)
    cycles1 = np.arange(131000, 202500+500, 500)
    cycles = np.concatenate((cycles0, cycles1))
    # print(df.head())
    df = interpolate_trajectory(df, t=cycles)
    # df = calculate_derivate_quantities(df, selection=selection)
    # print(df.head())
    # exit()
    dfpick = df

    # Add empty new columns to the dataframe
    df = df.assign(Ex=np.nan, Ey=np.nan, Ez=np.nan,
                   Bx=np.nan, By=np.nan, Bz=np.nan,
                   Jx=np.nan, Jy=np.nan, Jz=np.nan,
                   Vx=np.nan, Vy=np.nan, Vz=np.nan,
                   vexbx=np.nan, vexby=np.nan, vexbz=np.nan,
                   rho=np.nan, N=np.nan,)

    # Get all cycles (times) in the particle data
    # unique
    particle_cycles = np.unique(df['cycle'].to_numpy()).astype(int)

    # Loop over all cycles and load the field data
    for cycle in particle_cycles:
        print(f'{f"Processing cycle {int(cycle):,d}":-^72}')
        selection.cycle = cycle
        if fixed_cycle is not None:
            selection.cycle = fixed_cycle
        # Adjust the selection to the particle's position
        position = dfpick[dfpick["cycle"] == cycle][['x','y','z']].to_numpy()
        position = position.squeeze()
        selection.center_phys = position
        selection.delta_phys =  [smooth_length_phys, smooth_length_phys, 0]
        sim = selection.sim
        sp = selection.species
        qom = sim.qoms[sp]
        small = 1e-12*qom

        # Load the data from the field file
        B_fields, _ = load_fields(selection, keys=['Bx', 'By', 'Bz'])
        E_fields, _ = load_fields(selection, keys=['Ex', 'Ey', 'Ez'])
        j_fields, _ = load_fields(selection, keys=[f'Jx_{sp}', f'Jy_{sp}', f'Jz_{sp}'])
        j0_fields, _ = load_fields(selection, keys=[f'Jx_0', f'Jy_0', f'Jz_0'])
        j1_fields, _ = load_fields(selection, keys=[f'Jx_1', f'Jy_1', f'Jz_1'])
        rho_fields, _ = load_fields(selection, keys=[f'rho_{sp}'])
        n_fields, _ = load_fields(selection, keys=[f'N_{sp}'])

        # Gaussian Smoothing
        E_fields = [pypic.fields.smooth_field(_, std=smooth_std*2) for _ in E_fields]
        B_fields = [pypic.fields.smooth_field(_, std=smooth_std/2.) for _ in B_fields]
        j_fields = [pypic.fields.smooth_field(_, std=smooth_std/1.) for _ in j_fields]
        j0_fields = [pypic.fields.smooth_field(_, std=smooth_std/1.) for _ in j0_fields]
        j1_fields = [pypic.fields.smooth_field(_, std=smooth_std/1.) for _ in j1_fields]
        rho_fields = [pypic.fields.smooth_field(_, std=smooth_std/1.) for _ in rho_fields]

        # Calculations
        bx, by, bz = B_fields
        ex, ey, ez = E_fields
        jx, jy, jz = j_fields
        jx0, jy0, jz0 = j0_fields
        jx1, jy1, jz1 = j1_fields
        n = n_fields[0]
        rho = np.abs(rho_fields[0]) + small
        b = np.sqrt(bx**2 + by**2 + bz**2) + small
        vexbx, vexby, vexbz = np.cross(E_fields, B_fields, axis=0)
        vexbx, vexby, vexbz = vexbx/b, vexby/b, vexbz/b
        vx = jx/rho*np.sign(qom)
        vy = jy/rho*np.sign(qom)
        vz = jz/rho*np.sign(qom)
        jtx, jty, jtz = jx1+jx0, jy1+jy0, jz1+jz0

        # Normalization to Physical Units
        density = rho*sim.get_scale('rho', sp)
        bx, by, bz = (pypic.core.code_to_phys_rot([bx, by, bz], True)
                      *sim.get_scale('B', sp))
        ex, ey, ez = (pypic.core.code_to_phys_rot([ex, ey, ez], True)
                      *sim.get_scale('E', sp))
        vx, vy, vz = (pypic.core.code_to_phys_rot([vx, vy, vz], True)
                      *sim.get_scale('v', sp))
        vexbx, vexby, vexbz = (pypic.core.code_to_phys_rot([vexbx, vexby, vexbz], True)
                               *sim.get_scale('v', sp))
        jtx, jty, jtz = (pypic.core.code_to_phys_rot([jtx, jty, jtz], True)
                         *sim.get_scale('j', sp))
        ind = bx.shape[0]//2

        # Save to dataframe
        dfpick.loc[dfpick['cycle'] == cycle, 'Ex'] = ex[ind, ind]
        dfpick.loc[dfpick['cycle'] == cycle, 'Ey'] = ey[ind, ind]
        dfpick.loc[dfpick['cycle'] == cycle, 'Ez'] = ez[ind, ind]
        dfpick.loc[dfpick['cycle'] == cycle, 'Bx'] = bx[ind, ind]
        dfpick.loc[dfpick['cycle'] == cycle, 'By'] = by[ind, ind]
        dfpick.loc[dfpick['cycle'] == cycle, 'Bz'] = bz[ind, ind]
        dfpick.loc[dfpick['cycle'] == cycle, 'Jx'] = jx[ind, ind]
        dfpick.loc[dfpick['cycle'] == cycle, 'Jy'] = jy[ind, ind]
        dfpick.loc[dfpick['cycle'] == cycle, 'Jz'] = jz[ind, ind]
        dfpick.loc[dfpick['cycle'] == cycle, 'Vx'] = vx[ind, ind]
        dfpick.loc[dfpick['cycle'] == cycle, 'Vy'] = vy[ind, ind]
        dfpick.loc[dfpick['cycle'] == cycle, 'Vz'] = vz[ind, ind]
        dfpick.loc[dfpick['cycle'] == cycle, 'vexbx'] = vexbx[ind, ind]
        dfpick.loc[dfpick['cycle'] == cycle, 'vexby'] = vexby[ind, ind]
        dfpick.loc[dfpick['cycle'] == cycle, 'vexbz'] = vexbz[ind, ind]
        dfpick.loc[dfpick['cycle'] == cycle, 'rho'] = density[ind, ind]
        dfpick.loc[dfpick['cycle'] == cycle, 'N'] = n[ind, ind]

    # Save the dataframe to a file
    filename = filename.split('.')[0]
    np.save(f'{filename}_{int(abs(q)*1e12)}_fields.npy', dfpick.to_numpy())
    # save dfpick to h5
    dfpick.to_hdf(f'{filename}_{int(abs(q)*1e12)}_fields.h5', key='Block', mode='w', index=False)

def find_key_in_strings(splits, key, fmt='str'):
    """ Find a key in a list of strings and return the value """
    value = [s.strip(key).strip('[]') for s in splits if key in s]
    if np.size(value) > 0:
        value = value[0]
        # Check if the value is a range
        if 'to' in value:
            return [format_string(value.split('to')[0], fmt),
                    format_string(value.split('to')[1], fmt)]
        else:
            return format_string(value, fmt)
    else:
        return None

def format_string(string, fmt='str'):
    """ Format a string as an integer, float, or string """
    if fmt == 'int':
        return int(string)
    elif fmt == 'float':
        return float(string)
    else:
        return string

def get_files(dir_path, basename):
    """ Get all files in a directory with a given basename and wildcard """
    path = os.path.join(dir_path, basename)
    filelist = glob.glob(path)
    return(filelist)

def get_all_particle_cycles(selection):
    """ Find all model times (cycles) available """
    fnames = get_files(selection.data_dir, f'{selection.filebase_particles}_*_*NS_*.h5')
    cycles = []
    for fn in fnames:
        base, cyc, options = strip_particle_filename(fn, ext='.h5')
        cycles.append(cyc)
    cycles = np.unique(cycles).astype(int)
    return cycles

def get_all_particle_files(selection, note=None, dx=None):
    """ Find all particle files for a given time (cycle) and species 
        If no cycle is given, the latest cycle is found in data dir. """
    if selection.cycle is None:
        cycles_found = get_all_particle_cycles(selection)
        if np.size(cycles_found) == 0:
            print(f'No particle files for any cycle found.')
            return None, None, None, None
        else:
            cycle = np.max(cycles_found)
            selection.change_cycle(cycle)
    fglob = f'{selection.filebase_particles}_{selection.cycle:06d}_{selection.ns}NS_*.h5'
    fnames = get_files(selection.data_dir, fglob)
    fnames_selected = []
    centers = []
    sizes = []
    for fn in fnames:
        _, _, options = strip_particle_filename(fn, ext='.h5')
        x = np.asarray(options['x'])
        y = np.asarray(options['y'])
        z = np.asarray(options['z'])
        if dx != options['dx']: # Skip if dx doesn't match
            continue
        if note != options["note"]: # Skip if both None or match
            continue
        fnames_selected.append(fn)
        dx = options['dx']
        if np.size(x) > 1: # Coordinates were given as a range
            centers.append([x[0] + (x[1]-x[0])/2,
                            y[0] + (y[1]-y[0])/2,
                            z[0] + (z[1]-z[0])/2])
            sizes.append([x[1]-x[0], y[1]-y[0], z[1]-z[0]])
        else:              # Coordinates were given as a center value
            centers.append([x, y, z])
            sizes.append([dx, dx, dx])
    if len(fnames_selected) == 0:
        print(f'No particle files found for cycle {selection.cycle:,}'
              +f'(dx={dx}, note={note}).')
        return None, None, None, None
    else:
        centers = np.asarray(centers)
        sizes = np.asarray(sizes)
        first_choice = np.argmin(centers[:,0]) # Choose the file with the min x
        return fnames_selected, centers, sizes, first_choice

def get_particle_file(selection, note=None, dx=None, fname=None):
    """ Find the file for a particle time, species and position.
        Returns the filename and the selection box object."""
    if fname is not None: # If a filename is given, use it
        fnames = [fname]
    else:
        fglob = (f'{selection.filebase_particles}_'
                 +f'{selection.cycle:06d}_'
                 +f'{selection.ns}NS_*.h5')
        fnames = get_files(selection.data_dir, fglob)
    range_min_phys = None
    range_max_phys = None
    selection_particles = None
    center = selection.center_phys
    for fn in fnames:
        _, _, options = strip_particle_filename(fn, ext='.h5')
        x = np.asarray(options['x'])
        y = np.asarray(options['y'])
        z = np.asarray(options['z'])
        if dx is not None and dx != options['dx']: # Skip if dx doesn't match
            continue
        if note != options["note"]: # Skip if both None or match
            continue
        dx = options['dx']
        if np.size(x) > 1: # Coordinates were given as a range
            if center[0] >= x[0] and center[0] < x[1] and \
                 center[1] >= y[0] and center[1] < y[1] and \
                 center[2] >= z[0] and center[2] < z[1]:
                range_min_phys = [x[0], y[0], z[0]]
                range_max_phys = [x[1], y[1], z[1]]
                fname = fn
                # range_min_phys, range_max_phys = sort_selection(range_min_phys, range_max_phys)
                break
        else:             # Coordinates were given as a center value
            if center[0] >= x-dx/2. and center[0] < x+dx/2. and \
               center[1] >= y-dx/2. and center[1] < y+dx/2. and \
               center[2] >= z-dx/2. and center[2] < z+dx/2.:
                range_min_phys = [x-dx/2., y-dx/2., z-dx/2.]
                range_max_phys = [x+dx/2., y+dx/2., z+dx/2.]
                fname = fn
                # range_min_phys, range_max_phys = sort_selection(range_min_phys, range_max_phys)
                break
    if fname is not None and range_min_phys is not None:
        # Create a selection box object for the particle file
        selection_particles = copy.copy(selection)
        selection_particles.min_phys = range_min_phys
        selection_particles.max_phys = range_max_phys
        selection_particles.calculate_selection_from_range()
    else:
        print(f'No particle file found for cycle {selection.cycle:,}')
        print(f'fnames = {fnames}')
        # exit()
    return fname, selection_particles

def strip_particle_filename(fname, ext='.h5'):
    """ Strip the filename of the particle file """
    split_fname = fname.split('_')  # Split with underscores
    split_fname[-1] = split_fname[-1].replace(ext, '') # Remove extension
    base = split_fname[0]   # First part is the base
    cycle = split_fname[1]  # Second part is the model cycle
    split_fname.pop(0)      # Remove the base
    split_fname.pop(0)      # Remove the cycle
    species = find_key_in_strings(split_fname, 'NS', 'int') # Find the species
    x = find_key_in_strings(split_fname, 'X', 'float')      # Find the x pos
    y = find_key_in_strings(split_fname, 'Y', 'float')      # Find the y pos
    z = find_key_in_strings(split_fname, 'Z', 'float')      # Find the z pos
    dx = find_key_in_strings(split_fname, 'dx', 'float')    # Find the dx
    note = find_key_in_strings(split_fname, ']', 'str')     # Find the note

    optional = dict([('species', species),
                     ('x', x), ('y', y), ('z', z),
                     ('dx', dx), ('note', note)])
    return base, cycle, optional

def save_hdf5_slice(selection=None,
                    filename=None,
                    dtype='f8',
                    save_keys=None,
                    cut_to_selection=True,
                    ):
    """ Save a slice of a field file to a new file """
    
    if filename is None:
        filename = selection.path_fields(selection.cycle)

    with h5py.File(filename, 'r') as f:
        dataset = f
        dataset_names = dataset.keys()
        if save_keys is None:
            save_keys = dataset_names

        # check attributes
        attributes = dataset.attrs
        attributes = {attr: dataset.attrs[attr] for attr in attributes}
        if "name" not in attributes:
            attributes["name"] = 0
        if "coord" not in attributes:
            attributes["coord"] = "code"
        if "track" not in attributes:
            attributes["track"] = 0

        # check for groups and blocks
        if "Step#0" in dataset_names:
            dataset = dataset["Step#0"]
            dataset_names = dataset.keys()
        if "Block" in list(dataset_names):
            dataset = dataset["Block"]
            dataset_names = dataset.keys()

        # load and collect field cuts
        fields = []
        # for ds in dataset_names:
        for ds in save_keys:
            data = dataset[ds]
            if not isinstance(data, h5py.Dataset):
                data = data["0"]
            if cut_to_selection:
                fields.append(cut_field(data, selection, clip=False))
            else:
                fields.append(data)

        if cut_to_selection:
            # get cut axis and index
            cut_axis = selection.cut_axis_code
            cut_index = int(selection.center_cell[cut_axis])
            cut_axis_name = ['X', 'Y', 'Z'][cut_axis]
            new_filename = f'{filename.split(".")[0]}_{cut_axis_name}{cut_index}.h5'
            new_filename = os.path.basename(new_filename)
        else:
            first_letter_key = save_keys[0][0].upper()
            new_filename = f'{filename.split(".")[0]}_{first_letter_key}3D.h5'
            new_filename = os.path.basename(new_filename)

        # check if local data directory exists
        if not os.path.exists(selection.sim.local_data_dir):
            os.makedirs(selection.sim.local_data_dir)
        new_filename = os.path.join(selection.sim.local_data_dir, new_filename)

        # save field cuts to a new hdf5 file
        with h5py.File(new_filename, 'w') as g:
            print(f'Saving field cuts to {new_filename}.')
            # for i, key in enumerate(dataset_names):
            for i, key in enumerate(save_keys):
                g.create_dataset(key, data=fields[i], dtype=dtype)
            # save attributes
            attributes["datetime"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # print(f'attributes = {attributes}')
            for key in attributes:
                g.attrs[key] = attributes[key]  

def load_hdf5_array(filename,
                    keys             = ['x'],
                    split_by_species = False,
                    selection        = None,
                    incr_write       = False,
                    incr_read        = False,
                    dtype            = 'f4',
                    chunk_size       = 1e7,
                    fast_search      = True,
                    fast_search_ind  = [0, 1, 2],
                    write_filename   = 'dataset.h5',
                    erase_old        = False,
                    verbose          = False,
                    every_nth        = None,
                    ):
    """ Load HDF5 file and return data object and attributes """

    def print_progress(verbose, ns, start_ind, end_in, N):
        if verbose:
            print(f'Loading particle {ns}. Indices {start_ind:,} - {end_ind:,}'
                 +f'. Selected {N:,}.')

    with h5py.File(filename, 'r') as f:
        dataset = f
        dataset_names = list(f)
        attributes = dataset.attrs
        attributes = {attr: dataset.attrs[attr] for attr in attributes}
        if "name" not in attributes:
            attributes["name"] = None
        if "coord" not in attributes:
            attributes["coord"] = "code"
        if "track" not in attributes:
            attributes["track"] = None
        if "Step#0" in dataset_names:
            dataset = f.get("Step#0")
            dataset_names = list(dataset)
        if "Block" in list(dataset_names):
            dataset = dataset.get("Block")
        try:
            if split_by_species:
                ns_avail = np.unique([int(x.split('_')[1]) for x in dataset_names])
                keys_ns = [f"{key}_{ns}" for key in keys for ns in ns_avail]
                test_data = dataset.get(keys_ns[0])
                if test_data is None:
                    return None, None
                del test_data
                selection = np.atleast_1d(selection)
                species = [getattr(s, 'species', None) for s in selection]
                if erase_old:
                    dir = None if selection[0] is None else selection[0].output_dir
                    if write_filename is not None and os.path.exists(os.path.join(dir, write_filename)):
                        os.remove(os.path.join(dir, write_filename))
                    else:
                        for s in selection:
                            if s is not None:
                                wfile = s.selection_filenames["particles"]
                                if os.path.exists(os.path.join(dir, wfile)):
                                    os.remove(os.path.join(dir, wfile))
                container_opt = dict(name =attributes["name"],
                                     coord=attributes["coord"],
                                     track=attributes["track"])
                particle_data = particleContainer(**container_opt)
                if incr_read:
                    for ns in ns_avail:
                        if (None not in species and ns not in species):
                            continue
                        data = [dataset.get(f"{key}_{ns}") for key in keys]
                        partcl_opt = dict(name  = particle_data.names[ns],
                                          coord = attributes["coord"])
                        for start_ind in range(0, len(data[0]), int(chunk_size)):
                            end_ind = min(start_ind + int(chunk_size), len(data[0]))

                            # if fast_search and None not in selection:
                            if fast_search:
                                # chunk = data[:][start_ind:end_ind]
                                # print(f'type of chunk = {type(chunk)}')
                                x = data[fast_search_ind[0]][start_ind:end_ind]
                                y = data[fast_search_ind[1]][start_ind:end_ind]
                                z = data[fast_search_ind[2]][start_ind:end_ind]
                                for s in selection:
                                    new_particle = particle(**partcl_opt)
                                    if getattr(s, 'id', None) is not None:
                                        q = data[6][start_ind:end_ind]
                                        mask = new_particle.filter_by_id(s, q=q)
                                    else:
                                        mask = new_particle.filter_by_range(s, x=x, y=y, z=z)
                                    # mask = new_particle.filter_by_range(s, x=chunk[0], y=chunk[1], z=chunk[2])
                                    if mask is None or np.all(mask == False) or np.size(mask) == 0:
                                        print_progress(verbose, ns, start_ind, end_ind, new_particle.N)
                                        del new_particle, mask
                                        continue
                                    # print(mask)
                                    # mask = np.where(mask).tolist()
                                    # mask = np.nonzero(mask)
                                    # print(mask)
                                    if every_nth is not None:
                                        chunk = [data[i][start_ind:end_ind][mask][::every_nth] for i in range(len(keys))]
                                    else:
                                        chunk = [data[i][start_ind:end_ind][mask] for i in range(len(keys))]
                                    # chunk = data[:][start_ind:end_ind]
                                    # chunk = chunk[:,mask]


                                    # chunk = np.squeeze(data[:][start_ind:end_ind])
                                    # print(f'len chunk = {len(chunk)}')
                                    # print(chunk)
                                    # if len(chunk) == 0:
                                        # continue
                                    # print(f'shape of chunk: {chunk.shape}')
                                    # chunk = chunk[mask]
                                    # chunk = np.squeeze(chunk).astype(dtype)
                                    new_particle = particle(*chunk, **partcl_opt)
                                    # print(new_particle)
                                    new_particles = particleContainer(**container_opt)
                                    new_particles.__dict__[particle_data.names[ns]] = new_particle
                                    if incr_write:
                                        new_particles.write_to_file(file=write_filename, selection=s)
                                    else:
                                        particle_data += new_particles
                                    print_progress(verbose, ns, start_ind, end_ind, new_particle.N)
                                    del new_particle, new_particles, chunk

                            else:
                                # chunk = [data[i][start_ind:end_ind] for i in range(len(keys))]
                                if every_nth is not None:
                                    chunk = data[:][start_ind:end_ind]
                                else:
                                    chunk = data[:][start_ind:end_ind]
                                chunk = np.squeeze(chunk).astype(dtype)
                                for s in selection:
                                    # print(f'****************************************************')
                                    new_particle = particle(*chunk, **partcl_opt)
                                    # print(new_particle)
                                    new_particle.filter_by_range(s, inplace=True)
                                    # print(f's.center_phys = {s.center_phys} | s.delta_phys = {s.delta_phys}')
                                    # print(new_particle)
                                    new_particles = particleContainer(**container_opt)
                                    new_particles.__dict__[particle_data.names[ns]] = new_particle
                                    if incr_write:
                                        new_particles.write_to_file(file=write_filename, selection=s)
                                    else:
                                        particle_data += new_particles
                                    print_progress(verbose, ns, start_ind, end_ind, new_particle.N)
                                    del new_particle, new_particles
                elif not incr_read:
                    for ns in ns_avail:
                        if (None not in species and ns not in species):
                            continue
                        # print(f'Loading particle {ns} data (all).')
                        partcl_opt = dict(name  = particle_data.names[ns],
                                          coord = attributes["coord"])
                        for s in selection:
                            if fast_search and s is not None:
                                data = [dataset.get(f"{key}_{ns}") for key in keys]
                                x = data[fast_search_ind[0]]
                                y = data[fast_search_ind[1]]
                                z = data[fast_search_ind[2]]
                                new_particle = particle(**partcl_opt)
                                mask = new_particle.filter_by_range(s, x=x, y=y, z=z)
                                chunk = [data[i][mask] for i in range(len(keys))]
                                chunk = np.squeeze(chunk).astype(dtype)
                                new_particle = particle(*chunk, **partcl_opt)
                            else:
                                data = np.asarray([dataset.get(f"{key}_{ns}")[:] for key in keys])
                                data = np.squeeze(data).astype(dtype)
                                new_particle = particle(*data, **partcl_opt)
                                new_particle.filter_by_range(s, inplace=True)
                            particle_data.__dict__[particle_data.names[ns]] += new_particle
                            del new_particle, data
                return particle_data, attributes
            else:
                test_data = dataset.get(keys[0])
                if test_data is None:
                    return None, None
                read_zero_blocks = False
                if len(test_data) == 1:
                    read_zero_blocks = True
                del test_data

                if read_zero_blocks:
                    data = np.asarray([dataset.get(f"{key}/0") for key in keys]).astype(dtype)
                    # data = np.squeeze(data)
                else:
                    data = np.asarray([dataset[key] for key in keys]).astype(dtype)
        except KeyError:
            print(f'KeyError: {keys}')
            return None, None
        # save data to numpy file
        # np.save(f'test_data_{keys[0]}.npy', data.astype('f4'))
    return data, attributes

def test_hdf5_io():
    Bx = np.random.rand(321, 131, 461)
    By = np.random.rand(321, 131, 461)
    Bz = np.random.rand(321, 131, 461)
    file_out = h5py.File('/Users/leo/DATA/test.h5', "w")
    file_out.create_dataset('Bx', data=Bx, dtype='f4')
    file_out.create_dataset('By', data=By, dtype='f4')
    file_out.create_dataset('Bz', data=Bz, dtype='f4')
    file_out.attrs['name'] = 'Ring_mar23LG'
    file_out.attrs['comment'] = ''
    file_out.attrs['cycle'] = 130000
    file_out.attrs['size_cell'] = sim.size_cell
    file_out.attrs['min_phys'] = sim.min_phys
    file_out.attrs['max_phys'] = sim.max_phys
    file_out.attrs['size_code'] = sim.size_code
    file_out.close()

    filename = f"/Users/leo/DATA/test.h5"
    key = 'Bx'
    with h5py.File(filename, 'r') as f:
        dataset = f
        dataset_names = list(f)
        print(f'attributes = {list(dataset.attrs)}')
        attributes = list(dataset.attrs)
        if 'Step#0' in dataset_names:
            dataset = f.get('Step#0')
            dataset_names = list(dataset)
        if 'Block' in list(dataset_names):
            dataset = dataset.get('Block')
        data = dataset.get(key)
        if len(list(data)) == 1 and list(data)[0] == '0':
            data = data.get('0')
        # print(f'shape of {key}: {data.shape}')

def load_fields(selection, keys=['Bx', 'By', 'Bz'], cut=True):
    """ Load fields from a file and return the data and attributes """
    cycle = selection.cycle
    write_cycle = selection.sim.cycle_write_fields
    # interpolate = False if cycle % write_cycle == 0 else selection.interpolate
    interpolate = False

    local_field = False
    field_sliced = False

    def test_dirs(dirs, basename):
        for dir in dirs:
            path = os.path.join(dir, basename)
            if os.path.isfile(path):
                return path
        return None

    path = selection.path_fields(cycle)
    fixed_path = getattr(selection, 'filename_fields_fixed', None)
    path = path if fixed_path is None else fixed_path
    basename = os.path.basename(path)

    dirs = [selection.data_dir, selection.sim.local_data_dir, selection.data_external_dir]
    path = test_dirs(dirs, basename)

    cut_axis = selection.cut_axis_code
    cut_index = int(selection.center_cell[cut_axis])
    cut_axis_name = ['X', 'Y', 'Z'][cut_axis]
    cutname = f'{basename.split(".")[0]}_{cut_axis_name}{cut_index}.h5'
    if test_dirs(dirs, cutname) is not None and cut == True:
        path = test_dirs(dirs, cutname)
        field_sliced = True

    magnetic_field_file = f'{basename.split(".")[0]}_B3D.h5'
    if test_dirs(dirs, magnetic_field_file) is not None and all([key in ['Bx', 'By', 'Bz'] for key in keys]) and cut == False:
        local_field = True
        path = test_dirs(dirs, magnetic_field_file)

    if not interpolate:
        if not os.path.isfile(path):
            print(f'File not found: {path}. Aborting.')
            return None, None
        print(f'Loading fields from {path}.')
        field, attr = load_hdf5_array(path, keys=keys)
    else:  # --- linear interpolate between the fields at different cycles
        print(f'Interpolating fields between cycles.')
        cycle_prev = selection.cycle // write_cycle * write_cycle
        cycle_next = ((cycle_prev + write_cycle) // write_cycle) * write_cycle
        path_prev = selection.path_fields(cycle_prev)
        path_next = selection.path_fields(cycle_next)
        if not os.path.isfile(path_prev) or not os.path.isfile(path_next):
            print(f'File not found: {path_prev} or {path_next}. Aborting.')
            return None, None
        field_prev, attr = load_hdf5_array(path_prev, keys=keys)
        field_next, attr = load_hdf5_array(path_next, keys=keys)
        frac = (selection.cycle - cycle_prev) / write_cycle
        field = field_prev + (field_next - field_prev) * frac
    F = []
    for f in field:
        if cut:
            F.append(cut_field(f, selection, transpose=not field_sliced))
        else:
            F.append(f)
    return F, attr

def parse_config_file(file_path):
    """Parse a configuration file and return a dictionary with the
    key-value pairs."""
    config_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):  # Ignore empty lines and comments
                key_value = line.split('=')  # Split key and value by "="
                key = key_value[0].strip()    # Get the key (first word before the "=")
                values_with_comments = key_value[1].split('#')  # Split values and comments by "#"
                # values = values_with_comments[0].strip() if values_with_comments else ''  # Get values before the comment
                values = values_with_comments[0].split() if values_with_comments else ''  # Get values before the comment
                try:
                    values = [float(value) if ('.' in value) or
                              ('E' in value) or
                              ('e' in value)
                              else int(value) for value in values]
                except ValueError:
                    pass
                if len(values) == 1:
                    values = values[0]
                config_data[key] = values
    return config_data

def save_config_file(file_path, config_dict):
    """ Save a configuration file from a dictionary with the
    key-value pairs."""
    with open(file_path, 'w') as file:
        for key, value in config_dict.items():
            if isinstance(value, list):
                value = ' '.join([str(item) for item in value])
            file.write(f'{key} = {value}\n')

if __name__ == "__main__":
    # file_path = "config_file.txt"  # Replace this with the actual path of your configuration file
    file_path = "Earth3DKHR_Gianni.inp"  # Replace this with the actual path of your configuration file
    config_dict = parse_config_file(file_path)

    # print dict keys
    print(config_dict.keys())
    print(config_dict["SimulationName"])

    # Print the configuration file
    save_config_file("config_file.txt", config_dict)
    print(config_dict)
