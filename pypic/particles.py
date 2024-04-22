import numpy as np
from pypic.core import *
from pypic.geometry import *
from pypic.linalg import *
from pypic.interpolate import cubic_interpolation
from pypic.interpolate import cubic_interpolation1D
from pypic.fields import mask_from_range

class particle:
    """ A class to hold particle data """
    def __init__(self, X=None, Y=None, Z=None, U=None, V=None, W=None, Q=None,
                 name='particle', coord='code'):
        self.name = name
        if X is None:
            R = np.vstack(([], [], []))
        else:
            R = np.vstack((X, Y, Z))

        if U is None:
            V = np.vstack(([], [], []))
        else:
            V = np.vstack((U, V, W))

        if Q is None:
            Q = []
        else:
            Q = Q
        self.r = R
        self.v = V
        self.q = Q
        self.speed = None
        self.coord = coord
    @property
    def N(self):
        return len(self.q)
    # def __repr__(self):
    #     return f'{self.name}'
    def __str__(self):
        if self.N > 0:
            avg_r = np.average(self.r, axis=-1)
        else:
            avg_r = None
        return f'particle = {self.name} | N = {self.N:,d} | r(avg) = {avg_r} | coord = {self.coord}'
    def __len__(self):
        return len(self.q)
    def __getitem__(self, index):
        self.r = self.r[:,index]
        self.v = self.v[:,index]
        self.q = self.q[index]
        if self.speed is not None:
            self.speed = self.speed[index]
        return self
    def list_particles(self):
        for x in zip(*self.r, *self.v, self.q):
            yield particle(*x, name=self.name, coord=self.coord)
    def __iter__(self):
        return iter(self.list_particles())
    def __add__(self, other):
        if self.coord != other.coord and self.N > 0:
            raise ValueError('Cannot add particles with different coordinate systems')
        elif self.N == 0 and self.coord != other.coord:
            self.coord = other.coord
        # self.r = np.hstack((self.r, other.r))
        # self.v = np.hstack((self.v, other.v))
        # self.q = np.hstack((self.q, other.q))
        self.r = np.concatenate((self.r, other.r), axis=1)
        self.v = np.concatenate((self.v, other.v), axis=1)
        self.q = np.concatenate((self.q, other.q), axis=0)
        return self
        # return particle(*np.hstack((self.r, other.r)),
        #                 *np.hstack((self.v, other.v)),
        #                 np.hstack((self.q, other.q)),
        #                 name=self.name, coord=self.coord)
    def calculate_speed(self):
        self.speed = np.linalg.norm(self.v, axis=0)
    def to_phys(self, sim):
        if self.N > 0 and self.coord != 'phys':
            self.r = code_to_phys(self.r, sim.size_code, sim.min_phys, sim.max_phys)
            self.v = code_to_phys_rot(self.v, True)
        self.coord = 'phys'
    def check_unique(self):
        if self.N == 0:
            return None
        unique = np.unique(self.q)
        print(f'Total q = {len(self.q)} | Unique = {len(unique)} | Ratio = {len(unique)/len(self.q):.2f}')
        # select = self.filter_by_id(self, q=self.q)
        # filter particles by unique id
        # mask = np.isin(self.q, unique)
        # mask = np.array(self.q == unique)
        # self = self[mask]
        # unique = np.unique(self.q)
        # print(f'Total q = {len(self.q)} | Unique = {len(unique)} | Ratio = {len(unique)/len(self.q):.2f}')
        # print(f'-----------------')


        # print(f'Unique q = {len(unique)}')
        # exit()
        return np.unique(self.q)

    def filter_by_id(self, selection, inplace=True, q=None, mask=None):
        """ Select particles from a data array by min and max in code coords """
        if selection is None:
            # print(f'selection is None')
            return self
        if q is not None:
            # mask = np.array(q == selection.id)
            mask = np.isin(q, selection.id)
            return mask
        else:
            mask = np.array(self.q == selection.id)
        if inplace:
            self = self[mask]
            return self
        else:
            return particle(*self.r[:,mask], *self.v[:,mask], self.q[mask],
                            name=self.name, coord=self.coord)
            # return copy.deepcopy(self[mask])
    def filter_by_range(self, selection, inplace=True,
                        x=None, y=None, z=None,
                        mask=None):
        """ Select particles from a data array by min and max in code coords """
        if selection is None:
            # print(f'selection is None')
            # return self
            return np.full(len(self), True)
        if self.coord == 'code':
            smin = selection.min_code
            smax = selection.max_code
        elif self.coord == 'phys':
            smin = selection.min_phys
            smax = selection.max_phys
        else:
            print(f'Coords are not recognized')
            return None
        if mask is None:
            if x is not None and y is not None and z is not None:
                mask_x = mask_from_range(x, smin[0], smax[0])
                # if np.all(mask_x == False):
                #     return np.full(len(x), False)
                mask_y = mask_from_range(y, smin[1], smax[1])
                # if np.all(mask_y == False):
                #     return np.full(len(y), False)
                mask_z = mask_from_range(z, smin[2], smax[2])
                mask = np.logical_and(mask_x, mask_y) # Combine x and y masks
                mask = np.logical_and(mask,   mask_z) # Combine with the z mask
                # mask = np.intersect1d(mask_x, mask_y) # Combine x and y masks
                # mask = np.intersect1d(mask,   mask_z) # Combine with the z mask
                # print(f'x and y and z are not none')
                # print(mask)
                return mask
            else:
                mask_x = mask_from_range(self.r[0], smin[0], smax[0])
                # if np.all(mask_x == False):
                #     return np.full(len(self.r[0]), False)
                mask_y = mask_from_range(self.r[1], smin[1], smax[1])
                # if np.all(mask_y == False):
                #     return np.full(len(self.r[1]), False)
                mask_z = mask_from_range(self.r[2], smin[2], smax[2])
                mask = np.logical_and(mask_x, mask_y) # Combine x and y masks
                mask = np.logical_and(mask,   mask_z) # Combine with the z mask
        if inplace:
            # print(mask)
            self = self[mask]
            return self
        else:
            return particle(*self.r[:,mask], *self.v[:,mask], self.q[mask],
                            name=self.name, coord=self.coord)
            # return copy.deepcopy(self[mask])

    def average_movement(self):
        """ Average the movement of the particles in the selection box. """
        self.v_avg = np.mean(self.v, axis=-1)
        self.speed_avg = np.linalg.norm(self.v_avg)
        self.v_unit = r_unit_vector(self.v_avg)
        self.r_avg = np.mean(self.r, axis=-1)

class particleContainer:
    """ A class to hold all particles """
    def __init__(self,
                 name            = None,
                 electrons       = None,
                 ions            = None,
                 electrons_kappa = None,
                 ions_kappa      = None,
                 coord           = 'code',
                 track           = None, # track specified particle species
                 ):
        self.name = name
        if electrons is None:
            self.electrons = particle(name='electrons')
        else:
            self.electrons = electrons
        if ions is None:
            self.ions = particle(name='ions')
        else:
            self.ions = ions
        if electrons_kappa is None:
            self.electrons_kappa = particle(name='electrons_kappa')
        else:
            self.electrons_kappa = electrons_kappa
        if ions_kappa is None:
            self.ions_kappa = particle(name='ions_kappa')
        else:
            self.ions_kappa = ions_kappa
        self.coord = coord
        self.track = track
    @property
    def N(self):
        return [p.N for p in self.particles]

    def __iter__(self):
        return iter([self.electrons, self.ions, self.electrons_kappa, self.ions_kappa])

    @property
    def names(self):
        return [p.name for p in self]

    def __add__(self, other):
        # for p, o in zip(self, other):
        #     p += o
        # return self
        keys = self.names
        for key in keys:
            # print(f'adding key = {key} | self = {self.__dict__[key].N} | other = {other.__dict__[key].N}')
            self.__dict__[key] += other.__dict__[key]
        return self

    def check_unique(self):
        keys = self.names
        for key in keys:
            self.__dict__[key].check_unique()
        return self

    def __repr__(self):
        info = ('*'*5 +f' {self.name} ' +'*'*5+'\n'
                      +f'-------------------------------------------------\n'
                      +f'    electrons:       {self.electrons.N:,d}\n'
                      +f'    ions:            {self.ions.N:,d}\n'
                      +f'    electrons_kappa: {self.electrons_kappa.N:,d}\n'
                      +f'    ions_kappa:      {self.ions_kappa.N:,d}\n'
                      +f'-------------------------------------------------\n')
        if self.electrons.N > 0:
               info += f'    min electron pos: {np.min(self.electrons.r, axis=-1)}\n'
               info += f'    max electron pos: {np.max(self.electrons.r, axis=-1)}\n'
               info += f'-------------------------------------------------\n'
        if self.ions.N > 0:
               info += f'    min ion pos: {np.min(self.ions.r, axis=-1)}\n'
               info += f'    max ion pos: {np.max(self.ions.r, axis=-1)}\n'
        return info
    def calculate_speed(self):
        for p in self:
            if p.N > 0:
                p.calculate_speed()
    def to_phys(self, sim):
        self.coord = 'phys'
        for p in self:
            if p.N > 0:
                p.to_phys(sim)
    def average_movement(self):
        for p in self:
            if p.N > 0:
                p.average_movement()
    def filter_by_range(self, selection, inplace=True):
        nc = particleContainer(name=self.name, coord=self.coord)
        for p in self:
            if p.N > 0:
                nc.__dict__[p.name] = p.filter_by_range(selection, inplace=inplace)
        return nc

    def to_array(self):
        # return numpy array of all particles
        A = np.array([])
        for p in self:
            if p.N > 0:
                ind = self.names.index(p.name)
                A = np.vstack((p.r, p.v, p.q))
        return A

    def write_to_file(self,
                      file=None,
                      selection=None,
                      dir=None,
                      append=True,
                      dtype=None,
                      compression=None, # 'gzip'
                      ):

        if file is None:
            if selection is not None:
                file = selection.selection_filenames["particles"]
            else:
                file = 'dataset.f5'
        if dir is None:
            if selection is not None:
                dir = selection.output_dir
        file = os.path.join(dir, file)
        if dtype is None:
            if selection is not None:
                dtype = selection.dtype
            else:
                dtype = 'f4'

        if append is False:
            f = h5py.File(file, 'w')
            file_keys = list(f.keys())
            for p in self:
                if p.N > 0:
                    ind = self.names.index(p.name)
                    print(f'Writing to a new file. Particle {ind}.')
                    f.create_dataset(f'x_{ind}', data=p.r[0], compression=compression, chunks=True, maxshape=(None,), dtype=dtype)
                    f.create_dataset(f'y_{ind}', data=p.r[1], compression=compression, chunks=True, maxshape=(None,), dtype=dtype)
                    f.create_dataset(f'z_{ind}', data=p.r[2], compression=compression, chunks=True, maxshape=(None,), dtype=dtype)
                    f.create_dataset(f'u_{ind}', data=p.v[0], compression=compression, chunks=True, maxshape=(None,), dtype=dtype)
                    f.create_dataset(f'v_{ind}', data=p.v[1], compression=compression, chunks=True, maxshape=(None,), dtype=dtype)
                    f.create_dataset(f'w_{ind}', data=p.v[2], compression=compression, chunks=True, maxshape=(None,), dtype=dtype)
                    f.create_dataset(f'q_{ind}', data=p.q,    compression=compression, chunks=True, maxshape=(None,), dtype=dtype)
            if self.name is not None:
                f.attrs['name'] = self.name
            f.attrs['coord'] = self.coord
            if self.track is not None:
                f.attrs['track'] = self.track
            f.close()
        elif append is True:
            f = h5py.File(file, 'a')
            file_keys = list(f.keys())
            for p in self:
                file_keys = list(f.keys())
                if p.N > 0:
                    ind = self.names.index(p.name)
                    if f'x_{ind}' in file_keys:
                        # print(f'Appending to a file. Appending to Particle {ind}.')
                        n_new = p.r.shape[-1]
                        n_old = f[f'x_{ind}'].shape[0]
                        # print(f'  reshaping . . . old: {n_old}, new: {n_new}')
                        f[f'x_{ind}'].resize((n_old + n_new), axis=0)
                        f[f'x_{ind}'][-n_new:] = p.r[0]
                        f[f'y_{ind}'].resize((n_old + n_new), axis=0)
                        f[f'y_{ind}'][-n_new:] = p.r[1]
                        f[f'z_{ind}'].resize((n_old + n_new), axis=0)
                        f[f'z_{ind}'][-n_new:] = p.r[2]
                        f[f'u_{ind}'].resize((n_old + n_new), axis=0)
                        f[f'u_{ind}'][-n_new:] = p.v[0]
                        f[f'v_{ind}'].resize((n_old + n_new), axis=0)
                        f[f'v_{ind}'][-n_new:] = p.v[1]
                        f[f'w_{ind}'].resize((n_old + n_new), axis=0)
                        f[f'w_{ind}'][-n_new:] = p.v[2]
                        f[f'q_{ind}'].resize((n_old + n_new), axis=0)
                        f[f'q_{ind}'][-n_new:] = p.q
                    else:
                        # print(f'Appending to a file. Adding a new particle {ind} to {list(f.keys())}')
                        f.create_dataset(f'x_{ind}', data=p.r[0], compression=compression, chunks=True, maxshape=(None,), dtype=dtype)
                        f.create_dataset(f'y_{ind}', data=p.r[1], compression=compression, chunks=True, maxshape=(None,), dtype=dtype)
                        f.create_dataset(f'z_{ind}', data=p.r[2], compression=compression, chunks=True, maxshape=(None,), dtype=dtype)
                        f.create_dataset(f'u_{ind}', data=p.v[0], compression=compression, chunks=True, maxshape=(None,), dtype=dtype)
                        f.create_dataset(f'v_{ind}', data=p.v[1], compression=compression, chunks=True, maxshape=(None,), dtype=dtype)
                        f.create_dataset(f'w_{ind}', data=p.v[2], compression=compression, chunks=True, maxshape=(None,), dtype=dtype)
                        f.create_dataset(f'q_{ind}', data=p.q,    compression=compression, chunks=True, maxshape=(None,), dtype=dtype)
            if self.name is not None:
                f.attrs['name'] = self.name
            f.attrs['coord'] = self.coord
            if self.track is not None:
                f.attrs['track'] = self.track
            f.close()
        return

def particle_data_stats(data, verbose=False):
    """ returns a dictionary of the min and max values of the particle data """
    N = np.size(data[0])
    pos_min = np.asarray([data[0].min(), data[1].min(), data[2].min()])
    pos_max = np.asarray([data[0].max(), data[1].max(), data[2].max()])
    vel_min = np.asarray([data[3].min(), data[4].min(), data[5].min()])
    vel_max = np.asarray([data[3].max(), data[4].max(), data[5].max()])
    q_range = np.asarray([data[6].min(), data[6].max()])
    abs_pos_max = np.max([np.abs(pos_min), np.abs(pos_max)], axis=0)
    abs_vel_max = np.max([np.abs(vel_min), np.abs(vel_max)], axis=0)
    median_pos = np.median(data[0:3], axis=1)
    median_vel = np.median(data[3:6], axis=1)
    median_q = np.median(data[6])
    data_stats =  dict(N=N, pos_min=pos_min, pos_max=pos_max,
                       vel_min=vel_min, vel_max=vel_max,
                       q_range=q_range, abs_pos_max=abs_pos_max,
                       abs_vel_max=abs_vel_max, median_pos=median_pos,
                       median_vel=median_vel, median_q=median_q)
    if verbose:
        np.set_printoptions(formatter={'float': '{:.5f}'.format})
        print(f'================= Particle Stats =================\n'
             +f'    Particle Number = {N:,}\n'
             +f'    min(pos) [di][code] = {data_stats["pos_min"]}\n'
             +f'    max(pos) [di][code] = {data_stats["pos_max"]}\n'
             +f'    min(v) [c][code] = {data_stats["vel_min"]}\n'
             +f'    max(v) [c][code] = {data_stats["vel_max"]}\n'
             +f'    Q range = {data_stats["q_range"]}\n'
             +f'    max (abs(pos)) [di][code] = {data_stats["abs_pos_max"]}\n'
             +f'    max (abs(vel)) [di][code] = {data_stats["abs_vel_max"]}\n'
             +f'    median pos [di][code] = {data_stats["median_pos"]}\n'
             +f'    median v [di][code] = {data_stats["median_vel"]}\n'
             +f'    median Q = {data_stats["median_q"]}\n'
             +f'==================================================\n')
    return data_stats

def combine_kappa_particles(data, selection, sim, transform_to_phys=False):
    ns = selection.ns
    data0 = select_particles_by_range(data, selection, sim, ns=ns, transform_to_phys=transform_to_phys)
    data1 = select_particles_by_range(data, selection, sim, ns=ns+2, transform_to_phys=transform_to_phys)
    data_selected = np.concatenate((data0, data1), axis=-1)
    return data_selected

def select_velocities(data, display,
                      vaxes_choice=0,
                      axis0=0, axis1=1,
                      x_label=None, y_label=None,
                      transform_to_phys=False):

    # data_stats = particle_data_stats(data_selected, verbose=True)
    position = np.asarray(data[0:3])
    velocity = np.asarray(data[3:6])
    velocity_perp = np.sqrt(velocity[0]**2+velocity[1]**2)
    velocity = np.concatenate((velocity, velocity_perp[np.newaxis,:]), axis=0)
    # weight = np.abs(np.asarray(data[6]))/(1.6022e-19)
    weight = np.abs(np.asarray(data[6]))

    if display is not None:
        x_label = display["vaxes_choices"][vaxes_choice][0]
        y_label = display["vaxes_choices"][vaxes_choice][1]
        axis0 = display["vaxes_dict"][x_label]
        axis1 = display["vaxes_dict"][y_label]
    v0 = velocity[axis0,:]
    v1 = velocity[axis1,:]
    return v0, v1, weight, [x_label, y_label]

def draw_Bperp_circle(ax, df, selection, spacing=1,
                      color='yellow', alpha=0.1, lw=1, ls='-',
                      mean_radius=True, center_EM=False, num_points=20):
    """ Draw a circle for gyroradius in the plane perpendicular to the
    magnetic field. """
    N = len(df['x'])
    for i in range(N)[::spacing]:
        r = np.array([df['x'].iloc[i], df['y'].iloc[i], df['z'].iloc[i]])
        B = np.array([df['Bx'].iloc[i], df['By'].iloc[i], df['Bz'].iloc[i]])

        if mean_radius:
            dp = df['dc'].mean()
        else:
            dp = df['dc'].iloc[i]

        # Rescale the lengths to physical units
        dp *= selection.sim.scaling[0]

        # Default to current position
        center = r

        # Change the center if guiding center is found
        if df['x_gc'].iloc[i] is not np.nan:
            center = np.array([df['x_gc'].iloc[i],
                               df['y_gc'].iloc[i],
                               df['z_gc'].iloc[i]])

        # Change the center if position from EM forces is found
        if center_EM:
            center = np.array([df['x_EM'].iloc[i],
                               df['y_EM'].iloc[i],
                               df['z_EM'].iloc[i]])

        special = df['special'].iloc[i]
        # if special == 1:
            # color = 'red'

        draw_3d_circle(ax, B, radius=dp, center=center, num_points=num_points,
                       draw=True, ls=ls, lw=lw, color=color, alpha=alpha)

        ax.scatter(*center, color=color, s=10, alpha=alpha)

def find_guiding_center(df, selection):
    """ Find guiding center coordinates for a particle trajectory. """
    sim = selection.sim
    # Create a new column for guiding center coordinates
    df['x_gc'] = np.nan
    df['y_gc'] = np.nan
    df['z_gc'] = np.nan
    df['x_EM'] = np.nan
    df['y_EM'] = np.nan
    df['z_EM'] = np.nan
    df['v_perp'] = np.nan
    df['v_par'] = np.nan
    df['E_perp'] = np.nan
    df['E_par'] = np.nan
    df['B_perp'] = np.nan
    df['B_par'] = np.nan
    df['E'] = np.nan
    df['vxB'] = np.nan
    df['dc'] = np.nan
    df['Tc'] = np.nan
    df['special'] = 0
    dTc = 0 # Cumulative integral of cyclotron period along trajectory (fraction)
    R_gc = None

    # Iterate over all points in the dataframe
    for i in range(len(df['x'])):
        r = np.array([df['x'].iloc[i], df['y'].iloc[i], df['z'].iloc[i]])
        v = np.array([df['vx'].iloc[i], df['vy'].iloc[i], df['vz'].iloc[i]])
        B = np.array([df['Bx'].iloc[i], df['By'].iloc[i], df['Bz'].iloc[i]])
        E = np.array([df['Ex'].iloc[i], df['Ey'].iloc[i], df['Ez'].iloc[i]])
        cycle = df['cycle'].iloc[i]
        # Number of cycles since last saved cycle
        dcycle = df['cycle'].iloc[i] - df['cycle'].iloc[i-1] if i > 0 else 0
        df.loc[i, 'special'] = 0

        B_unit = B/np.linalg.norm(B)
        E_unit = E/np.linalg.norm(E)
        vxB = np.cross(v, B)
        vxB_unit = vxB/np.linalg.norm(vxB)
        v_par = np.dot(v, B_unit)
        v_perp = np.sqrt(np.linalg.norm(v)**2 - v_par**2)
        v_unit = v/np.linalg.norm(v)


        Bmag = np.linalg.norm(B)
        B_par = np.dot(B, v_unit)
        B_perp = np.sqrt(np.linalg.norm(B)**2 - B_par**2)
        E_par = np.dot(E, v_unit)
        E_perp = np.sqrt(np.linalg.norm(E)**2 - E_par**2)

        # Define a phi azimuthal position vector based on x and y
        # phi = np.array([r[1], -r[0], 0])
        # phi_unit = phi/np.linalg.norm(phi)
        # alpha = np.cross(B_par, phi_unit)
        # alpha_unit = alpha/np.linalg.norm(alpha)

        # Electric and magnetic forces
        f_E = selection.sim.e * E * 1e-3
        f_B = selection.sim.e * vxB * 1e3 * 1e-9
        a_E = f_E/selection.sim.me
        a_B = f_E/selection.sim.me

        # Guiding center position shift due to E field
        da = 0.5 * (a_E+a_B) * (dcycle*selection.sim.code_time)**2
        ds = da / selection.sim.unit_phys * selection.sim.scaling[0]
        ds_mag = np.linalg.norm(ds)
        r_EM = r + (vxB_unit * ds_mag)

        # Cyclotron Radius
        if selection.species in [0,2]:
            dc = sim.f_dce(Bmag*1e-9, v_perp*1e3)
            wc = sim.f_wce(Bmag*1e-9)
        else:
            dc = sim.f_dcp(Bmag*1e-9, v_perp*1e3)
            wc = sim.f_wcp(Bmag*1e-9)

        # Cyclotron radius in physical units
        dc = dc / selection.sim.unit_phys * selection.sim.code_space
        # Cyclotron period in code cycles
        Tc = (2.*np.pi/wc)/sim.code_time # period in cycles (<code_time> = s/cycle)
        # Fractional increase in cyclotron period integral since last cycle
        dTc += dcycle/Tc

        df.loc[i, 'dc'] = dc
        df.loc[i, 'Tc'] = Tc
        df.loc[i, 'v_perp'] = v_perp
        df.loc[i, 'v_par'] = v_par
        df.loc[i, 'E_perp'] = E_perp
        df.loc[i, 'E_par'] = E_par
        df.loc[i, 'B_perp'] = B_perp
        df.loc[i, 'E'] = np.linalg.norm(E)
        df.loc[i, 'vxB'] = np.linalg.norm(vxB)
        df.loc[i, 'x_EM'] = r_EM[0]
        df.loc[i, 'y_EM'] = r_EM[1]
        df.loc[i, 'z_EM'] = r_EM[2]

        # The particle has completed a cyclotron period
        if dTc > 1:
            # Assume no previous guiding center position
            x_gc_prev, y_gc_prev, z_gc_prev = None, None, None

            # Check for a previous guiding center position
            df_prev = df[df['cycle'] <= cycle]
            df_gc_prev = df_prev[df_prev['x_gc'].notna()]
            if len(df_gc_prev) > 0:
                x_gc_prev = df_gc_prev['x_gc'].iloc[-1]
                y_gc_prev = df_gc_prev['y_gc'].iloc[-1]
                z_gc_prev = df_gc_prev['z_gc'].iloc[-1]

            # Select the previous positions without a guiding center
            df_calc = df_prev[df_prev['x_gc'].isna()]
            if len(df_calc) > 0:
                # If there is no previous guiding center, use the first pos
                if x_gc_prev is None:
                    x_gc_prev = df_calc['x'].iloc[0]
                    y_gc_prev = df_calc['y'].iloc[0]
                    z_gc_prev = df_calc['z'].iloc[0]

                # Number of cycles since the last cyclotrong period
                dcycle_calc = (df_calc['cycle'].iloc[-1]
                              -df_calc['cycle'].iloc[0])

                # Average position of the particle during the cyclotron period
                x_gc = np.average(df_calc['x'])
                y_gc = np.average(df_calc['y'])
                z_gc = np.average(df_calc['z'])
                r_gc = np.array([x_gc, y_gc, z_gc])

                # Collect all guiding center positions
                R_gc = r_gc if R_gc is None else np.vstack((R_gc, r_gc))

                # Interpolate between the last two guiding centers
                for d in df_calc.index:
                    dt = df.loc[d, 'cycle'] - df_calc['cycle'].iloc[0]
                    t_ratio = dt / dcycle_calc
                    # Linear interpolation
                    df.loc[d, 'x_gc'] = x_gc_prev + (x_gc - x_gc_prev)*t_ratio
                    df.loc[d, 'y_gc'] = y_gc_prev + (y_gc - y_gc_prev)*t_ratio
                    df.loc[d, 'z_gc'] = z_gc_prev + (z_gc - z_gc_prev)*t_ratio

            # Reset the cyclotron period integral
            dTc = np.mod(dTc, 1)

            df.loc[i, 'special'] = 1
        else:
            df.loc[i, 'special'] = 0

    # center = r + vxB * dp/2
    return df, R_gc

def add_particle_dash(ax, sim, selection, df, key='energy',
                      edgecolor='white', lw=1, color='orange',
                      text_color='white', alpha=0.8,
                      fontsize=12):
    cycle = selection.cycle
    # species = df['species'].iloc[0]
    species = selection.species

    if key == 'jE':
        J = np.array([df['Jx'], df['Jy'], df['Jz']])
        E = np.array([df['Ex'], df['Ey'], df['Ez']])
        jdotE = np.sum(J*E, axis=0)
        # jdotE = np.dot(J, E.T)
        from scipy.ndimage import gaussian_filter1d
        # jdotE = smooth_fields(jdotE, 1)
        jdotE = gaussian_filter1d(jdotE, 1)
        # normalize jdotE to 1
        jdotE /= np.max(jdotE)


        color = 'grey'
        ax.plot(df['cycle'], jdotE, color='white', lw=1, ls='--', alpha=0.4)
        ax.plot(df['cycle'][df['cycle']<=cycle] , jdotE[df['cycle']<=cycle], color=color, lw=2, ls='-', alpha=0.9)
        ax.set_ylabel(r'$j \cdot E$', color=text_color, fontsize=fontsize)
        ax.set_ylim(-0.9,0.9)
        # draw a dashed line at y=0
        ax.axhline(y=0, color='white', lw=1, ls='--', alpha=0.6)
    else:


        ax.fill_between(df['time'], df['q1'], df['q3'],
                        color='gray', alpha=0.01)
        # ax.fill_between(df['time'][df['time']<=cycle], df['q1'][df['time']<=cycle], df['q3'][df['time']<=cycle],
                        # color='grey', alpha=0.1)
        ax.plot(df['time'], df['mean'], color='white', lw=1, ls='--', alpha=0.4)
        ax.plot(df['time'][df['time']<=cycle] , df['mean'][df['time']<=cycle], color=color, lw=2, ls='-', alpha=0.9)
        # plot fill area between df['q1'] and df['q3']
        ax.scatter(df['time'][df['time']==cycle] , df['mean'][df['time']==cycle], color=color, s=20, alpha=0.9)

        ax.set_ylim(20, 50)
        ax.set_ylim(1, 70)
        if species == 0:
            ax.set_ylim(5,150)

        ax.set_ylabel('Mean Ion Energy [keV]', color=text_color, fontsize=fontsize)
        if species == 0:
            ax.set_ylabel('Electron Energy [keV]', color=text_color, fontsize=fontsize)
    # ax.set_ylim(40, 110)
    # cycles_all = df['cycle'].to_numpy()
    # energies_all = df['energy'].to_numpy()
    # Z_all = df['z'].to_numpy()
    # df = df[df['cycle'] <= cycle]
    # cycles = df['cycle'].to_numpy()
    # energies = df['energy'].to_numpy()
    # Z = df['z'].to_numpy()
    # Bx = df['Bx'].to_numpy()
    # By = df['By'].to_numpy()
    # Bz = df['Bz'].to_numpy()
    # B = np.sqrt(Bx**2 + By**2 + Bz**2)
    # # Bmag = abs(B[-1]*1e-9) * np.sqrt(4*np.pi)
    # # Bmag = abs(B[-1]*1e-9) * 4*np.pi
    # Bmag = abs(B[-1]*1e-9)
    # wcp = sim.f_wcp(Bmag)
    # # Tcp = (2.*np.pi/wcp)/sim.dt_phys * np.sqrt(4*np.pi) # w = 2pi/T
    # Tcp = (2.*np.pi/wcp)/sim.dt_phys * sim.scaling[0] # w = 2pi/T
    # ax.set_title(r'$T_{cp}$ = '+f'{Tcp/1000:.1f}'+ r' k$\text{cycles}$'+
    #              # r' ($\omega_{cp}=$' + f'{}'
    #              r' ($B=$' + f'{Bmag*1e9:.1f}'+ r' nT)',
    #              pad=0, color=color, fontsize=16)
    # # print(f'Tcp = {Tcp}')
    # ax.axvline(x=cycles[-1]-Tcp/2, color=color, lw=1, ls='-', alpha=0.8)
    # ax.axvline(x=cycles[-1]+Tcp/2, color=color, lw=1, ls='-', alpha=0.8)
    # ax.plot(cycles_all, Z_all, color='white', lw=1, ls='--', alpha=0.4)
    # # ax.plot(cycles_all, energies_all, color='white', lw=1, ls='--', alpha=0.4)
    # ax.plot(cycles, Z, color=color, lw=lw, alpha=alpha)
    # # ax.plot(cycles, energies, color=color, lw=lw, alpha=alpha)
    # # ax.scatter(cycles[-1], energies[-1], color=color, s=20, alpha=alpha)
    # ax.scatter(cycles[-1], Z[-1], color=color, s=20, alpha=alpha)
    ax.set_xlim(sim.cycle_limits[0], sim.cycle_limits[1])

    ax.set_xlabel('Cycle', color=text_color, fontsize=fontsize)
    # ax.set_ylabel('$z_{GSM}$')
    # ax.set_ylabel('$z_{GSM}$')
    # disable axis edge
    ax.spines['bottom'].set_color(edgecolor)
    ax.spines['top'].set_color(edgecolor)
    ax.spines['right'].set_color(edgecolor)
    ax.spines['left'].set_color(edgecolor)
    for spine in ['top', 'right', 'left', 'right']:
        ax.spines[spine].set_visible(False)
    # disable ticks
    ax.tick_params(axis='x', colors=text_color, labelsize=fontsize)
    ax.tick_params(axis='y', colors=text_color, labelsize=fontsize)
    ax.yaxis.label.set_color(text_color)
    ax.xaxis.label.set_color(text_color)
    ax.set_frame_on(False)
    from matplotlib import ticker
    def my_formatter_fun(x, p):
        return "%dk" % (x/1000)
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(my_formatter_fun))
    # ax.tick_params(axis='both', which='both', length=0)
    # ax.grid(True, alpha=0.1)
    # disable axis border
    # ax.set_axisbelow(False)

def add_particle_projection(ax, selection, df,
                      edgecolor='white', lw=1, color='orange',
                      text_color='white', alpha=0.8,
                      fontsize=12):
    cycle = selection.cycle
    cycles_all = df['cycle'].to_numpy()
    sim = selection.sim
    energies_all = df['energy'].to_numpy()
    ax.plot(df['y'], df['z'], color='white', lw=1, ls='--', alpha=0.3)
    df = df[df['cycle'] <= cycle]
    cycles = df['cycle'].to_numpy()
    energies = df['energy'].to_numpy()
    X = df['x'].to_numpy()
    Y = df['y'].to_numpy()
    Z = df['z'].to_numpy()
    ax.plot(Y, Z, color=color, lw=lw, ls='--', alpha=0.5)
    # ax.plot(cycles, energies, color=color, lw=lw, alpha=alpha)
    # print(df)
    # print columns of df
    # print(df.columns)
    # exit()
    vx = df['vx'].to_numpy()
    vy = df['vy'].to_numpy()
    vz = df['vz'].to_numpy()
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    Bx = df['Bx'].to_numpy()
    By = df['By'].to_numpy()
    Bz = df['Bz'].to_numpy()
    B = np.sqrt(Bx**2 + By**2 + Bz**2)

    # Bmag = abs(B[-1]*1e-9) * np.sqrt(4*np.pi)
    # Bmag = abs(B[-1]*1e-9) * 4*np.pi
    Bmag = abs(B[-1]*1e-9)
    vmag = v[-1]

    # dp = sim.f_dcp(Bmag)/sim.unit_phys * sim.dt* sim.scaling[0]
    # dp = sim.f_dcp(Bmag)/sim.unit_phys * sim.dt

    # dp = vmag*1e3/sim.f_wcp(Bmag)/sim.unit_phys *sim.scaling[0]
    dp = vmag*1e3/sim.f_wcp(Bmag)/sim.unit_phys * sim.code_space

    print(f'dp = {dp}')
    r = dp
    rc = draw_circle(ax, [Y[-1],Z[-1]], r, n=100, draw=True, color=color, ls='-', lw=2)

    ax.set_title(r'$d_{cp}$ = '+f'{dp:.2f}'+ r' $R_E$', pad=0, color=color, fontsize=16)
    ax.scatter(Y[-1], Z[-1], color=color, s=20, alpha=alpha)
    ax.set_xlim(5, 2)
    ax.set_ylim(-3.5, -0.5)
    ax.set_xlabel('$y_{GSM}$', color=text_color, fontsize=fontsize)
    ax.set_ylabel('$z_{GSM}$', color=text_color, fontsize=fontsize)
    # disable axis edge
    ax.spines['bottom'].set_color(edgecolor)
    ax.spines['top'].set_color(edgecolor)
    ax.spines['right'].set_color(edgecolor)
    ax.spines['left'].set_color(edgecolor)
    for spine in ['top', 'right', 'left', 'right']:
        ax.spines[spine].set_visible(False)
    # disable ticks
    ax.tick_params(axis='x', colors=text_color, labelsize=fontsize)
    ax.tick_params(axis='y', colors=text_color, labelsize=fontsize)
    ax.yaxis.label.set_color(text_color)
    ax.xaxis.label.set_color(text_color)
    ax.set_aspect('equal')
    # ax.set_frame_on(False)

# def draw_plane_intersections(ax, points):
#     """ Draw a circle for an intersection of a set of points with a plane """
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     for seg in segments:
#         pt1 = np.array(seg[0])
#         pt2 = np.array(seg[1])
#         if pt1[2]*pt2[2] < 0:
#             pt = np.average([pt1, pt2], axis=0)
#             draw_3d_circle(ax, (0,0,1), radius=0.15, center=pt, color='white',
#                            alpha=0.6, lw=1, clip_on=True)


def interpolate_trajectory(df, t=None, step=500):
    """Interpolate the trajectory of a particle."""
    points = np.array([df['x'], df['y'], df['z']]).T
    # t = np.linspace(0, 1, len(points))
    # t_vals = np.linspace(0, 1, 100, endpoint=True)
    t_points = df['cycle']
    if t is None:
        t = np.arange(df['cycle'].iloc[0], df['cycle'].iloc[-1]+step, step=step)
    interpolated_points = cubic_interpolation(points, t_points, t)
    # Cubic Interpolation with velocity Tangents (not working)
    # dt = df['cycle'].diff().mean()*selection.sim.dt_phys
    # df['vx'] =  df['vx']/1e-3/selection.sim.unit_phys*dt
    # tangents = np.array([df['vx'], df['vy'], df['vz']]).T
    # interpolated_points_tangent = cubic_interpolation(points, t_points, t, tangents)
    interpolated_vx = cubic_interpolation1D(df['vx'], t_points, t)
    interpolated_vy = cubic_interpolation1D(df['vy'], t_points, t)
    interpolated_vz = cubic_interpolation1D(df['vz'], t_points, t)
    df_interp = pd.DataFrame({'cycle': t,
                              'species': df['species'].iloc[0],
                              'x': interpolated_points.T[0],
                              'y': interpolated_points.T[1],
                              'z': interpolated_points.T[2],
                              'vx': interpolated_vx,
                              'vy': interpolated_vy,
                              'vz': interpolated_vz,
                              'q': df['q'].iloc[0],
                              'interpolated': 1,
                              })
    df_interp.loc[df_interp['cycle'].isin(df['cycle']), 'interpolated'] = 0

    plot_interpolation = False
    if plot_interpolation:
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        # ax.plot(t_points, df['vx'], 'y-o', ms=4, alpha=0.7, label='vx')
        # ax.plot(t, interpolated_vx, 'y-o', ms=2, alpha=0.7, label='vx interpolated')
        # ax.plot(t_points, df['vy'], 'g-o', ms=4, alpha=0.7, label='vy')
        # ax.plot(t, interpolated_vy, 'g--o', ms=2, alpha=0.7, label='vy interpolated')
        # ax.plot(t_points, df['vz'], 'b-o', ms=4, alpha=0.7, label='vz')
        # ax.plot(t, interpolated_vz, 'b--o', ms=2, alpha=0.7, label='vz interpolated')
        ax.plot(t_points, points.T[0], 'r-o', ms=4, alpha=0.7, label='x')
        ax.plot(t_points, points.T[1], 'g-o', ms=4, alpha=0.7, label='y')
        ax.plot(t_points, points.T[2], 'b-o', ms=4, alpha=0.7, label='z')
        ax.plot(t, interpolated_points.T[0], 'r-o', ms=2, alpha=0.7, label='x interpolated')
        ax.plot(t, interpolated_points.T[1], 'g-o', ms=2, alpha=0.7, label='y interpolated')
        ax.plot(t, interpolated_points.T[2], 'b-o', ms=2, alpha=0.7, label='z interpolated')
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(*points.T, 'ro', ms=2, alpha=0.7, label='Original Points')
        # ax.plot(*interpolated_points.T, 'b-o', ms=1, label='Cubic Spline')
        # ax.plot(*interpolated_points_tangent.T, 'g-o', ms=2, alpha=0.5, label='Cubic Spline with Tangents')
        # ax.set_xlabel('X axis')
        # ax.set_ylabel('Y axis')
        # ax.set_zlabel('Z axis')
        ax.legend()
        # ax.set_aspect('equal')
        # ax.xaxis.pane.fill = False
        # ax.yaxis.pane.fill = False
        # ax.zaxis.pane.fill = False
        # plt.title("3D Cubic Spline Interpolation")
        plt.show()
        exit()
    return df_interp

def calculate_derivate_quantities(df, selection=None):
    """ Calculate derived quantities such as speed, energy, pitch angle, etc. """
    species = df['species'].iloc[0]
    v = np.array([df['vx'], df['vy'], df['vz']])
    r = np.array([df['x'], df['y'], df['z']])
    df['speed'] = selection.sim.f_speed(v)
    df['energy'] = selection.sim.f_energy(v, species)
    df['angle'] = selection.sim.f_angle(v)
    df['r'] = np.linalg.norm(r, axis=0)
    # if 'Bx' in df.columns:
    #     B = np.array([df['Bx'], df['By'], df['Bz']])
    #     df['pitch_angle'] = selection.sim.f_pitch_angle(v, B)

    # df['Jx'] = -df['Jx']
    # df['Ex'] = -df['Ex']
    # df['Bx'] = -df['Bx']
    # df['vx'] =  df['vx']*selection.sim.c_phys*1e-3
    # df['vy'] =  df['vy']*selection.sim.c_phys*1e-3
    # df['vz'] =  df['vz']*selection.sim.c_phys*1e-3
    R = np.array([df['x'], df['y'], df['z']]).T
    v = np.array([df['vx'], df['vy'], df['vz']]).T
    j = np.array([df['Jx'], df['Jy'], df['Jz']]).T
    E = np.array([df['Ex'], df['Ey'], df['Ez']]).T
    B = np.array([df['Bx'], df['By'], df['Bz']]).T
    # j_unit = j/np.linalg.norm(j, axis=1)[:, np.newaxis]
    v_unit = v/np.linalg.norm(v, axis=1)[:, np.newaxis]
    j_unit = j/np.linalg.norm(j, axis=1)[:, np.newaxis]
    E_unit = E/np.linalg.norm(E, axis=1)[:, np.newaxis]
    B_unit = B/np.linalg.norm(B, axis=1)[:, np.newaxis]
    jdotE = np.sum(j*E, axis=1)
    Edotv = np.sum(E*v_unit, axis=1)
    EdotB = np.sum(E*B_unit, axis=1)
    vdotB = np.sum(v_unit*B_unit, axis=1)
    vdotE = np.sum(v_unit*E_unit, axis=1)
    angle_to_B = np.arccos(vdotB)
    angle_to_B = np.degrees(angle_to_B)
    angle_to_B = np.abs(angle_to_B-90)
    angle_to_E = np.arccos(vdotE)
    angle_to_E = np.degrees(angle_to_E)
    angle_to_E = np.abs(angle_to_E-90)
    df['jdotE'] = jdotE
    df['Edotv'] = Edotv
    df['EdotB'] = EdotB
    df['vdotB'] = vdotB
    df['angle_to_B'] = angle_to_B
    df['angle_to_E'] = angle_to_E
    df['E'] = np.linalg.norm(E, axis=1)
    df['B'] = np.linalg.norm(B, axis=1)
    # smooth out Edotv using a rolling window
    df['Edotv'] = df['Edotv'].rolling(window=5, center=True, min_periods=1, win_type='hamming').mean()
    df['EdotB'] = df['EdotB'].rolling(window=5, center=True, min_periods=1, win_type='hamming').mean()
    df['Ex'] = df['Ex'].rolling(window=5, center=True, min_periods=1, win_type='hamming').mean()
    df['Ey'] = df['Ey'].rolling(window=5, center=True, min_periods=1, win_type='hamming').mean()
    df['Ez'] = df['Ez'].rolling(window=5, center=True, min_periods=1, win_type='hamming').mean()
    df['x_roll'] = df['x'].rolling(window=150, center=True, min_periods=1, win_type='hamming').mean()
    df['y_roll'] = df['y'].rolling(window=150, center=True, min_periods=1, win_type='hamming').mean()
    df['z_roll'] = df['z'].rolling(window=150, center=True, min_periods=1, win_type='hamming').mean()
    df['time'] = df['cycle']*selection.sim.dt_phys
    # df = df.drop_duplicates(keep='last')
    # df.to_hdf(f'df_S{selection.species}_{cycle:06d}.h5', key='Block', mode='w', index=False)
    return df
