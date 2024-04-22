import os
import sys
import h5py
import numpy as np
import glob
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.measure import block_reduce
from scipy import constants
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if path not in sys.path:
    sys.path.append(path)

particle_data_keys = ('x', 'y', 'z', 'u', 'v', 'w', 'q')

field_data_keys = ('Bx', 'By', 'Bz', 'EFx_0', 'EFx_1', 'EFy_0', 'EFy_1',
              'EFz_0', 'EFz_1', 'Ex', 'Ey', 'Ez', 'Jx_0', 'Jx_1',
              'Jy_0', 'Jy_1', 'Jz_0', 'Jz_1', 'N_0', 'N_1',
              'Pxx_0', 'Pxx_1', 'Pxy_0', 'Pxy_1', 'Pxz_0', 'Pxz_1',
              'Pyy_0', 'Pyy_1', 'Pyz_0', 'Pyz_1', 'Pzz_0', 'Pzz_1',
              'Qrem_0', 'Qrem_1', 'Vfx', 'Vfy', 'Vfz', 'divB', 'rho_0', 'rho_1')

def cell_centers(min, max, size, dx=None, coord='code'):
    """ Return the cell centers """
    dx = np.divide(np.subtract(max,min), size) if dx is None else dx
    if np.size(size) == 1:
        return np.linspace(min+dx/2, max-dx/2, size, endpoint=True)
    else:
        c = [cell_centers(min[i], max[i], size[i], dx[i]) for i in range(np.size(size))]
        if coord == 'phys':
            c[0] = c[0][::-1] # reverse order of x
        return c
    # return np.linspace(min, max, size, endpoint=True) + (max-min)/size/2
# -------------------------- Coordinate transformations -----------------------
def code_to_phys_rot(R, xref=False):
    """ Rotate code coords to physical coords (no rescaling)"""
    if np.size(R) == 3:
        R2 = np.asarray([R[0], R[2], R[1]])
    else:

        # print(f'shape of R: {np.shape(R)}')
        # print(f'shape of R[0]: {np.shape(R[0])}')
        # print(f'shape of R[1]: {np.shape(R[1])}')
        # print(f'shape of R[2]: {np.shape(R[2])}')
        # TODO: transpose doesn't work?
        # R2 = np.transpose(R, (0, 2, 1))
        # R2  np.transpose(R, (2, 0, 1))

        # print(f'shape of R2: {np.shape(R2)}')
        # R2 = np.transpose(R, [0, 2, 1])
        # R2 = np.transpose(R, (2, 0, 1))
        # R2 = np.asarray([R[2], R[0], R[1]])
        R2 = np.asarray([R[0], R[2], R[1]])
    if xref:
        R2[0] *= -1
    return R2

def phys_to_code_rot(R):
    """ Rotate physical coords to code coords (no rescaling)"""
    return code_to_phys_rot(R)

def code_to_phys(R, size_code, min_phys, max_phys,
                 vector=False, in_axes='code'):
    """ Convert code coords to physical coords """
    if in_axes == 'phys':
        R = phys_to_code_rot(R)
    size_phys = np.abs(np.subtract(max_phys, min_phys))
    if not vector:
        R2 = np.zeros_like(R)
        R2[0] = max_phys[0] - R[0]/size_code[0]*size_phys[0]
        R2[1] = min_phys[1] + R[2]/size_code[2]*size_phys[1]
        R2[2] = min_phys[2] + R[1]/size_code[1]*size_phys[2]
    else:
        phys_in_code_units = np.divide(size_code,
                                      phys_to_code_rot(size_phys))
        R2 = np.divide(code_to_phys_rot(R),
                       code_to_phys_rot(phys_in_code_units))
    return np.asarray(R2)

def code_to_cell(R, size_code, size_cell,
                 small_step=True, check=True):
    """ Convert code coords to cell coords """
    R2 = np.zeros_like(R)
    R = add_small_step(R) if small_step else R
    R2[0] = np.floor(R[0]/size_code[0]*size_cell[0])
    R2[1] = np.floor(R[1]/size_code[1]*size_cell[1])
    R2[2] = np.floor(R[2]/size_code[2]*size_cell[2])
    if check:
        R2 = check_cells(R2, size_cell)
    return np.asarray(R2)

def phys_to_code(R, size_code, min_phys, max_phys,
                 vector=False, small_step=False, check=False,
                 out_axes='code'):
    """ Convert physical coords to code coords """
    R = add_small_step(R) if small_step else R
    size_phys = np.abs(np.subtract(max_phys, min_phys))
    if not vector:
        R2 = np.zeros_like(R)
        R2[0] = -(R[0] - max_phys[0])*size_code[0]/size_phys[0]
        R2[1] =  (R[2] - min_phys[2])*size_code[1]/size_phys[2]
        R2[2] =  (R[1] - min_phys[1])*size_code[2]/size_phys[1]
    else:
        phys_in_code_units = np.divide(size_code,
                                      phys_to_code_rot(size_phys))
        R2 = np.multiply(phys_to_code_rot(R),
                           phys_in_code_units)

    if check: # check if cells are within bounds
        R2 = check_cells(R2, size_code)
    if out_axes == 'phys':
        R2 = code_to_phys_rot(R2)
    return np.asarray(R2)

def phys_to_cell(R, size_cell, min_phys, max_phys,
                 vector=False, out_axes='code',
                 check=True, small_step=True):
    """ Convert physical coords to cell coords """
    return phys_to_code(R, size_cell, min_phys, max_phys,
                        vector=vector, small_step=small_step, check=check,
                        out_axes=out_axes)

def cell_to_phys(R, size_cell, min_phys, max_phys,
                  in_axes='code', vector=False):
    """ Convert cell coords to physical coords """
    return code_to_phys(R, size_cell, min_phys, max_phys,
                        in_axes=in_axes, vector=vector)

def cell_to_code(R, size_cell, size_code):
    """ Convert cell coords to code coords """
    return code_to_cell(R, size_cell, size_code)

def check_cells(R, size_cell, min=[0,0,0], dtype=np.int16):
    """ Round cells coords to integers and check bounds """
    R = np.clip(np.asarray(R, dtype=np.float32), min, size_cell)
    return R.astype(dtype)

def add_small_step(R, step=1e-6, dtype=np.float32):
    """ Add a small step to the coordinates to avoid rounding errors """
    return np.asarray(R, dtype=dtype) + step
# -----------------------------------------------------------------------------
def sort_selection(U, V):
    """ Sort the selection box coords so that U is the min and V is the max """
    u,v = U[0], V[0]
    if u > v:
        U[0], V[0] = v, u
    return np.asarray(U), np.asarray(V)
# -----------------------------------------------------------------------------

def unit_prefix(key):
    """ Get prefix character or factor for a given number or unit string. """
    prefix_dict = {
                   # 'da': 1e1, # deca
                   # 'h':  1e2, # hecto
                   'k':  1e3, # kilo
                   'M':  1e6, # mega
                   'G':  1e9, # giga
                   'T':  1e12, # tera
                   'P':  1e15, # peta
                   # 'd':  1e-1, # deci
                   'c':  1e-2, # centi
                   'm':  1e-3, # milli
                   'u':  1e-6, # micro
                   'n':  1e-9, # nano
                   'p':  1e-12, # pico
                   'f':  1e-15, # femto
                   }
    # if key is a string, return factor
    if isinstance(key, str):
        if key in prefix_dict:
            return prefix_dict[key]
        else:
            return 1
    # if key is a number, return prefix character
    else:
        key = f'{key:0.0e}' # convert to scientific notation with one s.d.
        key = '1' + key[1:] # change the first number to 1
        key = float(key)
        prefix = '' # default
        for k, v in prefix_dict.items():
            if key == v:
                prefix = k
        return prefix

# -----------------------------------------------------------------------------

class ipic3D:
    name = "iPIC3D"
    def __init__(self,
                 unit_name_phys='Re',
                 unit_name_code='dp',
                 size_cell=None,
                 size_code=None,
                 min_phys=None,
                 max_phys=None,
                 center_phys=None,
                 name="MHDUCLA",
                 ):
        # --- GRID SIZE
        self.min_phys = [-33, -16, -6.5] if min_phys is None else min_phys
        self.max_phys = [ 13,  16,  6.5] if max_phys is None else max_phys
        self.size_cell = [460, 130, 320] if size_cell is None else size_cell
        # self.size_cell = [461, 131, 321] if size_cell is None else size_cell
        self.size_cell_phys = code_to_phys_rot(self.size_cell)
        self.size_phys = np.abs(np.subtract(self.max_phys, self.min_phys))
        self.size_code = [184, 52, 128] if size_code is None else size_code
        self.center_phys = [0,0,0] if center_phys is None else center_phys
        self.center_code = self.phys_to_code(self.center_phys)
        self.center_cell = self.phys_to_cell(self.center_phys)

        # --- GRID SPACING
        # Physical units per cell [phys units/cell][code coords]
        self.cell_in_phys_units = np.divide(phys_to_code_rot(self.size_phys),
                                            self.size_cell)
        # Codes units per cell [code units/cell][code coords]
        self.cell_in_code_units = np.divide(self.size_code,
                                            self.size_cell)
        # Number of code units in one phys unit [code coords]
        self.phys_in_code_units = np.divide(self.size_code,
                                            phys_to_code_rot(self.size_phys))

        # --- GRID UNITS
        self.unit_name_phys = unit_name_phys
        self.unit_name_code = unit_name_code
        # self.unit_phys = 6371.2e3 # Re [m]
        self.unit_phys = 6378.1e3 # m # Re [m] Astropy (https://docs.astropy.org/en/stable/units/)

        # --- PARTICLES
        self.n_per_cell_xyz = [5, 5, 5]
        self.n_per_cell = 5
        self.v_thermal = [0.0225, 0.00315, 0.1, 0.01] # thermal velocity
        self.v_drift = [0.0, 0.0, 0.0, 0.0] # drift velocity
        self.distribution = [3, 3, 2, 2] # 0: maxwellian, 1: monoenergetic 2:kappa, 3:
        self.kappa = [3, 3, 2, 2] # kappa index (kappa distribution)
        self.rho_init = [1, 1, 0.01, 0.01] # initial density
        self.rho_inject = [1, 1, 0.01, 0.01] # density of injected particles

        # --- BOUNDARIES
        self.periodic = [False, False, False]
        self.BC_phi = [[1,1], [1,1], [1,1]] # 0,1: Dirichlet, 2: Neumann
        self.BC_E = [[0,0], [3,3], [3,3]] # 0: perfect conductor, 1: magnetic mirror
        self.BC_particle = [[2,2], [2,2], [2,2]] # 0: exit, 1: mirror, 2: reemission
        self.L_square = 13.5 # Earth
        self.L_outer = 2 # Damping at the boundary
        self.MPI_domains = [20, 10, 10]

        # --- INPUT/OUTPUT
        self.cycle_write_fields = 1000
        self.cycle_write_particles = 2000
        self.cycle_restart = 2000
        self.cycle_limits = [1, 202500]
        self.local_data_dir = 'DATA'

        # --- REFERENCE VALUES
        # self.dipM = -7.96e15 # Tm^3
        # self.dipM = 8e22 # Am^2
        self.dipM = -31000 # nT (Earth's dipole moment)
        # seld.dipM = 6.4×1021 Am2.
        self.B0 = -31200 # nT (Earth's field at magnetic equator)
        self.n_ref = 0.25*1e6 # 1 for Mercury, 0.25 for Earth
        self.B_ref = 20e-9 # T
        self.Ti_ref = 3.5e3 # eV

        # --- PHYSICS CONSTANTS (SI)
        self.mrcode = 256 # mass ratio
        self.qom = -256 # -256, -1836 #1, -256, 1
        self.qoms = [self.qom, 1, self.qom, 1]
        self.vmax = 1.0 #.1 #.02, .2, .1
        self.phys_el = 0 # physical electron (yes/no)
        self.mu_factor = 1.
        self.eps_factor = 1.
        self.e = constants.elementary_charge
        self.mu0 = constants.mu_0/self.mu_factor**2 # permeability of free space
        self.eps0 = constants.epsilon_0/self.eps_factor**2 # permittivity of free space
        self.c_phys = 1./np.sqrt(self.mu0*self.eps0)
        self.kB = constants.Boltzmann
        if self.phys_el == 1:
            self.me = constants.electron_mass
            self.mp = self.me * self.mrcode # scale proton mass
        else:
            self.mp = constants.proton_mass
            self.me = self.mp / self.mrcode # scale electron mass
        self.np = self.n_ref # proton density
        self.ne = self.n_ref # electron density (quasi-neutrality)

        # --- PLASMA PARAMETERS
        self.wpp = self.f_wpp(self.np) # ref ion plasma frequency
        self.wpe = self.f_wpe(self.ne) # ref electron plasma frequency
        self.wce = self.f_wce(self.B_ref) # ref electron cyclotron frequency
        self.wcp = self.f_wcp(self.B_ref) # ref proton cyclotron frequency
        self.dp = self.f_dp(self.np) # ref proton inertial length
        self.de = self.f_de(self.ne) # ref electron inertial length
        self.dce = self.f_dce(self.B_ref) # ref electron gyroradius
        self.dcp = self.f_dcp(self.B_ref) # ref proton gyroradius

        # --- TIME
        self.dt = 0.1 # time step
        # self.wppDT = self.dt
        self.dt_phys = self.dt/self.wpp # time step in physical units

        # --- SCALING
        # dx = self.cell_in_phys_units[0] * self.unit_phys
        # de = sim.f_de(density*1e6)/sim.unit_phys
        # scaling = sim.dp/sim.unit_phys
        self.phys_in_code_units_phys = self.unit_phys/self.dp
        self.scaling = self.phys_in_code_units_phys/self.phys_in_code_units
        self.dt_phys *= self.scaling[0]

        # --- CONVERSION FACTORS (CODE -> SI)
        # self.code_space = self.c_phys/self.wpp*self.scaling[0]
        # self.code_time = self.dt/self.wpp*self.scaling[0]
        self.code_space = self.c_phys/self.wpp
        self.code_time = self.dt/self.wpp
        self.code_v = self.c_phys
        self.code_n = self.n_ref*(4*np.pi)
        # self.code_n = self.n_ref
        self.code_B = self.mp * self.wpp/self.e
        # self.code_B = self.mp * self.wcp/self.e
        # self.code_B = self.mp * self.wpp/self.e * np.sqrt(4*np.pi)
        self.code_E = self.c_phys * self.code_B
        # self.code_j = self.mp * self.wpp/self.e/self.mu0/self.dp * (4*np.pi)
        self.code_j = self.code_n * self.e * self.code_v / (4*np.pi)
        self.code_force = self.e * self.code_E # E.g., F=qE
        self.code_force2 = self.e * self.code_v * self.code_B # E.g., F=qvB
        self.code_forceden = self.code_j * self.code_B
        self.code_forceden2 = self.code_n * self.code_force # E.g., rhoE or jxB
        self.code_forceden3 = self.code_n * self.e * self.code_E # E.g., rhoE or jxB
        # self.code_forceden2 = self.code_force
        self.code_T = self.c_phys**2 * self.mp  # Assuming ions as ref
        self.code_energy = self.code_T * self.kB
        self.code_energy2 = self.code_j * self.code_E * self.code_space**2
        self.code_energy3 = self.c_phys**2 * self.mp  # Assuming ions as ref
        self.code_energy4 = self.mp*self.code_n*self.code_v**3 * self.code_space**3

        self.code_S = self.code_energy/self.code_T # Entropy (J/K)
        self.code_eflux = self.mp*self.code_n*self.code_v**3 # Enth flux, ExB
        self.code_eflux_e = self.me*self.code_n*self.code_v**3
        self.code_eflux2 = self.code_j*self.code_E*self.code_space
        # self.code_eflux2 = self.code_j*self.code_E*self.code_space / np.sqrt(4*np.pi)
        # self.code_eflux3 = self.code_E * self.code_B / self.mu0 * (4*np.pi)
        self.code_eflux3 = self.code_E * self.code_B / self.mu0
        self.code_eden = self.code_j * self.code_E # E field dissipation, j.E
        self.code_P = self.code_n*self.code_T
        # --- PARTICLE MASS CORRECTION
        # self.momentum_corrector = np.sqrt(self.mp/self.me/self.mrcode)
        # self.momentum_corrector = 1.
        self.momentum_corrector = [self.mrcode/1836., 1., self.mrcode/1836., 1.]

        # --- OTHER
        self.r_crit = self.cell_in_phys_units[0]*self.unit_phys # cell size in meters
        # critical density to resolve electron inertial length (n>size_cell unresolved)
        self.n_crit = self.eps0*self.me * (self.c_phys**2/(self.r_crit**2*self.e**2))

        # --- TIME FOR PARTICLES TO TRAVERSE THE GRID
        self.cross_dist_phys = 20 # (20 Re)
        self.cross_dist_code = self.phys_in_code_units[0] * self.cross_dist_phys
        self.vi_avg = 0.0025 * self.code_v # average ion velocity
        self.t_cross_phys = self.cross_dist_phys*self.unit_phys/self.vi_avg
        self.t_cross_cycles = self.t_cross_phys/self.dt_phys

        self.bins_phys = cell_centers(self.min_phys, self.max_phys, self.size_cell_phys, coord='phys')
        self.bins_code = cell_centers([0,0,0], self.size_code, self.size_cell, coord='code')

        # --- GRID CODE UNITS
        if self.unit_name_code == 'dp':
            self.unit_code = self.dp
        elif self.unit_name_code == 'de':
            self.unit_code = self.de
        elif self.unit_name_code == 'dcp':
            self.unit_code = self.dcp
        elif self.unit_name_code == 'dce':
            self.unit_code = self.dce
        else:
            self.unit_code = 1
        self.name = name

    # de = sim.f_de(density*4*np.pi*sim.scaling**3)/sim.unit_phys/0.1
    def mass(self, species=1):
        if species in [0, 2]:
            return self.me
        else:
            return self.mp
    def f_wpe(self, n_e):
        """ Calculate the electron plasma frequency (in Hz) """
        return np.sqrt(n_e*self.e**2/(self.eps0*self.me))
    def f_wpp(self, n_p):
        """ Calculate the ion plasma frequency (in Hz) """
        return np.sqrt(n_p*self.e**2/(self.eps0*self.mp))
    def f_wce(self, B):
        """ Calculate the electron cyclotron frequency (in Hz) """
        return self.e*B/self.me
    def f_wcp(self, B):
        """ Calculate the proton cyclotron frequency (in Hz) """
        return self.e*B/self.mp
    def f_dp(self, np, v=None):
        """ Calculate the proton inertial length (in meters) """
        v = self.c_phys if v is None else v
        return v/self.f_wpp(np)
    def f_de(self, ne, v=None):
        """ Calculate the electron inertial length (in meters) """
        v = self.c_phys if v is None else v
        return v/self.f_wpe(ne)
    def f_dce(self, B, v=None):
        """ Calculate the electron gyroradius (in meters) """
        v = self.c_phys if v is None else v
        return v/self.f_wce(B)
    def f_dcp(self, B, v=None):
        """ Calculate the proton gyroradius (in meters) """
        v = self.c_phys if v is None else v
        return v/self.f_wcp(B)
    def f_thermal_speed(self, T, species=1):
        """ Calculate the thermal speed of a particle (in km/s) """
        return np.sqrt(2*T/self.mass(species)) / 1000.
    def f_speed(self, v):
        """ Calculate the speed of a particle (in km/s) """
        return np.linalg.norm(v, axis=0) * self.c_phys / 1000.
    def f_energy(self, v, species=1):
        """ Calculate the kinetic energy of a particle (in keV) """
        return self.mass(species) * (self.f_speed(v)*1000)**2 / 2. / self.e / 1000
    def f_angle(self, v):
        """ Calculate the angle between the velocity and the x-y plane """
        return np.arctan2(v[2], np.sqrt(v[0]**2 + v[1]**2)) * 180. / np.pi
    def f_pitch_angle(self, v, B):
        """ Calculate the pitch angle between the velocity and the magnetic field """
        return np.arccos(np.dot(v, B) / (np.linalg.norm(v, axis=0) * np.linalg.norm(B,axis=0))) * 180. / np.pi

    def get_units(self, key, species=1, coord='phys'):
        """ Return the units and scaling factor for a given key """
        if key is None or len(key) == 0:
            return '', 1.
        key = key.lower() # make sure key is lowercase
        key = key.replace('_', '') # remove underscores

        if coord == 'code':
            return '', 1.
        if key in ['x', 'y', 'z', 'r', 'space']:
            return r'km$', self.code_space/unit_prefix('k')
        if key in ['xphys', 'yphys', 'zphys', 'rphys']:
            return self.unit_name_phys, 1.
        if key in ['xcode', 'ycode', 'zcode', 'rcode']:
            return self.unit_name_code, 1.
        if key in ['r', 'theta', 'phi']:
            return '', 1.
        elif key in ['t', 'time']:
            return r's', self.code_time
        elif key in ['cycle']:
            return r'cycle', 1.
        elif key in ['n']:
            return r'cell$^{-1}$', 1.
        elif key in ['rho', 'density']:
            return r'cm$^{-3}$', self.code_n/unit_prefix('M')
        elif key in ['v', 'velocity']:
            return r'km/s', self.code_v/unit_prefix('k')
        elif key in ['j', 'current']:
            return r'nA/m$^2$', self.code_j/unit_prefix('n')
        elif key in ['b', 'magneticfield', 'bfield']:
            return r'nT', self.code_B/unit_prefix('n')
        elif key in ['e', 'electricfield', 'efield']:
            return r'mV/m', self.code_E / unit_prefix('m')
        elif key in ['ef', 'energyflux', 'eflux']:
            if species in [0,2]:
                return r'mW/m$^2$', self.code_eflux_e/unit_prefix('m')
            elif species in [1,3]:
                return r'mW/m$^2$', self.code_eflux/unit_prefix('m')
        elif key in ['eden', 'energyden', 'energydensity', 'ed']:
                return r'mW/m$^3$', self.code_eden/unit_prefix('m')
        elif key in ['energy']:
            return r'keV', self.code_energy/unit_prefix('k')
        elif key in ['p', 'pressure']:
            return r'nPa', self.code_P/unit_prefix('n')
        elif key in ['f', 'force']:
            return r'N', self.code_force
        elif key in ['fden', 'forcedensity', 'forceden']:
            return r'N/m$3$', self.code_forceden
        elif key in ['s', 'entropy']:
            return r'J/K$', self.code_S
        else:
            return '', 1.
    def get_scale(self, key, species=1, coord='phys'):
        return self.get_units(key, species=species, coord=coord)[1]
    def get_uname(self, key, species=1, coord='phys'):
        return self.get_units(key, species=species, coord=coord)[0]

    def __repr__(self):
        up = self.unit_name_phys
        uc = self.unit_name_code

        def num_to_str(n, width=6, prec=2, center=True):
            """ Convert a number to a string """
            def center_int(s, prec, center):
                if center:
                    return f"{int(s):,}"+' '*(prec+1)
                else:
                    return f"{int(s):,}"
            if isinstance(n, int):
                return f"{center_int(n, prec, center):>{width}}"
            if isinstance(n, float):
                if n.is_integer():
                    return f"{center_int(n, prec, center):>{width}}"
                else:
                    if abs(n) < 0.01:
                        # return f"{n:>{width+1}.{prec+1}f}"
                        return f"{n:>{width}.{prec}e}"
                    else:
                        return f"{n:>{width},.{prec}f}"
            return f"{n:>{width}}"

        def list_to_str(l, width=6, prec=2, center=False, sep=' '):
            """ Convert a list of numbers to a string """
            # strs = ', '.join(map(num_to_str, l))
            sep = sep + ' '
            strs = sep.join(map((lambda x: num_to_str(x, width, prec, center)), l))
            # return f"{strs:>{width}}"
            return f"[{strs}]"

        def two_columns(a, b, units='', lcol=30, rcol=15,
                        prec=2, center=False, sep=' ', newline=True):
            """ Print two columns with a given width """
            if isinstance(b, str):
                bstr = b
            elif np.size(b) == 1:
                bstr = num_to_str(b, width=rcol, prec=prec, center=center)
            else:
                bstr = list_to_str(b, prec=prec, center=False, sep=sep)
            if newline:
                units_str = f" {units}\n"
            else:
                units_str = f" {units}"
            return (f"{a:>{lcol}} = " + f"{bstr:>{rcol}}" + units_str)

        lcol = 25
        rcol = 15
        return (
                 f"{self.name:=^72}" + "\n"
                + two_columns('Grid Size [cell]', self.size_cell, f'#')
                + two_columns('Grid Size [code]', self.size_code, f'{uc}')
                + two_columns('Grid Size [phys]', self.size_phys, f'{up}')
                + two_columns('Grid Min [phys]', self.min_phys, f'{up}')
                + two_columns('Grid Max [phys]', self.max_phys, f'{up}')
                + two_columns('Center [phys]', self.center_phys, f'{up}')
                + two_columns('Center [code]', self.center_code, f'{uc}')
                + two_columns('Cell in phys units [code]', self.cell_in_phys_units, f'{up}/cell')
                + two_columns('Phys in code units [code]', self.phys_in_code_units, f'{uc}/{up}')
                +f"{'PHYSICS CONSTANTS':-^72}" + "\n"
                + two_columns('B_ref', self.B_ref*1e9, 'nT')
                + two_columns('n_ref', self.n_ref*1e-6, 'cm-3')
                + two_columns('Ti_ref', self.Ti_ref, 'eV')
                + two_columns('mrcode', self.mrcode, 'q/m')
                + two_columns('eps_factor', self.eps_factor, '')
                + two_columns('mu_factor', self.mu_factor, '')
                + two_columns('c', self.c_phys, 'ms-2')
                +f"{'CONVERSION TO SI':-^72}" + "\n"
                +f"{'Multiply the code values by these factors to get SI:':^72}"+ "\n"
                + two_columns('code_v', self.code_v, '')
                + two_columns('code_n', self.code_n, '')
                + two_columns('code_j', self.code_j, '')
                + two_columns('code_E', self.code_E, '')
                + two_columns('code_B', self.code_B, '')
                + two_columns('code_T', self.code_T, '')
                + two_columns('code_force', self.code_force, '')
                + two_columns('code_force2', self.code_force2, '')
                + two_columns('code_forceden', self.code_forceden, '')
                + two_columns('code_forceden2', self.code_forceden2, '')
                + two_columns('code_forceden3', self.code_forceden3, '')
                + two_columns('code_eflux', self.code_eflux, '')
                + two_columns('code_eflux2', self.code_eflux2, '')
                + two_columns('code_eflux3', self.code_eflux3, '')
                # + two_columns('code_eden', self.code_eden, '')
                # + two_columns('code_energy', self.code_energy, '')
                # + two_columns('code_energy2', self.code_energy2, '')
                # + two_columns('code_energy3', self.code_energy3, '')
                # + two_columns('code_energy4', self.code_energy4, '')
                + two_columns('code_P', self.code_P, '')
                + two_columns('code_space', self.code_space, '')
                + two_columns('code_time', self.code_time, '')
                + two_columns('momentum_corrector', self.momentum_corrector, '')
                +f"{'TIME':-^72}" + "\n"
                + two_columns('dt_phys', self.dt_phys, 's (one cycle)')
                +f"{'TIME TO TRAVERSE THE GRID':-^72}" + "\n"
                + two_columns('t_cross (phys)', self.t_cross_phys/60, 'min')
                + two_columns('t_cross (cycles)', self.t_cross_cycles, 'cycles')
                +f"{'OTHER TIME AND LENGTH SCALES':-^72}" + "\n"
                + two_columns('plasma freq, wpp', self.wpp, f'Hz ({self.wpp*self.dt_phys*1000:6.1f} per 1k cycles)')
                + two_columns('electron plasma freq, wpe', self.wpe, f'Hz ({self.wpe*self.dt_phys*1000:6.1f} per 1k cycles)')
                + two_columns('proton cyclotron freq, wcp', self.wcp, f'Hz ({self.wcp*self.dt_phys*1000:6.1f} per 1k cycles)')
                + two_columns('electron cyclotron freq, wce', self.wce, f'Hz ({self.wce*self.dt_phys*1000:6.1f} per 1k cycles)')
                + two_columns('proton inertial length, dp', self.dp, f'm ({self.dp/self.unit_phys:.3f} {up})')
                + two_columns('electron inertial length, de', self.de, f'm ({self.de/self.unit_phys:.3f} {up})')
                + two_columns('proton gyroradius, dcp', self.dcp, f'm ({self.dcp/self.unit_phys:.2f} {up})')
                + two_columns('electron gyroradius, dce', self.dce, f'm ({self.dce/self.unit_phys:.3f} {up})')
                + two_columns('phys_in_code_units_phys', self.phys_in_code_units_phys, '')
                + two_columns('phys_in_code_units', self.phys_in_code_units[0], '')
                + two_columns('scaling', self.scaling[0], '')
                # +f"    critical n to resolve electron scale l., n_crit = {self.n_crit/1e6:,.4f} m-3\n"
                +f"========================================================\n")

    def calculate_magnetic_dipole(self, B0=None, center=[0,0,0], coord='phys'):
        """ Calculate the magnetic dipole field """
        if coord == 'phys':
            bins = self.bins_phys
        else:
            bins = self.bins_code
        X, Y, Z = np.meshgrid(bins[0]-center[0],
                              bins[1]-center[1],
                              bins[2]-center[2],
                              indexing='ij')
        r = np.sqrt(X**2 + Y**2 + Z**2)
        B0 = self.B0 if B0 is None else B0
        Bx = 3*B0*X*Z/r**5
        By = 3*B0*Y*Z/r**5
        Bz = self.B0*(3*Z**2 - r**2)/r**5
        return Bx, By, Bz

    def code_to_phys_rot(self, R):
        return code_to_phys_rot(R)
    def phys_to_code_rot(self, R):
        return phys_to_code_rot(R)
    def code_to_phys(self, R, **kwargs):
        return code_to_phys(R, self.size_code, self.min_phys, self.max_phys, **kwargs)
    def code_to_cell(self, R, **kwargs):
        return code_to_cell(R, self.size_code, self.size_cell, **kwargs)
    def phys_to_code(self, R, **kwargs):
        return phys_to_code(R, self.size_code, self.min_phys, self.max_phys, **kwargs)
    def phys_to_cell(self, R, **kwargs):
        return phys_to_cell(R, self.size_cell, self.min_phys, self.max_phys, **kwargs)
    def cell_to_phys(self, R, **kwargs):
        return cell_to_phys(R, self.size_cell, self.min_phys, self.max_phys, **kwargs)
    def cell_to_code(self, R, **kwargs):
        return cell_to_code(R, self.size_cell, self.size_code, **kwargs)


class Selection:
    """ Selection information in physical and code coordinates """
    def __init__(self,
                 sim,
                 species = None,
                 cycle = 0,
                 name = None,
                 data_dir = '/Users/leo/DATA',
                 data_external_dir = '/Volumes/T7/Ring_mar23LG',
                 output_dir = '',
                 figure_dir = 'figures',
                 center_code = None,
                 center_phys = None,
                 center_cell = None,
                 delta_code = None,
                 delta_phys = None,
                 delta_cell = None,
                 min_code = None,
                 max_code = None,
                 min_phys = None,
                 max_phys = None,
                 min_cell = None,
                 max_cell = None,
                 chunk_size = 5e7,
                 particle_data_keys = particle_data_keys,
                 field_data_keys = field_data_keys,
                 dtype = 'f8',
                 cycle_limits = None,
                 interpolate = False,
                 id = None,
                 ):
        # --- INITIALIZE
        self.sim = sim
        self.species = species
        self.name = name
        self.data_dir = data_dir
        self.data_external_dir = data_external_dir
        self.output_dir = output_dir
        self.figure_dir = figure_dir
        self.cycle = cycle
        self._delta_code = delta_code
        self._delta_phys = delta_phys
        self._center_code = center_code
        self._center_phys = center_phys
        self._center_cell = center_cell
        self._delta_code = delta_code
        self._delta_phys = delta_phys
        self._delta_cell = delta_cell
        self._min_code = min_code
        self._max_code = max_code
        self._min_phys = min_phys
        self._max_phys = max_phys
        self._min_cell = min_cell
        self._max_cell = max_cell
        self.cycle_limits = cycle_limits
        self.chunk_size = int(chunk_size)
        self.particle_data_keys = particle_data_keys
        self.field_data_keys = field_data_keys
        self.dtype = dtype
        self.interpolate = interpolate
        self.filename_fields_fixed = None
        self.filename_particles_fixed = None
        self.id = id

        # --- DEPENDENT VARIABLES
        if min_phys is not None and max_phys is not None:
            self.min_phys = min_phys
            self.max_phys = max_phys
        elif min_code is not None and max_code is not None:
            self.min_code = min_code
            self.max_code = max_code
        elif min_cell is not None and max_cell is not None:
            self.min_cell = min_cell
            self.max_cell = max_cell
        elif center_phys is not None:
            self.center_phys = center_phys
        elif center_code is not None:
            self.center_code = center_code
        elif center_cell is not None:
            self.center_cell = center_cell
        else:
            if delta_phys is not None:   # Set default center
                # self._center_phys = [0, 0, 0]
                self._center_phys = (0, 0, 0)
                self.delta_phys = delta_phys
            elif delta_code is not None: # Set default center
                # self._center_code = [0, 0, 0]
                self._center_code = (0, 0, 0)
                self.delta_code = delta_code
            elif delta_cell is not None: # Set default center
                # self._center_cell = [0, 0, 0]
                self._center_cell = (0, 0, 0)
                self.delta_cell = delta_cell
            else:                        # No parameters provided
                # self._delta_phys = [1, 1, 1]
                self._delta_phys = (1, 1, 1)
                # self.center_phys = [0, 0, 0]
                self.center_phys = (0, 0, 0)

    @property
    def range_code(self):
        return np.array([self.min_code, self.max_code]).T
    @property
    def range_phys(self):
        return np.array([self.min_phys, self.max_phys]).T
    @property
    def range_cell(self):
        return np.array([self.min_cell, self.max_cell]).T

    @property
    def size_code(self):
        return np.subtract(self.max_code, self.min_code)
    @property
    def size_phys(self):
        return np.subtract(self.max_phys, self.min_phys)
    @property
    def size_cell(self):
        return np.subtract(self.max_cell, self.min_cell)

    # TODO: change the bin functions
    # def cell_centers(min, max, size, dx=None, coord='code'):
    def x_bins(self, field=None):
        field = self.field if field is None else field
        return np.linspace(self.max_phys[0], self.min_phys[0], field.shape[0])

    def y_bins(self, field=None):
        field = self.field if field is None else field
        # if field.ndim < 3:
        if self.cut_axis_phys == 0:
            return np.linspace(self.min_phys[1], self.max_phys[1], field.shape[0])
        else:
            return np.linspace(self.min_phys[1], self.max_phys[1], field.shape[1])

    def z_bins(self, field=None):
        field = self.field if field is None else field
        # if field.ndim < 3:
        return np.linspace(self.min_phys[2], self.max_phys[2], field.shape[1])
        # if self.cut_axis_phys == 0:
        #     return np.linspace(self.min_phys[2], self.max_phys[2], field.shape[1])
        # else:
        #     return np.linspace(self.min_phys[2], self.max_phys[2], field.shape[2])

    def bins(self, field=None):
        field = self.field if field is None else field
        return [self.x_bins(field), self.y_bins(field), self.z_bins(field)]

    def xy_bins(self, field=None):
        field = self.field if field is None else field
        if self.cut_axis_phys == 0:
            a,b = self.y_bins(field), self.z_bins(field)
        elif self.cut_axis_phys == 1:
            a,b = self.x_bins(field), self.z_bins(field)
        elif self.cut_axis_phys == 2:
            a,b = self.x_bins(field), self.y_bins(field)
        else:
            print('No cut axis can be determined.')
            exit()
        return np.meshgrid(a, b)

    def xyz_bins(self, field=None):
        field = self.field if field is None else field
        cut_coord = self.cut_coord_phys
        if self.cut_axis_phys == 0:
            mesh = np.meshgrid(self.y_bins(field), self.z_bins(field))
            return np.full_like(mesh[0], cut_coord), mesh[0], mesh[1]
        elif self.cut_axis_phys == 1:
            mesh = np.meshgrid(self.x_bins(field), self.z_bins(field))
            return mesh[0], np.full_like(mesh[0], cut_coord), mesh[1]
        elif self.cut_axis_phys == 2:
            mesh = np.meshgrid(self.x_bins(field), self.y_bins(field))
            return mesh[0], mesh[1], np.full_like(mesh[0], cut_coord)
        else:
            print('No cut axis can be determined.')
            exit()

    def includes(self, r):
        """ Check if a given position is within the selection """
        return np.all(np.logical_and(self.min_phys <= r, r <= self.max_phys))
        # return includes(R, self.min_phys, self.max_phys)

    @property
    def field(self):
        field = getattr(self, 'bz', None)
        if field is None:
            randomize = True
            cut_axis = self.cut_axis_phys
            # print(f'cell in phys units = {self.sim.cell_in_phys_units}')
            # print(f'size_phys = {self.size_phys}')
            # exit()
            if cut_axis == 0:
                field = np.ones((int(self.size_phys[1]//self.sim.cell_in_phys_units[1]),
                                 int(self.size_phys[2]//self.sim.cell_in_phys_units[2])))
            elif cut_axis == 1:
                field = np.ones((int(self.size_phys[0]//self.sim.cell_in_phys_units[0]),
                                 int(self.size_phys[2]//self.sim.cell_in_phys_units[2])))
            elif cut_axis == 2:
                field = np.ones((int(self.size_phys[0]//self.sim.cell_in_phys_units[0]),
                                 int(self.size_phys[1]//self.sim.cell_in_phys_units[1])))
            else:
                field = np.ones((int(self.size_phys[0]//self.sim.cell_in_phys_units[0]),
                                 int(self.size_phys[1]//self.sim.cell_in_phys_units[1]),
                                 int(self.size_phys[2]//self.sim.cell_in_phys_units[2])
                                 ))
                print('No cut axis can be determined.')
                exit()
            if randomize:
                from scipy.ndimage import gaussian_filter
                field = np.random.rand(*field.shape)
                field = gaussian_filter(field, sigma=3)
        return field
    @property
    def cut_axis_code(self):
        if np.any(self.size_code == 0):
            return np.where(self.size_code == 0)[0][0]
        else:
            return 1
    @property
    def cut_axis_phys(self):
        if np.any(self.size_phys == 0):
            return np.where(self.size_phys == 0)[0][0]
        else:
            return 2
    @property
    def cut_coord_phys(self):
        return self.center_phys[self.cut_axis_phys]
    @property
    def cut_coord_code(self):
        return self.center_code[self.cut_axis_code]

    @property
    def cycle(self):
        return self._cycle
    @cycle.setter
    def cycle(self, cycle):
        self._cycle = cycle
    @property
    def x(self):
        x, y, z = self.xyz_bins()
        return x
    @property
    def y(self):
        x, y, z = self.xyz_bins()
        return y
    @property
    def z(self):
        x, y, z = self.xyz_bins()
        return z
    @property
    def r(self):
        x, y, z = self.xyz_bins()
        return np.sqrt(x**2 + y**2 + z**2)
    # --- BINS FOR PLOTTING
    # x, y = selection.xy_bins()
    # r = np.sqrt(x**2 + y**2)
    # z = np.zeros_like(x)

    @property
    def selection_filenames(self):
        if self.name is not None:
            if self.cycle is not None:
                filename_particles = f"{self.sim.name}-Partcl_{self.cycle:06d}_{self.name}.h5"
                filename_fields   = f"{self.sim.name}-Fields_{self.cycle:06d}_{self.name}.h5"
                return dict(particles=filename_particles, fields=filename_fields)
            else:
                filename_particles = f"{self.sim.name}-Partcl_{self.name}.h5"
                filename_fields   = f"{self.sim.name}-Fields_{self.name}.h5"
                return dict(particles=filename_particles, fields=filename_fields)
        else:
            return dict(particles=self.filename_particles, fields=self.filename_fields)

    def get_filenames(self, meta=None):
        if meta is None:
            fn_partcl = f"{self.sim.name}-Partcl_{self.cycle:06d}.h5"
            fn_fields = f"{self.sim.name}-Fields_{self.cycle:06d}.h5"
        else:
            fn_partcl = f"{self.sim.name}-Partcl_{self.cycle:06d}_{meta}.h5"
            fn_fields = f"{self.sim.name}-Fields_{self.cycle:06d}_{meta}.h5"
        return dict(particles=fn_partcl, fields=fn_fields)

    def filename_fields(self, cycle=None, meta=None):
        cycle = self.cycle if cycle is None else int(cycle)
        if meta is None:
            return f"{self.sim.name}-Fields_{self.cycle:06d}.h5"
        else:
            return f"{self.sim.name}-Fields_{self.cycle:06d}_{meta}.h5"

    def filename_particles(self, cycle=None, meta=None):
        cycle = self.cycle if cycle is None else int(cycle)
        if meta is None:
            return f"{self.sim.name}-Partcl_{self.cycle:06d}.h5"
        else:
            return f"{self.sim.name}-Partcl_{self.cycle:06d}_{meta}.h5"

    def path_fields(self, cycle=None, meta=None):
        return os.path.join(self.data_dir, self.filename_fields(cycle, meta))
    def path_particles(self, cycle=None, meta=None):
        return os.path.join(self.data_dir, self.filename_particles(cycle, meta))

    @property
    def center_phys(self):
        return self._center_phys
    @center_phys.setter
    def center_phys(self, center_phys):
        self._center_phys = center_phys
        self._center_code = self.sim.phys_to_code(center_phys)
        self._center_cell = self.sim.phys_to_cell(center_phys)
        self.find_min_max()

    @property
    def center_code(self):
        return self._center_code
    @center_code.setter
    def center_code(self, center_code):
        self._center_code = center_code
        self._center_phys = self.sim.code_to_phys(center_code)
        self._center_cell = self.sim.code_to_cell(center_code)
        self.find_min_max()

    @property
    def center_cell(self):
        return self._center_cell
    @center_cell.setter
    def center_cell(self, center_cell):
        self._center_cell = center_cell
        self._center_phys = self.sim.cell_to_phys(center_cell)
        self._center_code = self.sim.cell_to_code(center_cell)
        self.find_min_max()

    @property
    def delta_code(self):
        return self._delta_code
    @delta_code.setter
    def delta_code(self, delta_code):
        self._delta_code = delta_code
        self._delta_phys = self.sim.code_to_phys(delta_code, vector=True)
        self._delta_cell = self.sim.code_to_cell(delta_code, vector=True)
        self.find_min_max()

    @property
    def delta_phys(self):
        return self._delta_phys
    @delta_phys.setter
    def delta_phys(self, delta_phys):
        self._delta_phys = delta_phys
        self._delta_code = self.sim.phys_to_code(delta_phys, vector=True)
        self._delta_cell = self.sim.phys_to_cell(delta_phys, vector=True)
        self.find_min_max()

    @property
    def delta_cell(self):
        return self._delta_cell
    @delta_cell.setter
    def delta_cell(self, delta_cell):
        self._delta_cell = delta_cell
        self._delta_code = self.sim.cell_to_code(delta_cell, vector=True)
        self._delta_phys = self.sim.cell_to_phys(delta_cell, vector=True)
        self.find_min_max()

    @property
    def min_phys(self):
        return self._min_phys
    @min_phys.setter
    def min_phys(self, min_phys):
        self._min_phys = min_phys
        self._min_code = self.sim.phys_to_code(min_phys)
        self._min_cell = self.sim.phys_to_cell(min_phys)
        self.find_delta_and_center()

    @property
    def max_phys(self):
        return self._max_phys
    @max_phys.setter
    def max_phys(self, max_phys):
        self._max_phys = max_phys
        self._max_code = self.sim.phys_to_code(max_phys)
        self._max_cell = self.sim.phys_to_cell(max_phys)
        self.find_delta_and_center()

    @property
    def min_code(self):
        return self._min_code
    @min_code.setter
    def min_code(self, min_code):
        self._min_code = min_code
        self._min_phys = self.sim.code_to_phys(min_code)
        self._min_cell = self.sim.code_to_cell(min_code)
        self.find_delta_and_center()

    @property
    def max_code(self):
        return self._max_code
    @max_code.setter
    def max_code(self, max_code):
        self._max_code = max_code
        self._max_phys = self.sim.code_to_phys(max_code)
        self._max_cell = self.sim.code_to_cell(max_code)
        self.find_delta_and_center()

    @property
    def min_cell(self):
        return self._min_cell
    @min_cell.setter
    def min_cell(self, min_cell):
        self._min_cell = min_cell
        self._min_phys = self.sim.cell_to_phys(min_cell)
        self._min_code = self.sim.cell_to_code(min_cell)
        self.find_delta_and_center()

    @property
    def max_cell(self):
        return self._max_cell
    @max_cell.setter
    def max_cell(self, max_cell):
        self._max_cell = max_cell
        self._max_phys = self.sim.cell_to_phys(max_cell)
        self._max_code = self.sim.cell_to_code(max_cell)
        self.find_delta_and_center()

    def find_min_max(self):
        if self.delta_phys is not None and self.delta_code is None:
            self._delta_code = self.sim.phys_to_code(self.delta_phys, vector=True)
            self._delta_cell = self.sim.phys_to_cell(self.delta_phys, vector=True)
        if self.delta_code is not None and self.delta_phys is None:
            self._delta_phys = self.sim.code_to_phys(self.delta_code, vector=True)
            self._delta_cell = self.sim.code_to_cell(self.delta_code, vector=True)
        if self.delta_cell is not None and self.delta_phys is None:
            self._delta_phys = self.sim.cell_to_phys(self.delta_cell, vector=True)
            self._delta_code = self.sim.cell_to_code(self.delta_cell, vector=True)
        if self.delta_phys is None and self.delta_code is None:
            # self._delta_phys = [1, 1, 1]
            self._delta_phys = (1, 1, 1)
            self._delta_code = self.sim.phys_to_code(self.delta_phys, vector=True)
            self._delta_cell = self.sim.phys_to_cell(self.delta_phys, vector=True)
        if self.center_code is not None and self.delta_code is not None:
            self._min_code = np.subtract(self.center_code, np.asarray(self.delta_code)/2.)
            self._max_code = np.add(     self.center_code, np.asarray(self.delta_code)/2.)
        if self.center_phys is not None and self.delta_phys is not None:
            self._min_phys  = np.subtract(self.center_phys, np.asarray(self.delta_phys)/2.)
            self._max_phys  = np.add(     self.center_phys, np.asarray(self.delta_phys)/2.)
        if self.center_cell is not None and self.delta_cell is not None:
            self._min_cell  = np.subtract(self.center_cell, np.asarray(self.delta_cell)//2)
            self._max_cell  = np.add(     self.center_cell, np.asarray(self.delta_cell)//2)

    def find_delta_and_center(self):
        """ Find the delta and center from the min and max coordinates. """
        if self.min_code is not None and self.max_code is not None:
            self._min_code,  self._max_code  = sort_selection(self.min_code, self.max_code)
            self._delta_code = np.subtract(self.max_code, self.min_code)
            self._center_code = np.add(self.min_code, np.asarray(self.delta_code)/2.)
        if self.min_phys is not None and self.max_phys is not None:
            self._min_phys, self._max_phys = sort_selection(self.min_phys, self.max_phys)
            self._delta_phys = np.subtract(self.max_phys, self.min_phys)
            self._center_phys = np.add(self.min_phys, np.asarray(self.delta_phys)/2.)
        if self.min_cell is not None and self.max_cell is not None:
            self._min_cell, self._max_cell = sort_selection(self.min_cell, self.max_cell)
            self._delta_cell = np.subtract(self.max_cell, self.min_cell)
            self._center_cell = np.add(self.min_cell, np.asarray(self.delta_cell)//2)
        # if self._min_code is not None and self._max_code is not None:
        #     self._min_code,  self._max_code  = sort_selection(self._min_code, self._max_code)
        #     self._delta_code = np.subtract(self._max_code, self._min_code)
        #     self._center_code = np.add(self._min_code, np.asarray(self._delta_code)/2.)
        # if self._min_phys is not None and self._max_phys is not None:
        #     self._min_phys, self._max_phys = sort_selection(self._min_phys, self._max_phys)
        #     self._delta_phys = np.subtract(self._max_phys, self._min_phys)
        #     self._center_phys = np.add(self._min_phys, np.asarray(self._delta_phys)/2.)
        # if self._min_cell is not None and self._max_cell is not None:
        #     self._min_cell, self._max_cell = sort_selection(self._min_cell, self._max_cell)
        #     self._delta_cell = np.subtract(self._max_cell, self._min_cell)
        #     self._center_cell = np.add(self._min_cell, np.asarray(self._delta_cell)//2)

    def get_filename_particles_output(self):
        """ Get the filename for a particle chunk-sized data file """
        self.filename_particles_output = None
        if (self.cycle is None or
            self.delta_phys is None or
            self.min_phys is None or
            self.max_phys is None or
            self.center_phys is None):
            return None

        if ((self.delta_phys[0] != self.delta_phys[1])
            or (self.delta_phys[0] != self.delta_phys[2])):
            self.filename_particles_output = (
                     f'{self.sim.name}-Partcl'+
                     f'_{self.cycle:06d}'+
                     f'_{self.min_phys[0]:.2f}to{self.max_phys[0]:.2f}X'+
                     f'_{self.min_phys[1]:.2f}to{self.max_phys[1]:.2f}Y'+
                     f'_{self.min_phys[2]:.2f}to{self.max_phys[2]:.2f}Z'+
                     f'.h5')
        else:
            self.filename_particles_output = (
                     f'{self.sim.name}-Partcl'+
                     f'_{self.cycle:06d}'+
                     f'_{self.center_phys[0]:.2f}X'+
                     f'_{self.center_phys[1]:.2f}Y'+
                     f'_{self.center_phys[2]:.2f}Z'+
                     f'_{self.delta_phys[0]:.2f}dx'+
                     f'.h5')
        return self.filename_particles_output

    def duplicate(self):
        """ Return a copy of the selection """
        return copy.deepcopy(self)

    def get_field(self, key=None, mask=True, mask_radius=None):
        """ Get a field from the Selection object """
        if isinstance(key, list):
            for k in key:
                if not hasattr(self, k):
                    return [None]*len(key)
            field = [getattr(self, _, None) for _ in key]
        else:
            if key is None:
                return None
            field = getattr(self, key, None)
        
        # Mask fields
        if field is not None and mask:
            from pypic.calculate import mask_fields
            field = mask_fields(field, self, value=np.nan, radius=2)
        return field

    def calculate(self, interpolate=False, quick=False):
        from pypic.calculate import calculate_fields, calculate_quick, calculate_quick_3Dfields
        if quick:
            # calculate_quick(self, [self.tmp, self.tmp2, self.tmp3])
            if hasattr(self, 'tmp'):
                calculate_quick(self, [self.tmp, self.tmp2, self.tmp3])
            else:
                calculate_quick_3Dfields(self)
        else:
            calculate_fields(self, interp=interpolate)

    def calculate_quick(self):
        from pypic.calculate import calculate_fields, calculate_quick_3Dfields
        calculate_quick_3Dfields(self)

    def to_dataframe(self, keys, y_cut=None):
        """ Convert the selection to a pandas dataframe """
        import pandas as pd
        if y_cut is None:
            y_cut = self.center_cell[2]
        key_values = {}
        for k in keys:
            if k == 'x':
                values = self.x_bins()
            elif k == 'y':
                values = self.y_bins()
            elif k == 'z':
                values = self.z_bins()
            else:
                values = self.get_field(k)
                if y_cut is not None:
                    # y_bins = self.y_bins()
                    y_bins = values.shape[1]
                    y_cut = y_bins//2
                    values = values[:, y_cut]
            key_values[k] = values
        df = pd.DataFrame(key_values)
        df['cycle'] = self.cycle
        return df

    def __repr__(self):
        uc = self.sim.unit_name_code
        up = self.sim.unit_name_phys
        return (f"====================== {self.name} =======================\n"
                +f"    Sim Grid Size    [{up}][phys]  = {self.sim.size_phys}\n"
                +f"    Sim Grid Size    [{uc}][code]  = {self.sim.size_code}\n"
                +f"    Sim Grid Size    [ #][cell]  = {self.sim.size_cell}\n"
                +f'    --------------------------------------------------\n'
                +f'    Selection Center [{up}][phys]  = {self.center_phys}\n'
                +f'    Selection Delta  [{up}][phys]  = {self.delta_phys}\n'
                +f'    Selection Min    [{up}][phys]  = {self.min_phys}\n'
                +f'    Selection Max    [{up}][phys]  = {self.max_phys}\n'
                +f'    Selection Center [{uc}][code]  = {self.center_code}\n'
                +f'    Selection Delta  [{uc}][code]  = {self.delta_code}\n'
                +f'    Selection Min    [{uc}][code]  = {self.min_code}\n'
                +f'    Selection Max    [{uc}][code]  = {self.max_code}\n\n'
                +f'    Selection Center [ #][cell]  = {self.center_cell}\n'
                +f'    Selection Delta  [ #][cell]  = {self.delta_cell}\n'
                +f'    Selection Min    [ #][cell]  = {self.min_cell}\n'
                +f'    Selection Max    [ #][cell]  = {self.max_cell}\n\n'
                # +f'    Range in code units [{uc}][code] = {self.range_code}\n'
                # +f'    Range in phys units [{up}][phys] = {self.range_phys}\n'
                # +f'    Range in cells      [ #][cell]   = {self.range_cell}\n'
                +f'    --------------------------------------------------\n'
                +f'    Cycle         = {self.cycle:,d}\n'
                +f'    Species       = {self.species} (0=e-, 1=ions, 2=HE e-, 3=HE ions)\n'
                +f'    --------------------------------------------------\n'
                +f'    data dir      = "{self.data_dir}"\n'
                +f'    output dir    = "{self.output_dir}"\n'
                +f'    figure dir    = "{self.figure_dir}"\n'
                +f'    particle file = "{self.filename_particles()}"\n'
                +f'    fields file   = "{self.filename_fields()}"\n'
                +f'    --------------------------------------------------\n'
                +f'    Reading particles in chunks of {int(self.chunk_size):,} particles.\n'
                +f'    id = {self.id if self.id is not None else "None"}\n'
                +f"=======================================================\n")

if __name__ == "__main__":
    sim = ipic3D()
    print(sim)
