import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if path not in sys.path:
    sys.path.append(path)

from pypic.core import *
from pypic.plot import *
from pypic.input_output import *
import pypic.colors as cl

def step_Euler(f, r, b=None, step=1.):
    """ Incremental step using the Euler method """
    if b is None:
        b = f(r)
    bt = np.sqrt(b.dot(b))
    step = np.divide(b, bt) * step
    r = r + step
    b = f(r)
    return r, b

def step_RK4(f, r, step=1.):
    """ Incremental step using the Runge-Kutta method """
    k1 = f(r)
    k1 = k1 / np.linalg.norm(k1)
    k2 = f(r + 0.5 * step * k1)
    k2 = k2 / np.linalg.norm(k2)
    k3 = f(r + 0.5 * step * k2)
    k3 = k3 / np.linalg.norm(k3)
    k4 = f(r + 1.0 * step * k3)
    k4 = k4 / np.linalg.norm(k4)
    kavg = (k1 + 2.0 * (k2 + k3) + k4) * (step / 6.0)
    r = r + kavg
    b = f(r)
    return r, b

def get_trace_error(b1, b2, step):
    """ Return the trace error as a percentage of the step size """
    b1 = b1/np.linalg.norm(b1)
    b2 = b2/np.linalg.norm(b2)
    theta = np.arccos(np.dot(b1, b2))
    error = step * np.sin(theta)
    return error * 100. / step

def check_boundaries(R,
                     n_iter = None,
                     max_iter = 1e5,
                     r_min = 1.0,
                     r_max = 50.,
                     lower_bound = None,
                     upper_bound = None,
                     verbose = True,
                     z_intersect = None,
                     max_intersections = 3,
                     ):
    """ Check field line tracing conditions """

    if n_iter is not None and n_iter >= max_iter:
        message = f'Exceeded Maximum Iterations ({n_iter}). Tracing finished.'
        if verbose:
            print(f'{message}')
        return False, 9

    # calculate the radius at the position
    r = np.linalg.norm(R, axis=-1)

    # calculate the spheroid radius closest to the position
    # r_spheroid = np.sqrt(np.cos(np.pi/2 - th)**2 * 1.
                       # + np.sin(np.pi/2 - th)**2 * (1. - self.flatenning)**2)
    # if r_spheroid is not None:
    #     if r <= r_spheroid:
    #         print(f'r_sph < {r_spheroid:.2f}.')
    #         return False
    if np.any(np.isnan(r)):
        message = f'r = nan. Tracing finished.'
        if verbose:
            print(f'{message}')
        return False, 10
    if r_min is not None and r <= r_min:
        message = f'r <= {r_min:.2f} ({r:.2f}). Tracing finished.'
        if verbose:
            print(f'{message}')
        return False, 2

    if r_max is not None and  r >= r_max:
        message = f'r >= {r_max:.2f} (r_max). Tracing finished.'
        if verbose:
            print(f'{message}')
        return False, 3

    if lower_bound is not None and np.any(R <= lower_bound):
        message = f'R <= {lower_bound} (lower_bound). Tracing finished'
        if verbose:
            print(f'{message}')
        return False, 4

    if upper_bound is not None and np.any(R >= upper_bound):
        message = f'R >= {upper_bound} (upper_bound). Tracing finished'
        if verbose:
            print(f'{message}')
        return False, 5

    # otherwise, a successful trace
    return True, 1


class fieldline:
    """ fieldline class for trace parameters """
    _id_counter = 0
    def __init__(self,
                 r=None, # Initial position
                 name=None,
                 step=1., # Step size
                 method='RK4', # Tracing method
                 max_error=None, # Max error in percentage of step size
                 direction=1, # Tracing direction
                 B_func=None, # Magnetic field function
                 n_func=None, # Density function
                 boundary_conditions=None, # Boundary conditions for tracing
                 backtrace=False,
                 ):
        if r is not None:
            r = np.array(r)
        self.r = r[np.newaxis, :]
        self.L = None # Field line equatorial distance
        self.ds = None # Field line steps
        self.vA = None # Alfven velocity
        self.n = None
        self.B = None
        self.markers = None # Markers for field line points
        self.errors = None # Errors in the field line tracing
        self.name = name
        self.step = step
        self.method = method
        self.n_iter = 0 # Number of iterations
        self.max_error = max_error
        self.direction = direction
        self.B_func = B_func
        self.n_func = n_func
        self.backtrace = backtrace
        self.N_intersects = None
        self.id = type(self)._id_counter  # Set the ID for the instance
        type(self)._id_counter += 1  # Increment the counter for each new instance

        # Set up default boundary conditions 
        bcs = {'r_min': 3.,
               'r_max': 30.,
               'lower_bound': None,
               'upper_bound': None,
               'max_iter': 1e4,
               'verbose': True,
               'z_intersect': None,
               'max_intersections': 1,
               }

        # Update the boundary conditions if provided
        if boundary_conditions is not None:
            bcs.update(boundary_conditions)
            if bcs['verbose']:
                print(f'Using boundary conditions: {bcs}')
        else:
            print(f'Using default boundary conditions: {bcs}')

        self.boundary_conditions = bcs
        # Add the initial point and corresponding field data
        self.new_point(step=0)

    def next_trace_step(self,
                        r = None,
                        B = None,
                        step = None,
                        save = True,
                        ):
        """ Next trace point from the last point """
        r = r if r is not None else self.r[-1]
        step = step if step is not None else self.step

        # Either step along or against the field
        step *= self.direction

        # if self.max_error is not None:
        # TODO: Implement an adaptive step size
        if self.method.lower() == 'euler':
            r, B = step_Euler(self.B_func, r, step=step)
        else: # 'RK4':
            r, B = step_RK4(self.B_func, r, step=step)

        if np.any(np.isnan(r)) or np.any(np.isnan(B)):
            # print(f'Nan values found while tracing: r = {r}, B = {B}')
            return r, B, False
        if np.any(np.isinf(r)) or np.any(np.isinf(B)):
            # print(f'Inf values found while tracing: r = {r}, B = {B}')
            return r, B, False
        # if np.any(abs(B)<1e-12):
        #     print(f'Zero values found while tracing: r = {r}, B = {B}')
        #     return r, B, False
        # if np.any(abs(B)<0.01):
        #     print(f'r = {r}, B = {B}')

        error = get_trace_error(B, self.B[-1], abs(step))

        check, marker = check_boundaries(r, self.n_iter,
                                         **self.boundary_conditions)
        if save:
            self.new_point(r, B, step=step, marker=marker, error=error)
            self.n_iter += 1
        return r, B, check

    """ Trace a field line from existing position data """
    def trace(self, directions=(1, -1)):
        # Trace for both directions along the field
        for direction in directions:
            self.n_iter = 0
            self.direction = direction
            within_bounds = True
            while within_bounds:
                _,_,within_bounds = self.next_trace_step()
            if not self.backtrace:
                self.reverse_data()
        radius = np.linalg.norm(self.r, axis=-1)
        # Find the max equatorial crossing distance
        self.L = np.max(radius)
        self.count_intersections(z_intersect=0)

    def new_point(self, r=None, B=None, step=None, marker=0, error=0):
        """ Add position and field data to the field line trace """
        if r is None:
            r = self.r[-1]
        else:
            self.r = np.vstack((self.r, r))

        # --- Add the field
        if B is None:
            B = self.B_func(r)

        if self.B is None:
            self.B = B[np.newaxis, :]
        else:
            self.B = np.vstack((self.B, B))

        # --- Check if a density function is provided
        if self.n_func is not None:
            n = self.n_func(r)
        else:
            n = np.nan

        # --- Calculate the Alfven velocity
        vA = np.linalg.norm(B)/np.sqrt(n)
        # mu0 = 4*np.pi*1e-7
        # vA = np.linalg.norm(B*1e-9)/np.sqrt(mu0*1e6)

        if step is None:
            step = self.step

        # --- Add the rest of data
        if self.n is None:
            self.n = [n]
            self.markers = [marker]
            self.ds = [step]
            self.vA = [vA]
            self.errors = [error]
        else:
            self.n.append(n)
            self.markers.append(marker)
            self.ds.append(step)
            self.vA.append(vA)
            self.errors.append(error)

    def reverse_data(self):
        """ Reverse the data in the field line trace """
        self.r = np.flip(self.r, axis=0)
        self.B = np.flip(self.B, axis=0)
        self.n = np.flip(self.n, axis=0).tolist()
        self.vA = np.flip(self.vA, axis=0).tolist()
        self.ds = np.flip(self.ds, axis=0).tolist()
        self.markers = np.flip(self.markers, axis=0).tolist()
        self.errors = np.flip(self.errors, axis=0).tolist()

    def plot_errors(self):
        """ Plot the errors in the field line tracing """
        fig, ax = plt.subplots()
        from scipy.ndimage import gaussian_filter1d
        smoothed_errors = gaussian_filter1d(self.errors, len(self.errors)//20)
        ax.plot(self.errors, label=self.name)
        ax.plot(smoothed_errors, label=f'{self.name} (smoothed)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Error [%]')
        ax.legend()
        plt.show()

    # field line ID generator using next
    def __next__(self):
        return self

    def is_it_closed(self, selection=None, z_intersect=None):
        """ Check if the field line is closed """
        # First and last position
        r0 = self.r[0]
        r1 = self.r[-1]
        dr = r1 - r0
        separation = np.linalg.norm(dr)
        radius0 = np.linalg.norm(r0)
        radius1 = np.linalg.norm(r1)

        max_intersects = self.boundary_conditions['max_intersections']
        if z_intersect is None:
            z_intersect = self.boundary_conditions['z_intersect']
        r_min = self.boundary_conditions['r_min']

        # First and last z coordinates
        z0 = r0[2]
        z1 = r1[2]

        # Closed field line conditions
        N_intersects = self.count_intersections(selection=selection, z_intersect=z_intersect)
        both_hemispheres = z0*z1 < 0
        reached_center = radius0 <= r_min and radius1 <= r_min
        intersect_cond = N_intersects <= max_intersects if max_intersects is not None else True

        # Combine all closed conditions
        if both_hemispheres and reached_center and intersect_cond:
            return True
        else:
            return False

    @property
    def radius(self):
        """ Return the radius of the field line """
        return np.linalg.norm(self.r, axis=-1)
    @property
    def theta(self):
        """ Return the theta angle of the field line """
        return np.arccos(self.r.T[2]/self.radius)
    @property
    def colatitude(self):
        """ Return the colatitude angle of the field line """
        return np.pi/2 - self.theta
    @property
    def phi(self):
        """ Return the phi angle of the field line """
        return np.arctan2(self.r.T[1], self.r.T[0])

    @property
    def length(self):
        """ Return the length of the field line """
        # return np.sum(self.ds)
        return len(self.r)*self.step

    def count_intersections(self,
                            selection=None,
                            z_intersect=None,
                            ):
        """ Count the intersections with the selection planes """
        if selection is None and z_intersect is None:
            return 0
        if z_intersect is None:
            z_intersect = self.boundary_conditions['z_intersect']
        positions, zdirs = find_intersections(self.r,
                                              selection=selection,
                                              z_intersect=z_intersect)
        self.N_intersects = len(positions)
        return len(positions)

    def find_arrow_indeces(self):
        """ Find the indices for the arrows """
        ind_1 = len(self.r)//4
        ind_2 = len(self.r)//4*3
        from scipy.ndimage import gaussian_filter1d
        errors_smooth = gaussian_filter1d(self.errors, len(self.errors)//50)
        max_error = np.max(errors_smooth)
        small_errors = np.argwhere(errors_smooth < max_error/50)
        if len(small_errors) == 0:
            return ind_1, ind_2
        small_errors1 = small_errors[small_errors < ind_1]
        small_errors2 = small_errors[small_errors > ind_2]
        if len(small_errors1) > 0:
            ind_1 = np.max(small_errors1)
        if len(small_errors2) > 0:
            ind_2 = np.min(small_errors2)
        return ind_1, ind_2

    def plot_perp_vectors(self, ax, color='red', color2='green', scale=1):
        """ Plot the perpendicular vectors to the field line """
        for i in range(0, len(self.r), 10):
            r = self.r[i]
            B = self.B[i]
            small = 1e-9
            bx, by, bz = B
            b2D = bx**2 + by**2 + small
            b = np.linalg.norm(B)
            # b = B/np.linalg.norm(B)
            perp2x = bz * bx / np.sqrt(b * b2D)
            perp2y = bz * by / np.sqrt(b * b2D)
            perp2z = -np.sqrt(b2D / b)

            B_perp2 = np.array([perp2x, perp2y, perp2z])
            B_perp2 = B_perp2/np.linalg.norm(B_perp2)

            B_perp1 = np.cross(self.B[i], B_perp2)
            B_perp1 = B_perp1/np.linalg.norm(B_perp1)

            B_perp3 = np.array([B[1], -B[0], 0])
            B_perp3 = B_perp3/np.linalg.norm(B_perp3)

            # Reverse coords
            # B_perp1 = [B_perp1[0], B_perp1[2], B_perp1[1]]
            # B_perp2 = [B_perp2[0], B_perp2[2], B_perp2[1]]

            ax.quiver(*r, *B_perp2*scale, color=color, zorder=10, label='Perp 2')
            # ax.quiver(*r, *B_perp1*scale, color=color2, zorder=10, label='Perp 1')
            # ax.quiver(*r, *B_perp3*scale*1.2, color='yellow', zorder=10, label='Perp 3')

    def plot_fieldline(self,
                       ax=None,
                       color=None,
                       alpha=0.9,
                       color_key=None,
                       cmap=None,
                       norm=None,
                       lw=3,
                       colorbar=True,
                       end_points=True,
                       intersections=True,
                       z_intersect=None,
                       intersection_alpha=None,
                       arrows=True,
                       arrow_scale=25,
                       arrow_color='white',
                       arrow_alpha=1,
                       zdir='y', # 'x', 'y', 'z' (plotting priority)
                       zorder=None,
                       selection=None,
                       clip_on=None,
                       draw=True,
                       path_effects=True,
                       path_effects_alpha=0.6,
                       path_effects_lw=0.5,
                       path_effects_color='white',
                       path_effects_type='stroke',
                       **kwargs):
        """ Plot the field line """
        if color is None:
            if color_key is None:
                c = None
                cm = mpl.colormaps['turbo']
                nm = mpl.colors.Normalize(vmin=0, vmax=1)
                label = None
            else:
                if color_key.lower() in ['b', 'bt']:
                    c = np.linalg.norm(self.B, axis=-1)
                    cm = mpl.colormaps['turbo']
                    # cm = cmap
                    # nm = mpl.colors.Normalize(vmin=0, vmax=20)
                    nm = mpl.colors.Normalize(vmin=0, vmax=50)
                    label = r'$B_T$ [nT]'
                elif color_key.lower() in ['bz']:
                    c = self.B.T[-1]
                    cm = mpl.colormaps['turbo']
                    cm = cmap
                    nm = mpl.colors.Normalize(vmin=-20, vmax=20)
                    label = r'$B_z$ [nT]'
                elif color_key.lower() in ['by']:
                    c = self.B.T[1]
                    cm = mpl.colormaps['turbo']
                    nm = mpl.colors.Normalize(vmin=-20, vmax=20)
                    label = r'$B_y$ [nT]'
                elif color_key.lower() in ['bx']:
                    c = self.B.T[0]
                    cm = mpl.colormaps['turbo']
                    nm = mpl.colors.Normalize(vmin=-50, vmax=50)
                    label = r'$B_x$ [nT]'
                elif color_key.lower() in ['n', 'density']:
                    c = self.n
                    cm = mpl.colormaps['turbo']
                    nm = mpl.colors.Normalize(vmin=0, vmax=0.2)
                    label = r'n [cm$^{-3}$]'
                elif color_key.lower() == 'markers':
                    c = self.markers
                    cm = mpl.colormaps['Set1']
                    nm = mpl.colors.Normalize(vmin=0, vmax=10)
                    label = r'Field line markers'
                elif color_key.lower() == 'errors':
                    c = self.errors
                    cm = mpl.colormaps['turbo']
                    nm = mpl.colors.Normalize(vmin=0, vmax=30)
                    label = r'Tracing Error [%]'
                elif color_key.lower() in ['open', 'closed']:
                    closed = self.is_it_closed(selection=selection, z_intersect=z_intersect)
                    c = np.full_like(self.r.T[0], closed*0.5)
                    cm = mpl.colormaps['viridis']
                    nm = mpl.colors.Normalize(vmin=0, vmax=1.)
                    label = r'Open/Closed'
                else:
                    c = np.linalg.norm(self.B, axis=-1)
                    cm = mpl.colormaps['turbo']
                    nm = mpl.colors.Normalize(vmin=0, vmax=50)
                    label = r'$B_T$ [nT]'
        else:
            c = None
            cm = cmap
            nm = norm
            label = None

        if norm is None:
            norm = nm
        if cmap is None:
            cmap = cm
            if cmap is not None:
                cmap.set_over(cl.adjust_lightness(cmap(1.0), 1.1))
                cmap.set_under(cl.adjust_lightness(cmap(0.0), 1.1))
                # cmap.set_over((0, 0, 0, 0))

        if clip_on is None:
            clip_on = getattr(selection, 'clip_on', False)

        # points = self.r.reshape(-1, 1, 3)
        # segments = np.concatenate([points[:-1], points[1:]], axis=1)

        col = colored_line(ax,
                     self.r,
                     c=c,
                     color=color,
                     lw=lw,
                     ls='dashed',
                     alpha=alpha,
                     clip_on=False,
                     cmap=cm,
                     norm=nm,
                     zdir=zdir,
                     zorder=zorder,
                     path_effects=True,
                     path_effects_alpha=path_effects_alpha,
                     path_effects_lw=path_effects_lw,
                     path_effects_color=path_effects_color,
                     path_effects_type=path_effects_type,
                     draw=draw,
                    )

        if arrows:
            arrow_ind_1, arrow_ind_2 = self.find_arrow_indeces()
            if arrow_color is None and c is not None:
                color_B1 = cmap(norm(c[arrow_ind_1]))
                color_B2 = cmap(norm(c[arrow_ind_2]))
            else:
                color_B1 = 'white' if arrow_color is None else arrow_color
                color_B2 = 'white' if arrow_color is None else arrow_color
            r1 = self.r[arrow_ind_1]
            r2 = self.r[arrow_ind_2]
            B1 = self.B[arrow_ind_1]
            B2 = self.B[arrow_ind_2]
            B1 = r1 + B1/np.linalg.norm(B1)*0.5
            B2 = r2 + B2/np.linalg.norm(B2)*0.5
            arrow_B1 = Arrow3D(*zip(r1, B1), color=color_B1, ls='-', lw=lw,
                               alpha=arrow_alpha, mutation_scale=arrow_scale, clip_on=clip_on, zorder=zorder-1)
            arrow_B2 = Arrow3D(*zip(r2, B2), color=color_B2, ls='-', lw=lw,
                               alpha=arrow_alpha, mutation_scale=arrow_scale, clip_on=clip_on, zorder=zorder-1)
            ax.add_artist(arrow_B1)
            ax.add_artist(arrow_B2)

        if end_points:
            color_points = 'white' if color is None else color
            ax.scatter(*self.r.T[:,0], s=5, marker='o', color=color_points,
                       alpha=alpha, zorder=10, clip_on=clip_on)
            ax.scatter(*self.r.T[:,-1], s=5, marker='o', color=color_points,
                       alpha=alpha, zorder=10, clip_on=clip_on)
        if intersections:
            plot_intersections(ax,
                               data=self.r,
                               selection=selection,
                               z_intersect=z_intersect,
                               clip_on=clip_on,
                               alpha=intersection_alpha,
                               )

        if colorbar:
            fig = ax.get_figure()
            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                     shrink=1, aspect=30, extend='both', pad=0.02, extendfrac=0.02,
                     ax=ax, orientation='vertical', location='right', label=label)
        return col



def B_dip_func(R):
    R = np.array(R)
    B0 = -31200 # nT (Earth's field at magnetic equator)
    X, Y, Z = R
    r = np.linalg.norm(R)
    Bx = 3*B0*X*Z/r**5
    By = 3*B0*Y*Z/r**5
    Bz = B0*(3*Z**2 - r**2)/r**5
    return np.array([Bx, By, Bz])


def find_closed_surface(ax=None,
                        selection=None,
                        color_key='b',
                        cmap=None,
                        norm=None,
                        bcs=None,
                        z_intersect=None,
                        ):
    if selection is None:
        return
    if bcs is None:
        bcs = {'r_min': 3.,
               'r_max': 32.,
               'max_iter': 1e2,
               'verbose': False,}
    bcs = {'r_min': 3., # 2.8
           'r_max': 33.,
           'max_iter': 2e3,
           'verbose': False,}

    r_min = 5.0
    r = 5.0
    boundary_crossings = 3
    phi_list = np.linspace(0, 2*np.pi, 68, endpoint=False)
    # phi_list = np.linspace(np.pi/2, 1.5*np.pi, 26, endpoint=False)
    # seed_points = np.array([r*np.cos(phi), r*np.sin(phi), 0*np.ones_like(phi)]).T

    # seed_point = [df_stream['x'].iloc[-1], df_stream['y'].iloc[-1], df_stream['z'].iloc[-1]]
    # for seed_point in seed_points:
    fl_opts = dict(step=0.1,
                   # B_func=B_dip_func,
                   B_func=selection.f_B,
                   boundary_conditions=bcs)

    colatitudes = []
    def smart_boundary_finder(r, phi_list, fl_opts):
        boundary_fls = []
        for phi in phi_list:
            n_crossings = 0
            dir = 1
            step = 1
            fls = []
            while n_crossings < boundary_crossings:
                # print(f'phi = {phi:.2f}, r = {r:.2f}, n_crossings = {n_crossings}')
                seed_point = [r*np.cos(phi), r*np.sin(phi), 0]
                fl = fieldline(r=seed_point, **fl_opts)
                fl.trace()
                # print(f'length = {fl.length:.2f}')
                fls.append(fl)
                closed = fl.is_it_closed(z_intersect=0)
                if dir == 1 and fl.length > 1:
                    if not closed:
                        n_crossings += 1
                        step /= 2
                        dir *= -1
                if dir == -1 and fl.length > 1:
                    if closed:
                        n_crossings += 1
                        step /= 2
                        dir *= 1
                if fl.length < 1:
                    # n_crossings += 1
                    step = 1
                    dir = -1
                if r > 32 or r < 3:
                    n_crossings = boundary_crossings
                r += step * dir
            boundary_fls.append(fls[-1])
        return boundary_fls

    def dumb_boundary_finder(r_min, phi_list, fl_opts):
        boundary_fls = []
        for phi in phi_list:
            r = r_min
            fls = []
            closed = True
            while closed == True:
                seed_point = [r*np.cos(phi), r*np.sin(phi), 0]
                fl = fieldline(r=seed_point, **fl_opts)
                fl.trace()
                fls.append(fl)
                if fl.length > 1:
                    closed = fl.is_it_closed(z_intersect=0)
                else:
                    closed = True
                if r > 32 or r < 3:
                    closed = False
                r += 1
            boundary_fls.append(fls[-2])
        return boundary_fls

    fls = smart_boundary_finder(r, phi_list, fl_opts)
     # fls to a npy file
    for fl in fls:
        fl.B_func = None
        fl.n_func = None
    np.save(f'boundary-field-lines/boundary_{selection.cycle:06d}.npy', fls, allow_pickle=True)
    # save fls to hdf5 file
    # fls_df = pd.DataFrame([fl.__dict__ for fl in fls])

    # fls = dumb_boundary_finder(r, phi_list, fl_opts)
            # seed_point = [r*np.cos(phi), r*np.sin(phi), 0]
            # fl = fieldline(r=seed_point,
            #                step=0.2,
            #                B_func=selection.f_B,
            #                boundary_conditions=bcs)
            # fl.trace()
            # fls.append(fl)
        # fl = fls[-2]
        # fl = fls[-1]
        # fl = fls[0]
    for fl in fls:

        r0 = fl.radius[0]
        r1 = fl.radius[-1]
        colat0 = fl.colatitude[0]*180/np.pi
        colat1 = fl.colatitude[-1]*180/np.pi
        colatitudes.append(abs(colat0))
        colatitudes.append(abs(colat1))
        # print(f'At phi = {phi:.2f}, r0 = {r0:.2f}, th0 = {th0:.2f}, r1 = {r1:.2f}, th1 = {th1:.2f}')
        # print(f'At phi = {phi:.2f}, r0 = {r0:.2f}, colat0 = {colat0:.2f}, r1 = {r1:.2f}, colat1 = {colat1:.2f}')

        # print(f'At phi = {phi:.2f}, r = {r:.2f}, L = {fl.L:.2f}')
        fl_lw = 2.
        fl_color = 'white'
        z_intersect = 0
        fl.plot_fieldline(ax=ax,
                          color_key=color_key, # 'b', 'bz', 'n', 'errors', 'markers'
                          # color='white', # single color
                          # color=fl_color, # single color
                          # cmap=cmap, # overrides default colors
                          # norm=norm, # overrides default normalization
                          lw=fl_lw, # line width
                          alpha=0.7, # transparency
                          arrows=True, # add arrows to show field direction
                          end_points=True, # add circles to end points
                          colorbar=False if fl.id == 0 else False,
                          clip_on=False,
                          selection=selection,
                          intersections=True,
                          z_intersect=z_intersect,
                          )
    print(f'colatitude min = {np.min(colatitudes):.2f}, max = {np.max(colatitudes):.2f}')
    # 2000  , colat min = 38.50, max = 53.09, r=7.8
    # 50000,  colat min = 40.41, max = 54.55, r=7.9
    # 122000, colat min = 41.22, max = 55.66, r=8
    # 202500, colat min = 41.60, max = 55.94, r=8


if __name__ == "__main__":
    sim = ipic3D()
    cycle = 0
    selection_view = Selection(sim,
                               species    = 1,
                               cycle      = cycle,
                               min_phys   = [ -15, -14, -6],
                               max_phys   = [   0,   14,  6],
                               )

    # --- Field data directory
    selection_view.data_dir = '/Users/leo/DATA/'
    selection_view.clip_on = True # whether to draw objects outside axes

    # --- Load the field data (filename will be found by cycle)
    # B_fields, _ = load_fields(selection_view, keys=['Bx', 'By', 'Bz'], cut=False)
    # n_fields, _ = load_fields(selection_view, keys=['rho_0'], cut=False)

    # --- Rotate the field data to physical coordinates
    # B_fields = [np.transpose(_, (2, 0, 1)) for _ in B_fields]
    # Bx, By, Bz = code_to_phys_rot(B_fields, True)*sim.get_scale('B')
    # n = np.transpose(n_fields[0], (2, 0, 1))
    # n = np.abs(n) * sim.get_scale('n', 1) + 1e-12

    # --- Grid points for the field
    # x = np.linspace(sim.max_phys[0], sim.min_phys[0], Bx.shape[0])
    # y = np.linspace(sim.min_phys[1], sim.max_phys[1], By.shape[1])
    # z = np.linspace(sim.min_phys[2], sim.max_phys[2], Bz.shape[2])

    # Linear Field Interpolators for each component
    # f_Bx = RegularGridInterpolator((x, y, z), Bx, bounds_error=False)
    # f_By = RegularGridInterpolator((x, y, z), By, bounds_error=False)
    # f_Bz = RegularGridInterpolator((x, y, z), Bz, bounds_error=False)
    # f_N = RegularGridInterpolator((x, y, z), n, bounds_error=False)

    # Interpolation Function (gives field in nT)
    f_B = lambda x: B_dip_func(x)
    # f_B = lambda x: np.array([f_Bx(x)[0], f_By(x)[0], f_Bz(x)[0]])
    # f_n = lambda x: f_N(x)[0]

    # f_B = lambda x: B_dip_func(x)                 # Dipole field

    # --- Field line seed points
    seed_points = [
                    # [8, 0, 0],
                    # [0, 8, 0],
                    [-8, 3, 0],
                    # [0, -8, 0],
                    # [-19, 3, 0],
                    # [-20, 3, 0],
                    # [-22, 3, 0],
                    # [-25, -10, 0],
                    # [-25, -5, 0],
                    # [-25, -1, 0],
                    # [-25, 6, 0],
                    # [-25, 5, 0],
                    # [-25, 10, 0],
                    ]

    boundary_conditions = {'r_min': 3.,
                           'r_max': 50.,
                           'lower_bound': selection_view.min_phys,
                           'upper_bound': selection_view.max_phys,
                           'max_iter': 1e5,
                           }

    # --- Test the field interpolators
    # print(f'Bz_dip at equator = {B_dip_func([-5,0,0])[-1]}')
    # print(f'Bz_pic at equator = {f_Bz([-5,0,0])}')

    # --- Trace the field lines
    fls = []
    for s in seed_points:
        fl = fieldline(r=s,
                       step=0.1, # 0.05-0.2 for RK4, 0.01-0.05 for Euler
                       method='RK4',
                       backtrace=False,
                       B_func=f_B,
                       # n_func=f_n,
                       boundary_conditions=boundary_conditions)
        fl.trace()
        print(f'L (field line) = {fl.L:.2f}')
        fls.append(fl)

    # --- Plotting
    plt.style.use('dark_background')
    bg_color = cl.bg
    grid_color = 'white'

    mpl.rcParams.update({"axes.grid" : True,
                         "grid.color": grid_color,
                         "grid.linestyle":"--",
                         "grid.linewidth":0.5,
                         "grid.alpha":0.2,
                         "axes.facecolor": bg_color,
                         "figure.facecolor": bg_color,
                         # "axes.edgecolor": grid_color,
                         "font.size": 14,
                         # "font.family": "sans-serif",
                         # "font.sans-serif": ["Roboto"],
                         "figure.titlesize": 18,
                         "axes.titlesize": 18,
                         "axes.labelsize": 16,
                         "xtick.labelsize": 14,
                         "ytick.labelsize": 14,
                         "legend.fontsize": 14,
                         })

    # import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(15, 9),
                     # constrained_layout=True,
                     )

    ax = fig.add_subplot(111, projection='3d',
                          computed_zorder=False,
                         )
    configure_axes(ax,
                   selection=selection_view,
                   coord="phys",
                   draw_radii=[5,8],
                   color=grid_color,
                   dark_mode=True,
                   transparent=False,
                   lw=1,
                   ls='--',
                   alpha=0.6,
                   )

    plt.axis('off')
    # zoom = 0.6
    # ax.set_xbound(ax.get_xbound()[0]*zoom, ax.get_xbound()[1]*zoom)
    # ax.set_ybound(ax.get_ybound()[0]*zoom, ax.get_ybound()[1]*zoom)
    # ax.set_zbound(ax.get_zbound()[0]*zoom, ax.get_zbound()[1]*zoom)
    # ax.view_init(elev=25, azim=220, roll=0)

    cmap = mpl.colormaps['turbo']
    # cmap = bkr_extra_cmap # divergent (lightblue-blue-black-red-orange)
    norm = mpl.colors.Normalize(vmin=0, vmax=50)

    for fl in fls:
        print(f'Field line {fl.id}')
        fl.plot_fieldline(ax=ax,
                          color_key='b', # 'b', 'bz', 'n', 'errors', 'markers'
                          # color='white', # single color
                          # cmap=cmap, # overrides default colors
                          # norm=norm, # overrides default normalization
                          lw=3, # line width
                          # alpha=0.9, # transparency
                          arrows=True, # add arrows to show field direction
                          end_points=True, # add circles to end points
                          colorbar=True if fl.id == 0 else False,
                          clip_on=False,
                          zorder=10,
                          # selection=selection_view,
                          )
        fl.plot_perp_vectors(ax, color='red', scale=1)
    ax.legend()
    # plt.savefig('fieldlines.png', dpi=300, bbox_inches='tight')
    # ax.legend()
    plt.show()
