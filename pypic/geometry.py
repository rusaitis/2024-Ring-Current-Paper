import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.art3d import Path3DCollection
from matplotlib import patheffects
from matplotlib.legend_handler import HandlerLineCollection
from pypic.colors import cmap_from_color
import pandas as pd

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        default_dict = dict(arrowstyle='-|>', color='y',
                            mutation_scale=12,    # arrow head size
                            shrinkA=0, shrinkB=0, # don't shrink the arrow
                            )
        default_dict.update(kwargs)
        super().__init__((0,0), (0,0), *args, **default_dict)
        # super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

def draw_circle(ax=None, origin=None, radius=5, normal='z', n=40, draw=True,
                color='white', lw=1, ls='--', alpha=0.5, selection=None,
                clip_on=None,
                **kwargs):
    """ Draw a circle in 3D or 2D plane """
    if clip_on is None:
        if selection is None:
            clip_on = True
        else:
            clip_on = getattr(selection, 'clip_on', True)

    if ax is None:
        ax = plt.gca()
    if origin is None:
        origin = np.array([0,0,0])
    origin = np.asarray(origin)
    if ax.name != '3d':
        origin = origin[:2]

    theta = np.linspace(0, 2*np.pi, n)
    rc = [radius*np.cos(theta),
          radius*np.sin(theta),
          np.zeros(theta.shape)]
    rc = np.asarray(rc)
    if normal == 'y':
        rc = rc[[1, 2, 0], :]
    elif normal == 'x':
        rc = rc[[2, 0, 1], :]
    # print(f'ax name: {ax.name}')
    # print(f'shape: {rc.shape}')
    if ax.name != '3d':
        rc = rc[:-1] # reduce to 2D
        # print(f'shape: {rc.shape}')
    rc = origin + np.asarray(rc).T
    if selection is not None and ax.name == '3d':
        # clip the drawing to plot min and max
        rc[(rc > selection.max_phys)] = np.nan
        rc[(rc < selection.min_phys)] = np.nan
    if draw:
        rc = ax.plot(*rc.T, color=color, lw=lw, ls=ls, alpha=alpha,
                     clip_on=clip_on, **kwargs)
    return rc

def draw_3d_circle(ax, normal_vector, radius=1,
                   center=(0, 0, 0), num_points=100,
                   draw=True,
                   **kwargs):
    """
    Draw a 3D circle with a given normal vector using matplotlib.

    Parameters:
    normal_vector (tuple): A 3-tuple representing the normal vector of the plane.
    radius (float): The radius of the circle. Default is 1.
    center (tuple): A 3-tuple representing the center of the circle. Default is (0, 0, 0).
    num_points (int): Number of points to use for drawing the circle. Default is 100.
    """
    # Normalize the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # Create a random vector not in the same direction as the normal vector
    random_vector = np.random.rand(3)
    random_vector -= random_vector.dot(normal_vector) * normal_vector
    random_vector /= np.linalg.norm(random_vector)

    # Create a second vector orthogonal to both the normal vector and random_vector
    orthogonal_vector = np.cross(normal_vector, random_vector)

    # Parametric equations for the circle
    theta = np.linspace(0, 2*np.pi, num_points)
    x = center[0] + radius * (np.cos(theta) * random_vector[0] + np.sin(theta) * orthogonal_vector[0])
    y = center[1] + radius * (np.cos(theta) * random_vector[1] + np.sin(theta) * orthogonal_vector[1])
    z = center[2] + radius * (np.cos(theta) * random_vector[2] + np.sin(theta) * orthogonal_vector[2])
    if draw:
        artist = ax.plot(x, y, z, **kwargs)
        return artist
    else:
        return np.array([x, y, z])

def draw_cuboid(ax,
                scale,
                origin=None,
                draw=True,
                center=True,
                theta_x=0,
                theta_y=0,
                theta_z=0,
                path_effects=False,
                path_effects_alpha=0.8,
                path_effects_lw=1,
                path_effects_color='k',
                **kwargs):
    from itertools import combinations, product
    origin = np.zeros(3) if origin is None else origin
    r = [0, 1]
    pts = combinations(np.array(list(product(r, r, r))), 2)
    vertices = np.array(list(pts))
    # select the parallel edges (only one dimesion is different)
    vertices = vertices[np.sum(np.abs(vertices[:, 0] - vertices[:, 1]), axis=1) == 1]
    # scale the edges to the right size and add the origin
    vertices = origin + np.array(scale[:, None]).T * vertices
    if center:
        # vertices -= np.array(scale[:, None]).T / 2
        vertices = vertices - scale/2
    if draw:
        artists = []
        for v in vertices:
            # cube = ax.plot3D(*zip(*v), **kwargs)
            start, end = zip(v)
            start, end = np.array(start), np.array(end)
            # rotate vertices around x, y, z axis
            # start = start.dot(Rz(theta_z).T)
            # start = start.dot(Ry(theta_y).T)
            # start = start.dot(Rx(theta_x).T)
            # end = end.dot(Rz(theta_z).T)
            # end = end.dot(Ry(theta_y).T)
            # end = end.dot(Rx(theta_x).T)
            start, end = np.squeeze(start), np.squeeze(end)
            pts = list(zip(start, end))
            artist = ax.plot(*pts, **kwargs)
            if path_effects:
                lw = kwargs.get('linewidth', 1)
                for a in artist:
                    a.set(path_effects=[patheffects.withStroke(linewidth=lw+path_effects_lw,
                                                                    alpha=path_effects_alpha,
                                                                    foreground=path_effects_color)])
            artists.append(artist)
        return artists
    else:
        # rc = np.reshape(rc, (-1, 3)) # collapse the 1st dimension
        # ax.plot(*rc.T, color='y', linewidth=1, linestyle='--')
        pts = []
        for v in vertices:
            start, end = zip(v)
            start, end = np.array(start), np.array(end)
            start, end = np.squeeze(start), np.squeeze(end)
            pt = list(zip(start, end))
            pts.append(np.array(pt))
        return np.asarray(pts)
        # return vertices


def colored_line(ax=None,
                 data=None,
                 x=None,
                 y=None,
                 z=None,
                 c=None,
                 color=None,
                 lw=2,
                 ls='solid',
                 alpha=None,
                 clip_on=False,
                 cmap=None,
                 norm=None,
                 capstyle=None,
                 zdir='z',
                 zorder=None,
                 lightness_range=None,
                 alpha_range=None,
                 path_effects=False,
                 path_effects_alpha=0.6,
                 path_effects_lw=0.5,
                 path_effects_color='white',
                 path_effects_type='stroke',
                 draw=True,
                 **kwargs):
    import pandas as pd
    if ax is None:
        ax = plt.gca()

    # check if data is a pandas dataframe
    if isinstance(data, pd.DataFrame):
        x = 'x' if x is None else x
        y = 'y' if y is None else y
        z = 'z' if z is None else z
        c = 'energy' if c is None else c
        if c is None:
            try_columns = ['energy', 'speed', 'Bt', 'B', 'v', 'cycle']
            for col in try_columns:
                if col in data.columns:
                    c = col
                    break
        pts_x = data[x] if x in data.columns else None
        pts_y = data[y] if y in data.columns else None
        pts_z = data[z] if z in data.columns else None
        pts_c = data[c] if c in data.columns else None
    elif data is not None and data.ndim == 2:
        # print('data shape:', data.shape)
        # print('data dim:', data.ndim)
        # if data.ndim == 1: 
        # exit()
        pts_x = data[:,0]
        pts_y = data[:,1]
        pts_z = data[:,2]
        pts_c = c
    else:
        pts_x = x
        pts_y = y
        pts_z = z
        pts_c = c

    if ax.name == '3d':
        if pts_x is None or pts_y is None or pts_z is None:
            raise ValueError('x, y, and z cannot be None.')
    else:
        if pts_x is None or pts_y is None:
            raise ValueError('x and y cannot be None.')

    if zorder is None:
        if ax.name == '3d':
            if zdir == 'x':
                zorder = np.median(pts_x)
            elif zdir == 'y':
                zorder = np.median(pts_y)
            else:
                zorder = np.median(pts_z)
        else:
            zorder = 0

    if ax.name == '3d':
        points = np.array([pts_x, pts_y, pts_z]).T.reshape(-1, 1, 3)
    else:
        points = np.array([pts_x, pts_y]).T.reshape(-1, 1, 2)

    alpha = 1 if alpha is None else alpha
    if pts_c is None:
        pts_c = pts_z if ax.name == '3d' else pts_y
    if cmap is None:
        if color is not None:
            cmap = cmap_from_color(color=color,
                                   alpha=alpha,
                                   alpha_range=alpha_range,
                                   lightness_range=lightness_range
                                   )
        else:
            cmap = mpl.colormaps['turbo']
    else:
        if alpha is not None:
            cmap = cmap_from_color(color=None,
                                   cmap=cmap,
                                   alpha=alpha,
                                   )


    if norm is None:
        norm = mpl.colors.Normalize(vmin=pts_c.min(), vmax=pts_c.max())

    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # print(f'segments = {segments.shape}')
    if ax.name == '3d':
        lc = Line3DCollection(segments,
                              cmap=cmap,
                              norm=norm,
                              # color=color,
                              # antialiaseds=np.zeros(segments.shape[0]), #faster?
                              # antialiaseds=np.ones(segments.shape[0]),
                              linewidths=lw,
                              zorder=zorder,
                              linestyles=ls, # 'solid', 'dashed', 'dashdot', 'dotted'
                              # alpha=alpha,
                              )
    else:
        lc = LineCollection(segments,
                            cmap=cmap,
                            norm=norm,
                            # color=color,
                            linewidths=lw,
                            zorder=zorder,
                            linestyles=ls, # 'solid', 'dashed', 'dashdot', 'dotted'
                            # alpha=alpha,
                            )
    if capstyle is not None:
        lc.set_capstyle(capstyle)
    else:
        if alpha < 1:
            lc.set_capstyle('butt') # 'butt', 'round', 'projecting'
        else:
            lc.set_capstyle('round') # 'butt', 'round', 'projecting'

    # Whether to clip the line to the axes bounding box
    lc.set_clip_on(clip_on)

    if path_effects:
        if path_effects_type == 'stroke':
            lc.set(path_effects=[patheffects.withStroke(linewidth=lw+path_effects_lw,
                                                        alpha=path_effects_alpha,
                                                        foreground=path_effects_color)])
        elif path_effects_type == 'shadow':
            lc.set_path_effects([patheffects.withSimplePatchShadow(offset=(1, -1),
                                                                   shadow_rgbFace=path_effects_color,
                                                                   alpha=path_effects_alpha,
                                                                   rho=10.)])
        elif path_effects_type == 'new-line' and draw:
            if ax.name == '3d':
                ax.plot(pts_x, pts_y, pts_z,
                        color=path_effects_color,
                        lw=lw+path_effects_lw,
                        alpha=path_effects_alpha,
                        zorder=zorder-1)
            else:
                ax.plot(pts_x, pts_y,
                        color=path_effects_color,
                        lw=lw+path_effects_lw,
                        alpha=path_effects_alpha,
                        zorder=zorder-1)

        # lc.set(path_effects=[patheffects.withStroke(linewidth=lw+0.05, foreground=path_effects_color)])

    # lc.set_rasterized(True) # faster rendering? Doesn't change much

    # if color is None:
        # lc.set_array(pts_c)
    lc.set_array(pts_c)

    # Doesn't seem to do much for zorder of the points
    # lc.set_sort_zpos(np.min(pts_z))
    # ax.add_collection3d(lc, zs=pts_x, zdir='x')
    # ax.add_collection3d(lc, zs=pts_z, zdir='z')

    # Make a collection and return object
    if draw:
        if ax.name == '3d':
            line = ax.add_collection3d(lc, zdir=zdir)
        else:
            line = ax.add_collection(lc)
        # return line
    return lc


class HandlerColorLineCollection(HandlerLineCollection):
    """ Custom legend handler for LineCollection """
    def create_artists(self, legend, artist ,xdescent, ydescent,
                        width, height, fontsize,trans):
        x = np.linspace(0,width,self.get_numpoints(legend)+1)
        y = np.zeros(self.get_numpoints(legend)+1)+height/2.-ydescent
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=artist.cmap,
                     transform=trans)
        lc.set_array(x)
        lc.set_linewidth(artist.get_linewidth())
        return [lc]

def find_intersections(data=None,
                       x=None,
                       y=None,
                       z=None,
                       selection=None,
                       z_intersect=None,
                       ):
    """ Find path intersections with planes """
    if selection is None and z_intersect is None:
        return [], []
    selection = np.atleast_1d(selection)

    # check if data is a pandas dataframe
    if isinstance(data, pd.DataFrame):
        x = 'x' if x is None else x
        y = 'y' if y is None else y
        z = 'z' if z is None else z
        pts_x = data[x] if x in data.columns else None
        pts_y = data[y] if y in data.columns else None
        pts_z = data[z] if z in data.columns else None
        # positions = data[[x, y, z]].values
    elif data is not None and data.ndim == 2:
        pts_x = data[:,0]
        pts_y = data[:,1]
        pts_z = data[:,2]
    else:
        pts_x = x
        pts_y = y
        pts_z = z
    if pts_x is None or pts_y is None or pts_z is None:
        raise ValueError('x, y, and z cannot be None to find intersections.')
    if np.any(np.isnan(pts_x)) or np.any(np.isnan(pts_y)) or np.any(np.isnan(pts_z)):
        print('Positions have nan values.')
        return [], []
    positions = np.array([pts_x, pts_y, pts_z]).T

    intersects = []
    zdirs = []
    for s in selection:
        if z_intersect is None:
            cut_axis = s.cut_axis_phys
            plane_pos = s.center_phys[cut_axis]
        else:
            cut_axis = 2
            plane_pos = z_intersect
        dist_to_plane = positions.T[cut_axis] - plane_pos
        sign = np.sign(dist_to_plane)
        sign_changes= np.where((np.roll(sign, -1)[:-1] - sign[:-1]) != 0)[0]
        for i in sign_changes[:-1]:
            r0 = positions[i]
            r1 = positions[i+1]
            dr = r1 - r0
            # linear interpolation to find the intersection with the plane
            t_cross = (plane_pos - r0[cut_axis])/dr[cut_axis]
            r_new = r0 + dr*t_cross
            zdir = np.zeros(3)
            zdir[cut_axis] = 1
            visible = s.includes(r_new) if z_intersect is None else True
            if visible:
                intersects.append(r_new)
                zdirs.append(zdir)
                # print(f'Intersection at {r_new}')
    return intersects, zdirs
