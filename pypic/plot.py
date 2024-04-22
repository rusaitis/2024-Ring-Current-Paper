import os
import sys
import pypic
from pypic.colors import *
from cycler import cycler

import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
# from geometry import *
from pypic.geometry import *
from pypic.fields import *
from matplotlib import patheffects

# mpl.use('Agg')
# mpl.use('TkAgg') # slow
# mpl.use('TkCairo') # slow
# mpl.use('Qt5Agg') # fast
# mpl.rcParams['lines.linewidth'] = 2
# mpl.rcParams['lines.linestyle'] = '--'

mpl.rcParams['axes.prop_cycle'] = cycler(color=['r', 'g', 'b', 'y'])

# set rc params for xaxis pane fill
# mpl.rcParams['axes.xaxis.pane.fill'] = False
# ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
# plt.rcParams["figure.figsize"] = [14.5,10]
# plt.rcParams.update({"font.family": "Fira Sans Compressed"})


# # turn off axis background color
# ax.xaxis.pane.fill = False
# ax.yaxis.pane.fill = False
# ax.zaxis.pane.fill = False
# # set axes line color to grey
# ax.xaxis.pane.set_edgecolor('grey')
# ax.yaxis.pane.set_edgecolor('grey')
# ax.zaxis.pane.set_edgecolor('grey')

# plt.plot(data)  # first color is red

def configure_axes(ax=None,
                   selection=None,
                   coord='phys',
                   draw_radii=None,
                   planet=False,
                   axis_lines=True,
                   grid_lines=True,
                   lw=1,
                   ls='--',
                   color=None,
                   dark_mode=True,
                   transparent = True,
                   alpha=0.4,
                   x_lines=None,
                   y_lines=None,
                   z_lines=None,
                   x_minor_lines=None,
                   y_minor_lines=None,
                   z_minor_lines=None,
                   x_label=None,
                   y_label=None,
                   z_label=None,
                   labels=True,
                   labelpad=1,
                   minor_labels=False,
                   planet_zorder=0,
                   clip_on=False,
                   zoom=0.8,
                   fontsize=12,
                   tick_loc=None,
                   label_loc=None,
                   tick_marks=None,
                   title=None,
                   zorder=None,
                   ):
    if ax is None:
        ax = plt.gca()
    if color is None:
        color = 'white' if dark_mode else 'black'
    tick_color = mpl.colors.to_rgba(color, alpha=alpha)
    if x_label is None:
        x_label = r'$X$'
        y_label = r'$Y$'
        z_label = r'$Z$'

    if tick_loc is None:
        tick_loc = ['left', 'bottom'],
    if label_loc is None:
        label_loc = ['left', 'bottom'],
    if tick_marks is None:
        tick_marks = ['left', 'right', 'bottom', 'top'],
    # ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
    # ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
    if planet:
        if selection.includes((0,0,0)):
            draw_planet(ax,
                        wireframe=True,
                        res='low',
                        dark_mode=dark_mode,
                        alpha=alpha,
                        zorder=planet_zorder,
                        clip_on=clip_on,
                        )
    if axis_lines and selection is not None:
        plot_axis_lines(ax,
                        selection,
                        color=color,
                        lw=lw,
                        ls=ls,
                        alpha=alpha,
                        x_label=x_label,
                        y_label=y_label,
                        z_label=z_label,
                        clip_on=clip_on,
                        fontsize=fontsize,
                        zorder=zorder,
                        )
    if grid_lines and selection is not None:
        plot_grid_lines(ax,
                        selection,
                        color=color,
                        lw=lw,
                        ls=ls,
                        alpha=alpha,
                        x_lines=x_lines,
                        y_lines=y_lines,
                        z_lines=z_lines,
                        x_minor_lines=x_minor_lines,
                        y_minor_lines=y_minor_lines,
                        z_minor_lines=z_minor_lines,
                        labels=labels,
                        minor_labels=minor_labels,
                        clip_on=clip_on,
                        fontsize=fontsize,
                        zorder=zorder,
                        )

    import pypic.colors as cl
    
    if dark_mode:
        bg = cl.bg
    else:
        bg = 'white'

    if transparent:
        bg = [0, 0, 0, 0]
        ax.set_facecolor(bg)
    else:
        ax.set_facecolor(bg)

        # plt.gcf().set_facecolor(bg)
    if draw_radii is not None:
        for r in draw_radii:
            rc = draw_circle(ax, [0,0,0], r, normal='z', n=50, draw=True,
                             selection=selection, lw=lw, color=color, clip_on=clip_on)

    if  ax.name == "3d":
        ax.axis('off')
        # ax.set_proj_type('ortho')
        if selection is not None:
            ax.set_xlim(selection.min_phys[0], selection.max_phys[0])
            ax.set_ylim(selection.min_phys[1], selection.max_phys[1])
            ax.set_zlim(selection.min_phys[2], selection.max_phys[2])
        xrange = (ax.get_xbound()[1] - ax.get_xbound()[0])*(1-zoom)/2
        yrange = (ax.get_ybound()[1] - ax.get_ybound()[0])*(1-zoom)/2
        zrange = (ax.get_zbound()[1] - ax.get_zbound()[0])*(1-zoom)/2
        ax.set_xbound(ax.get_xbound()[0]+xrange, ax.get_xbound()[1]-xrange)
        ax.set_ybound(ax.get_ybound()[0]+yrange, ax.get_ybound()[1]-yrange)
        ax.set_zbound(ax.get_zbound()[0]+zrange, ax.get_zbound()[1]-zrange)
        # ax.set_xbound(ax.get_xbound()[0]*zoom, ax.get_xbound()[1]*zoom)
        # ax.set_ybound(ax.get_ybound()[0]*zoom, ax.get_ybound()[1]*zoom)
        # ax.set_zbound(ax.get_zbound()[0]*zoom, ax.get_zbound()[1]*zoom)

        # Test rots
        # rc3 = draw_circle(ax, [0,0,0], 5, normal='y', n=22, draw=False)
        # theta = 40 * np.pi / 180
        # rc3 = rc3.dot(Rz(theta).T)
        # rc3 = rc3.dot(Ry(theta).T)
        # ax.plot(*rc3.T, color='y', linewidth=1, linestyle='--')

        # turn off axis background color
        # ax.xaxis.pane.fill = False
        # ax.yaxis.pane.fill = False
        # ax.zaxis.pane.fill = False
        # set axes line color to grey
        # ax.xaxis.pane.set_edgecolor('grey')
        # ax.yaxis.pane.set_edgecolor('grey')
        # ax.zaxis.pane.set_edgecolor('grey')
        #set axis grid color to grey
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)

        #equal aspect
        ax.set_aspect("equal")
        ax.invert_xaxis()
        ax.invert_yaxis()
    # 2D
    else:
        if selection is not None:
            if selection.cut_axis_phys == 2:
                ax.set_xlim(selection.min_phys[0], selection.max_phys[0])
                ax.set_ylim(selection.min_phys[1], selection.max_phys[1])
            elif selection.cut_axis_phys == 1:
                ax.set_xlim(selection.min_phys[0], selection.max_phys[0])
                ax.set_ylim(selection.min_phys[2], selection.max_phys[2])
            else:
                ax.set_xlim(selection.min_phys[1], selection.max_phys[1])
                ax.set_ylim(selection.min_phys[2], selection.max_phys[2])


        if x_label is not None:
            ax.set_xlabel(x_label, labelpad=labelpad)
        if y_label is not None:
            ax.set_ylabel(y_label, labelpad=labelpad)

        # Set the axes ticks
        ax.tick_params(axis='both', which='both',
                       left='left' in tick_marks,
                       right='right' in tick_marks,
                       top='top' in tick_marks,
                       bottom='bottom' in tick_marks,
                       direction='in', length=5, width=1.5,
                       color=tick_color,
                       # labelcolor=color,
                       # pad=5,
                       labelleft='left' in tick_loc,
                       labelright='right' in tick_loc,
                       labelbottom='bottom' in tick_loc,
                       labeltop='top' in tick_loc,
                       # labelsize=fontsize,
                       )

        # Set the axes label positions
        if 'left' in label_loc:
            ax.yaxis.set_label_position("left")
        if 'bottom' in label_loc:
            ax.xaxis.set_label_position("bottom")
        if 'right' in label_loc:
            ax.yaxis.set_label_position("right")
        if 'top' in label_loc:
            ax.xaxis.set_label_position("top")
        if 'bottom' not in label_loc and 'top' not in label_loc:
            ax.set_xlabel(None)
        if 'left' not in label_loc and 'right' not in label_loc:
            ax.set_ylabel(None)

        # Disable the black border around the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        if title is not None:
            ax.set_title(title, fontsize=fontsize+2)

        # Invert x and y axes for GSM
        if coord == 'phys':
            ax.invert_xaxis()
            ax.invert_yaxis()

        # Set Aspect Equal
        if coord == 'phys':
            ax.set_aspect("equal")


def configure_matplotlib(dark_mode, transparent):
    import pypic.colors as cl
    if dark_mode:
        plt.style.use('dark_background')
        bg_color = cl.bg
        grid_color = 'white'
    else:
        bg_color = 'white'
        grid_color = 'black'
    if transparent:
        # cl.bg = (0, 0, 0, 0)
        bg_color = (0, 0, 0, 0)
    mpl.rcParams.update({"axes.grid" : True,
                         "grid.color": grid_color,
                         "grid.linestyle":"--",
                         "grid.linewidth":0.5,
                         "grid.alpha":0.2,
                         "axes.facecolor": bg_color,
                         "figure.facecolor": bg_color,
                         "font.size": 14,
                         # "font.family": "sans-serif",
                         # "font.sans-serif": ["Roboto"],
                         # rcParams["axes.facecolor"] = bg_color,
                         # rcParams["legend.edgecolor"] = bg_color,
                         # rcParams["legend.framealpha"] = 0.5,
                         # rcParams["legend.fancybox"] = True,
                         "figure.titlesize": 14,
                         "axes.titlesize": 14,
                         "axes.labelsize": 12,
                         "xtick.labelsize": 12,
                         "ytick.labelsize": 12,
                         "legend.fontsize": 12,
                         })

def color_norm(range, log=False, min=1e-3):
    """ Return a color normalization object """
    if log:
        range[0] = min if range[0] <= 0 else range[0]
        norm = mpl.colors.LogNorm(vmin=range[0], vmax=range[1])
    else:
        norm = mpl.colors.Normalize(vmin=range[0], vmax=range[1])
    return norm

def draw_ring_current(ax=None, color='yellow', R=5, r=2, N=1000, s=2, zorder=1,
                      alpha=1):
    if ax is None:
        ax = plt.gca()

    # Parameters for the torus
    # R = 4  # Distance from center to the torus center
    # r = 2  # Radius of the tube
    # if origin is None:
        # origin = np.array([0,0,0])
    # origin = np.asarray(origin)
    # if ax.name != '3d':
    #     origin = origin[:2]

    # Number of points
    # N = 1000
    # Randomly sample the angles
    theta = 2 * np.pi * np.random.rand(N)
    phi = 2 * np.pi * np.random.rand(N)
    r = r * np.random.rand(N)

    # Convert the angles to Cartesian coordinates
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)

    rs = np.sqrt((x+R*np.cos(np.pi/4))**2 + (y-R*np.sin(np.pi/4))**2 + z**2)
    a = 1.-rs/8.
    a[a<0] = 0

    # add alpha to color
    # color = mpl.colors.to_rgba(color, alpha=alpha)
    # from matplotlib.colors import to_rgb, to_rgba
    # r, g, b = to_rgb(color)
    # r, g, b, _ = to_rgba(color)
    # color = [(r, _, b, _) for _ in a]
    s=s*a

    # Plotting
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, s=s, color=color, zorder=zorder, alpha=alpha)


    # theta = np.linspace(0, 2*np.pi, n)
    # rc = [radius*np.cos(theta),
    #       radius*np.sin(theta),
    #       np.zeros(theta.shape)]
    # rc = np.asarray(rc)
    # if normal == 'y':
    #     rc = rc[[1, 2, 0], :]
    # elif normal == 'x':
    #     rc = rc[[2, 0, 1], :]
    # # print(f'ax name: {ax.name}')
    # # print(f'shape: {rc.shape}')
    # if ax.name != '3d':
    #     rc = rc[:-1] # reduce to 2D
    #     # print(f'shape: {rc.shape}')
    # rc = origin + np.asarray(rc).T
    # if selection is not None and ax.name == '3d':
    #     # clip the drawing to plot min and max
    #     rc[(rc > selection.max_phys)] = np.nan
    #     rc[(rc < selection.min_phys)] = np.nan
    # if draw:
    #     rc = ax.plot(*rc.T, color=color, lw=lw, ls=ls, alpha=alpha, **kwargs)
    return sc


def plot_grid_lines(ax=None,
                    selection=None,
                    color='white',
                    lw=1,
                    ls='--',
                    alpha=0.6,
                    scale=1.0,
                    labels=True,
                    minor_labels=False,
                    x_lines=None,
                    y_lines=None,
                    z_lines=None,
                    x_minor_lines=None,
                    y_minor_lines=None,
                    z_minor_lines=None,
                    offset=0.7,
                    clip_on=None,
                    zorder=None,
                    fontsize=12,
                    ):
    if ax is None:
        ax = plt.gca()
    if clip_on is None:
        if selection is None:
            clip_on = True
        else:
            clip_on = getattr(selection, 'clip_on', True)

    s = selection
    zorder = zorder if zorder is not None else int(s.center_phys[1])

    opt = dict(color=color,
               alpha=alpha*2/3,
               zorder=zorder,
               clip_on=clip_on,
               fontsize=fontsize,
               horizontalalignment='center',
               verticalalignment='center',
               rotation=0,
               )
    major_opt = dict(color=color, lw=lw*1, ls=ls, alpha=alpha/2, zorder=zorder, clip_on=clip_on)
    minor_opt = dict(color=color, lw=lw/2, ls=ls, alpha=alpha/4, zorder=zorder, clip_on=clip_on)
    label_offset = 2.5*offset
    artists = []
    if ax.name == "3d":
        if x_lines is not None:
            for x in x_lines:
                x0 = [x, s.min_phys[1]-offset, 0]
                x1 = [x, s.max_phys[1]+offset, 0]
                if selection.includes((np.array(x0)+np.array(x1))/2):
                    xaxis = ax.plot(*np.array([x0, x1]).T, **major_opt)
                    artists.append(xaxis)
                    if labels:
                        ax.text(x, s.max_phys[1]+label_offset, 0, r'''${x}$'''.format(x=int(x)), **opt)
        if x_minor_lines is not None:
            for x in x_minor_lines:
                x0 = [x, s.min_phys[1]-offset, 0]
                x1 = [x, s.max_phys[1]+offset, 0]
                if selection.includes((np.array(x0)+np.array(x1))/2):
                    xaxis = ax.plot(*np.array([x0, x1]).T, **minor_opt)
                    artists.append(xaxis)
                    if minor_labels:
                        ax.text(x, s.max_phys[1]+label_offset, 0, r'''${x}$'''.format(x=int(x)), **opt)
        if y_lines is not None:
            for y in y_lines:
                y0 = [s.min_phys[0]-offset, y, 0]
                y1 = [s.max_phys[0]+offset, y, 0]
                if selection.includes((np.array(y0)+np.array(y1))/2):
                    major_opt.update(alpha=alpha/4)
                    yaxis = ax.plot(*np.array([y0, y1]).T, **major_opt)
                    artists.append(yaxis)
                    if labels:
                        ax.text(s.min_phys[0]-label_offset, y, 0, r'''${y}$'''.format(y=int(y)), **opt)
        if y_minor_lines is not None:
            for y in y_minor_lines:
                y0 = [s.min_phys[0]-offset, y, 0]
                y1 = [s.max_phys[0]+offset, y, 0]
                if selection.includes((np.array(y0)+np.array(y1))/2):
                    minor_opt.update(alpha=alpha/4)
                    yaxis = ax.plot(*np.array([y0, y1]).T, **minor_opt)
                    artists.append(yaxis)
                    if minor_labels:
                        ax.text(s.min_phys[0]-label_offset, y, 0, r'''${y}$'''.format(y=int(y)), **opt)
        if z_lines is not None:
            for z in z_lines:
                z0 = [s.min_phys[0]-offset, 0, z]
                z1 = [s.max_phys[0]+offset, 0, z]
                if selection.includes((np.array(z0)+np.array(z1))/2):
                    zaxis = ax.plot(*np.array([z0, z1]).T, **major_opt)
                    artists.append(zaxis)
                    if labels:
                        ax.text(s.min_phys[0]-label_offset, 0, z, r'''${z}$'''.format(z=int(z)), **opt)
        if z_minor_lines is not None:
            for z in z_minor_lines:
                z0 = [s.min_phys[0]-offset, 0, z]
                z1 = [s.max_phys[0]+offset, 0, z]
                if selection.includes((np.array(z0)+np.array(z1))/2):
                    zaxis = ax.plot(*np.array([z0, z1]).T, **minor_opt)
                    artists.append(zaxis)
                    if minor_labels:
                        ax.text(s.min_phys[0]-label_offset, 0, z, r'''${z}$'''.format(z=int(z)), **opt)

    else:
        center = [0, 0]
        xaxis = ax.axhline(y = center[0], color = color, lw=lw, ls=ls, alpha=alpha)
        yaxis = ax.axvline(x = center[1], color = color, lw=lw, ls=ls, alpha=alpha)
        artists = (xaxis, yaxis)
    return artists


def plot_axis_lines(ax=None,
                    selection=None,
                    selection_position=False,
                    color='white',
                    lw=1,
                    ls='--',
                    alpha=0.6,
                    scale=1.0,
                    labels=True,
                    x_label=r'$X$',
                    y_label=r'$Y$',
                    z_label=r'$Z$',
                    offset=0.7,
                    clip_on=None,
                    zorder=None,
                    fontsize=12,
                    ):
    if ax is None:
        ax = plt.gca()
    if clip_on is None:
        if selection is None:
            clip_on = True
        else:
            clip_on = getattr(selection, 'clip_on', True)

    s = selection
    zorder = zorder if zorder is not None else int(s.center_phys[1])

    opt = dict(color=color,
               alpha=alpha*2/3,
               zorder=zorder,
               clip_on=clip_on,
               fontsize=fontsize,
               horizontalalignment='center',
               verticalalignment='center',
               rotation=0,
               )
    major_opt = dict(color=color, lw=lw*1, ls=ls, alpha=alpha/2, zorder=zorder, clip_on=clip_on)
    minor_opt = dict(color=color, lw=lw/2, ls=ls, alpha=alpha/4, zorder=zorder, clip_on=clip_on)
    label_offset = 2.5*offset
    if ax.name == "3d":
        if selection_position:
            if 0 > selection.max_phys[0]:
                xmin = selection.max_phys[0]
            elif 0 < selection.min_phys[0]:
                xmin = selection.min_phys[0]
            else:
                xmin = 0
            if 0 > selection.max_phys[1]:
                ymin = selection.max_phys[1]
            elif 0 < selection.min_phys[1]:
                ymin = selection.min_phys[1]
            else:
                ymin = 0
            if 0 > selection.max_phys[2]:
                zmin = selection.max_phys[2]
            elif 0 < selection.min_phys[2]:
                zmin = selection.min_phys[2]
            else:
                zmin = 0
            x0 = [xmin,                     selection.center_phys[1], 0]
            x1 = [selection.center_phys[0], selection.center_phys[1], 0]
            y0 = [selection.center_phys[0], ymin,                     0]
            y1 = [selection.center_phys[0], selection.center_phys[1], 0]
            z0 = [selection.center_phys[0], selection.center_phys[1], zmin]
            z1 = [selection.center_phys[0], selection.center_phys[1], selection.center_phys[2]]
            xa = ax.plot(*np.array([x0, x1]).T, **major_opt)
            ya = ax.plot(*np.array([y0, y1]).T, **major_opt)
            za = ax.plot(*np.array([z0, z1]).T, **major_opt)
        else:
            x0 = [selection.min_phys[0], 0, 0]
            x1 = [selection.max_phys[0]+offset*3.5, 0, 0]
            y0 = [0, selection.min_phys[1], 0]
            y1 = [0, selection.max_phys[1]+offset, 0]
            z0 = [0, 0, selection.min_phys[2]]
            z1 = [0, 0, selection.max_phys[2]+offset]

            # xaxis = ax.plot(*np.array([x0, x1]).T, color=color, lw=lw, ls=ls, zorder=zorder)
            # yaxis = ax.plot(*np.array([y0, y1]).T, color=color, lw=lw, ls=ls, zorder=zorder)
            # zaxis = ax.plot(*np.array([z0, z1]).T, color=color, lw=lw, ls=ls, zorder=zorder)
            # artists = (xaxis, yaxis, zaxis)
            xa, ya, za = None, None, None
            if selection.includes((np.array(x0)+np.array(x1))/2):
                xa = Arrow3D(*zip(x0, x1), color=color, ls=ls, lw=lw, alpha=alpha, clip_on=clip_on)
                ax.add_artist(xa)
                if labels:
                    ax.text(selection.max_phys[0]+label_offset*2.5*offset, 0, 0, x_label, **opt)
            if selection.includes((np.array(y0)+np.array(y1))/2):
                ya = Arrow3D(*zip(y0, y1), color=color, ls=ls, lw=lw, alpha=alpha, clip_on=clip_on)
                ax.add_artist(ya)
                if labels:
                    ax.text(0, selection.max_phys[1]+label_offset*offset, 0, y_label, **opt)
            if selection.includes((np.array(z0)+np.array(z1))/2):
                za = Arrow3D(*zip(z0, z1), color=color, ls=ls, lw=lw, alpha=alpha, clip_on=clip_on)
                ax.add_artist(za)
                if labels:
                    ax.text(0, 0, selection.max_phys[2]+label_offset*offset, z_label, **opt)

        artists = (xa, ya, za)

    else:
        center = [0, 0]
        xaxis = ax.axhline(y = center[0], color = color, lw=lw, ls=ls, alpha=alpha)
        yaxis = ax.axvline(x = center[1], color = color, lw=lw, ls=ls, alpha=alpha)
        artists = (xaxis, yaxis)
    return artists


def draw_particle_vectors(ax,
                          df,
                          all_positions=True,
                          min_length=1.,
                          max_length=5,
                          alpha_max=0.9,
                          zorder=100,
                          norm_B=10,
                          norm_E=2,
                          lw=3,
                          color_B=None,
                          color_E=None,
                          fontsize=13,
                          ):
    if ax is None:
        ax = plt.gca()
    # rc = draw_circle(ax, [0,0,0], r, normal='z', n=50, draw=True,
    #                  selection=selection, lw=lw, color=color)

    color, ls, alpha = 'b', '-', 0.8
    color_B = color_B if color_B is not None else '#504AFC'
    color_E = color_E if color_E is not None else 'yellow'

    if all_positions:
        for i in range(len(df['x'])):
            x,y,z = df['x'][i], df['y'][i], df['z'][i]
            Bx, By, Bz = df['Bx'][i], df['By'][i], df['Bz'][i]
            Ex, Ey, Ez = df['Ex'][i], df['Ey'][i], df['Ez'][i]
            r0 = [x, y, z]
            r1_B = [x + Bx/norm_B, y + By/norm_B, z + Bz/norm_B]
            r1_E = [x + Ex/norm_E, y + Ey/norm_E, z + Ez/norm_E]
            arrow_B = Arrow3D(*zip(r0, r1_B), color=color_B, ls=ls, lw=lw, alpha=alpha)
            arrow_E = Arrow3D(*zip(r0, r1_E), color=color_E, ls=ls, lw=lw, alpha=alpha)
            ax.add_artist(arrow_B)
            ax.add_artist(arrow_E)
    else:
        opt = dict(
                   # color=color,
                   zorder=zorder,
                   clip_on=False,
                   fontsize=fontsize,
                   fontweight='bold',
                   horizontalalignment='center',
                   verticalalignment='center',
                   rotation=0,
                   )
        import pypic.colors as cl
        alpha_B = alpha_max
        alpha_E = alpha_max
        lw_B = lw
        lw_E = lw
        r0 = np.asarray([df['x'].iloc[-1], df['y'].iloc[-1], df['z'].iloc[-1]])
        B = np.asarray([df['Bx'].iloc[-1], df['By'].iloc[-1], df['Bz'].iloc[-1]])
        E = np.asarray([df['Ex'].iloc[-1], df['Ey'].iloc[-1], df['Ez'].iloc[-1]])
        B_unit = B/np.linalg.norm(B)
        E_unit = E/np.linalg.norm(E)
        B_show = B/norm_B
        E_show = E/norm_E
        max_length_for_color = max_length*5

        if np.linalg.norm(B_show) > max_length:
            B_show = B_unit*max_length
            color_decrease = 1. - (np.linalg.norm(B_show)/max_length_for_color)
            color_decrease = 0. if color_decrease < 0 else color_decrease
            color_B = cl.adjust_lightness(color_B, color_decrease)
        if np.linalg.norm(E_show) > max_length:
            E_show = E_unit*max_length
            color_decrease = 1. - (np.linalg.norm(E_show)/max_length_for_color)
            color_decrease = 0. if color_decrease < 0 else color_decrease
            color_E = cl.adjust_lightness(color_E, color_decrease)
        if np.linalg.norm(B_show) < min_length:
            # alpha_B = alpha_max * (np.linalg.norm(B_show)/min_length)
            lw_B = lw * (np.linalg.norm(B_show)/min_length)
        # else:
            # alpha_B = alpha_max
        if np.linalg.norm(E_show) < min_length:
            # alpha_E = alpha_max * (np.linalg.norm(E_show)/min_length)
            lw_E = lw * (np.linalg.norm(E_show)/min_length)
        # else:
            # alpha_E = alpha_max
        r1_B = r0 + B_show
        r1_E = r0 + E_show
        from matplotlib.patheffects import withStroke
        pe = [withStroke(linewidth=4.8, foreground='black', alpha=0.9)]
        arrow_B = Arrow3D(*zip(r0, r1_B), color=color_B, ls=ls, lw=lw_B, alpha=alpha_B, clip_on=False, zorder=zorder, path_effects=pe)
        arrow_E = Arrow3D(*zip(r0, r1_E), color=color_E, ls=ls, lw=lw_E, alpha=alpha_E, clip_on=False, zorder=zorder, path_effects=pe)
        ax.add_artist(arrow_B)
        ax.add_artist(arrow_E)
        label_offset = 0.3
        label_offset = np.asarray(label_offset)
        labels = True
        # labels = False
        label_B = r1_B+B_unit*label_offset
        label_E = r1_E+E_unit*label_offset
        # opt['alpha'] = alpha
        if labels:
            # ax.text(*label_B, r'$\mathbf{\vec{B}}$', color=color_B, alpha=alpha_B, **opt)
            # ax.text(*label_E, r'$\mathbf{\vec{E}}$', color=color_E, alpha=alpha_E, **opt)
            from matplotlib.patheffects import withStroke
            pe = [withStroke(linewidth=1, foreground='black', alpha=0.5)]
            ax.text(*label_B, r'$\vec{B}$', color=color_B, path_effects=pe, **opt)
            ax.text(*label_E, r'$\vec{E}$', color=color_E, path_effects=pe, **opt)


def test_imshow(X, selection=None, cmap=None, norm=None, range=None):
    plt.style.use('dark_mode')
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111)
    if selection is None:
        extent = None
    else:
        extent = selection.range_phys[:2, ::-1].flatten()
    if range is not None:
        norm = mpl.colors.Normalize(vmin=range[0], vmax=range[1])
    im = ax.imshow(X.T,
                      cmap=cmap,
                      norm=norm,
                      aspect='equal',
                      interpolation='lanczos',
                      origin='upper',
                      extent=extent,
                      )
    cbar = fig.colorbar(im, ax=ax, pad=0.02,
                 shrink=1, location='right',
                 aspect=30, extend='both', extendfrac=0.02, )
    configure_axes(ax, selection=selection, coord="phys", radii=[5,8,28])
    plt.show()



def streamlines(ax, field,
                selection, # selection object for bins and limits
                color=None, # array or color
                scale=1, # reference size
                density=3, # float or (float, float)
                lw=3, # linewidth of streamlines
                lw_min=0.4, # minimum linewidth
                lw_max=2, # maximum linewidth
                cmap=None,
                norm=None,
                stream_seed_points=None, # array of points to seed streamlines
                broken_streamlines=True,
                integration_direction='both',
                smooth_field=False,
                average_cells=2,
                seed_point_size=10,
                seed_point_color='blue',
                seed_point_marker='o',
                arrowsize=1.1,
                arrowstyle='-|>',
                alpha=1,
                zorder=50,
                patheffects_color=None,
                patheffects_alpha=0.15,
                patheffects_lw=0.5,
                patheffects_type='stroke',
                cbar=True,
                cbar_loc='bottom left',
                cbar_alpha=0.8,
                label='',
                minlength=0.01,
                maxlength=2,
                dark_mode=True,
                ):
    if color is None:
        color = 'k'
    if field is not None:
        sx, sy = field
        if smooth_field:
            sx = reduce_field(sx, average_cells)
            sy = reduce_field(sy, average_cells)
            if np.size(color) > 4:
                color = reduce_field(color, average_cells)
                # color = np.sqrt(sx**2 + sy**2)
        st = np.sqrt(sx**2 + sy**2)
        x, y = selection.xy_bins(sx)
        if stream_seed_points is not None:
            ax.scatter(stream_seed_points.T[0],
                    stream_seed_points.T[1],
                    s=seed_point_size,
                    color=seed_point_color,
                    marker=seed_point_marker,
                    )

        if np.size(color) > 4:
            color = np.fliplr(color.T)
        linewidth = st/scale*lw
        linewidth[linewidth < lw_min] = lw_min
        linewidth[linewidth > lw_max] = lw_max
        # stream plot expects x and y to be increasing, so we need to flip
        im = ax.streamplot(np.fliplr(x), y,
                           np.fliplr(sx.T), np.fliplr(sy.T),
                           density=density,
                           color=color,
                           cmap=cmap,
                           norm=norm,
                           linewidth=np.fliplr(linewidth.T),
                           # linewidth=1,

                           arrowsize=arrowsize,
                           arrowstyle=arrowstyle, # -|> ->

                           minlength=minlength, # in axes units
                           maxlength=maxlength, # in axes units

                           integration_direction=integration_direction,
                           broken_streamlines=broken_streamlines,
                           start_points=stream_seed_points,
                           zorder=zorder,
                           )
        if alpha != 1:
            im_lines = im.lines
            im_arrows = im.arrows
            im_lines.set_alpha(alpha)
            im_arrows.set_alpha(alpha)
            # fix matlotlib bug with im.arrows not returning a list of artists
            for art in ax.get_children():
                if not isinstance(art, mpl.patches.FancyArrowPatch):
                    continue
                art.set_alpha(alpha)

        if patheffects_color is not None:
            from matplotlib import patheffects
            im_lines = im.lines
            im_arrows = im.arrows
            im_lines.set_path_effects([patheffects.withStroke(linewidth=lw+patheffects_lw, alpha=patheffects_alpha, foreground=patheffects_color)])
            # im_lines.set_path_effects([patheffects.SimpleLineShadow(offset=(.2, -.2), shadow_color=patheffects_color, alpha=0.5, rho=10.)])
            # fix matlotlib bug with im.arrows not returning a list of artists
            # Removes the arrow heads
            # for art in ax.get_children():
            #     if not isinstance(art, mpl.patches.FancyArrowPatch):
            #         continue
            #     art.set_path_effects([patheffects.SimpleLineShadow(offset=(.2, -.2),
            #                                                        shadow_color=patheffects_color,
            #                                                        alpha=0.6, rho=10.)])

        im_lines = im.lines
        # im_arrows = im.arrows
        im_lines.set_zorder(0)
        if cbar and cmap is not None and not isinstance(color, str) and not isinstance(color, list):
            cbar = colorbar(ax, norm=norm, cmap=cmap, label=label,
                            location=cbar_loc, dark_mode=dark_mode,
                            alpha=cbar_alpha
                            )

        if cbar and (isinstance(color, str) or isinstance(color, list)):
            from matplotlib.lines import Line2D
            legend_color = color
            # if legend_color is string, convert to RGBA list
            if isinstance(legend_color, str):
                legend_color = list(mpl.colors.to_rgba(legend_color))
                legend_color[-1] = alpha
            legend_color[-1] += 0.3
            legend_color[-1] = 1 if legend_color[-1] > 1 else legend_color[-1]
            arrow = Line2D([0, 0.1], [0, 0], color=legend_color, lw=1.5)
            # arrow.set_path_effects([patheffects.withStroke(linewidth=1, foreground='white', alpha=0.95)])
            # arrow = mpatches.FancyArrow(0, 3, 25, 0, color=color, width=2.5, length_includes_head=True)
            legend = ax.legend([arrow], [label],
            # legend = ax.legend(custom_lines, ['label'],
                      loc="lower left", 
                      # labelcolor='white' if dark_mode else 'black',
                      labelcolor=legend_color,
                      # fontsize=14, 
                      framealpha=cbar_alpha,
                      facecolor='black' if dark_mode else 'white',
                      edgecolor='black' if dark_mode else 'white',
                      borderpad=0.2,
                      # color='white' if dark_mode else 'black',
                      borderaxespad=0.7, # Legend closer to the border
                      # handletextpad=0.3, # Distance between circle and label is smaller
                      # labelspacing=1.5,  # Vertical space between labels
                      # markerscale=1,     # The size of the dots is twice as large.
                      )
            cbar_patheffects = False
            if cbar_patheffects:
                import matplotlib.patheffects as patheffects
                for text in legend.get_texts():
                    text.set_path_effects([patheffects.withStroke(linewidth=1.5, foreground='white', alpha=0.6)])
                for handle in legend.legendHandles:
                    handle.set_path_effects([patheffects.withStroke(linewidth=1.5, foreground='white', alpha=0.6)])

            legend.get_frame().set_boxstyle('Round', pad=0.4,
                                            # rounding_size=2,
                                            )

        # boxstyle = mpatches.BoxStyle("Round", pad=0.01)
        # rec = mpatches.FancyBboxPatch((xpos-0.1-extra_width/2, ypos-0.01),
        #                               width, height+extra_width/2,
        #                               fc=bg_color,
        #                               ec="none",
        #                               alpha=alpha,
        #                               boxstyle=boxstyle,
        #                               transform=ax.transAxes,
        #                               zorder=0,
        #                               )
            # add_quiver_key(ax,
            #                im,
            #                scale,
            #                units=None,
            #                pretty_name=label,
            #                bg_color='black' if dark_mode else 'white',
            #                text_color='white' if dark_mode else 'black',
            #                color=color)

        return im

    else:
        return None

def quivers(ax,
            field,
            selection,
            color_field=None,
            color='k',
            scale=1,
            max_length=None, 
            average_cells=2,
            alpha=1,
            cmap=None,
            norm=None,
            width=0.1, # width of arrow in plot units
            headwidth=3, # head width as multiple of width
            headlength=4,
            headaxislength=4,
            minshaft=1.5, # length below which arrow scales
            minlength=0.5, # min length as a mult of width
            pivot='tail',
            visible_ceiling=None, 
            alpha_min=0.5,
            zorder=0,
            legend_loc='upper right',
            cbar_loc='bottom right',
            cbar=True,
            legend=True,
            units=None,
            pretty_name=None,
            bg_color='black',
            text_color='white',
            ):

    transparent_cmap = False
    if transparent_cmap:
        from matplotlib.colors import LinearSegmentedColormap
        ncolors = 256
        mid_n = 200
        min_alpha = 0.2
        max_alpha = 1
        color_array = cmap(range(ncolors))
        alphas0 = np.linspace(max_alpha, min_alpha, ncolors-mid_n)
        alphas_mids = np.full(mid_n, max_alpha)
        alphas = np.concatenate((alphas_mids, alphas0))
        color_array[:,-1] = alphas
        cmap = LinearSegmentedColormap.from_list(name='turbo_alpha',colors=color_array)

    """ Plot a quiver plot of a vector field """
    # coords = selection.coords
    # scale = 2
    qx, qy = field
    qx = reduce_field(qx, average_cells)
    qy = reduce_field(qy, average_cells)
    qt = np.sqrt(qx**2 + qy**2)


    # Scale down qx and qy if necessary
    # qx = qx * scaling_factor
    # qy = qy * scaling_factor
    qt = np.sqrt(qx**2 + qy**2)

    visible_ceiling = 2*scale
    if visible_ceiling is not None:
        alpha = np.full_like(qt, 1.)
        alpha[qt>visible_ceiling] = np.full_like(qt, 1.)[qt>visible_ceiling] * (1-(qt[qt>visible_ceiling]-visible_ceiling)/scale)
        alpha[alpha<alpha_min] = alpha_min
        visible_floor = 0.1*scale
        alpha[qt<visible_floor] = np.full_like(qt, 0.01)[qt<visible_floor]
        alpha=alpha.T
    # set a max values of qx and qy if qt > 2*scale
    # if np.max(qt) > 2*scale:
    #     qx = qx*2*scale/np.max(qt)
    #     qy = qy*2*scale/np.max(qt)
    max_length = 2.5*scale
    if max_length is not None:
        scaling_factor = np.full_like(qt, 1)
        scaling_factor[qt > max_length] = max_length / qt[qt > max_length]
        qx = qx * scaling_factor
        qy = qy * scaling_factor
    
    # linewidths = qt.T/scale*width
    # linewidths[linewidths < 0.1] = 0.1
    # linewidths[linewidths > 4] = 4 
    # linewidths = np.full_like(qt, 0.5)

    x, y = selection.xy_bins(qx)
    options = dict(pivot=pivot,
                   cmap=cmap,
                   norm=norm,
                   angles='xy',
                   units='xy',
                   scale_units='xy',
                   scale=scale,
                   width=width,
                   headwidth=headwidth,
                   headlength=headlength,
                   headaxislength=headaxislength,
                   minshaft=minshaft,
                   minlength=minlength,
                   # edgecolor='k',
                   # linewidth=0.5,
                   alpha=alpha,
                   zorder=zorder,
                  )
    if color_field is not None:
        # print('color_field')
        color_field = reduce_field(color_field, average_cells)
        im = ax.quiver(x, y, qx.T, qy.T, color_field.T, **options)
    else:
        im = ax.quiver(x, y, qx.T, qy.T, color=color, **options)

    from matplotlib import patheffects
    patheffects_color = 'k'
    # im.set(path_effects=[patheffects.withStroke(linewidth=1+0.5, alpha=0.4, foreground=path_effects_color)])
    im.set_path_effects([patheffects.withSimplePatchShadow(offset=(0.5, -0.5), shadow_rgbFace=patheffects_color, alpha=0.1, rho=10.)])


    if legend:
        add_quiver_key(ax,
                       im,
                       scale,
                       units=units,
                       pretty_name=pretty_name,
                       bg_color=bg_color,
                       text_color=text_color,
                       color=color)
    if cbar:
        cbar = colorbar(ax, im=im, label=pretty_name, location=cbar_loc)
    # plot.add_quiver_key(axes['b'], im_quiver, vector_scale,
    #                quiver_units=vector_units,
    #                quiver_fancy_name=vector_pretty_name,
    #                xpos=0.89, ypos=0.9, width=0.19, height=0.05,
    #                quiver_key_bg='black', quiver_key_color='white')
    return im

def add_quiver_key(ax,
                   im,
                   scale,
                   units=None,
                   pretty_name=None,
                   alpha=0.6,
                   color='w',
                   text_color='w',
                   bg_color='k',
                   fontsize=None,
                   angle=0,
                   zorder=100,
                   ):
    xpos=0.87; ypos=0.88; width=0.19; height=0.05
    text_start = (f'{pretty_name}, ' if pretty_name not in ('', None) else '')
    text_end = (f' {units}' if units not in ('', None) else '')
    if scale < 0.01:
        text = text_start + f'{scale:.3f}' + text_end
    else:
        text = text_start + f'{scale:.1f}' + text_end

    # Adjust the width if the quiver text is long
    extra_width = (len(text)-30)*0.0055*3

    xpos = xpos - extra_width/2
    width = width + extra_width

    if bg_color is not None:
        boxstyle = mpatches.BoxStyle("Round", pad=0.01)
        rec = mpatches.FancyBboxPatch((xpos-0.1-extra_width/2, ypos-0.01),
                                      width, height+extra_width/2,
                                      fc=bg_color,
                                      ec="none",
                                      alpha=alpha,
                                      boxstyle=boxstyle,
                                      transform=ax.transAxes,
                                      zorder=0,
                                      )
        ax.add_artist(rec)

    qk = ax.quiverkey(im,
                      xpos,
                      ypos,
                      scale,
                      text,
                      labelpos='N',
                      coordinates='axes',
                      angle=angle,
                      color=color,
                      labelsep=0.1,
                      labelcolor=text_color,
                      fontproperties=dict(size=fontsize),
                      # zorder=zorder+1000,
                      zorder=1000,
                      )
    # get text artist
    # for text in qk.artist:
    #     text.set_color(color)
    #     text.set_zorder(1000)
        # text.set_alpha(alpha)
    return qk


def colored_line_legend(ax,
                        artists,
                        labels=None,
                        bg_alpha=0.3,
                        bg_color=None,
                        fontsize=None,
                        frameon=True,
                        dark_mode=True,
                        loc='lower center',
                        ncols=2,
                        zorder=200,
                        ):
    """ Add a colored line to the legend """
    # artists = np.atleast_1d(artists)
    if dark_mode:
        text_color = 'white'
        bg_color = 'black'
    else:
        text_color = 'black'
        bg_color = 'white'

    if labels is None:
        labels = ['']*len(artists)
    legend = ax.legend(artists,
                       labels,
                       loc=loc,
                       fontsize=fontsize,
                       frameon=frameon,
                       ncols=ncols,
                       fancybox=True,
                       handler_map={a: pypic.geometry.HandlerColorLineCollection(numpoints=4) for a in artists},
                       # handlelength=2.5,
                       handletextpad=0.5,
                       labelspacing=0.5,
                       facecolor=bg_color,
                       edgecolor='none',
                       labelcolor=text_color,
                       borderpad=0.1,
                       # borderaxespad=0.7,
                       # columnspacing=1.5,
                       framealpha=bg_alpha,
                      )
    legend.get_frame().set_boxstyle('Round', pad=0.3)
    legend.set_zorder(zorder)
    return legend


def field_slice(ax,
                field,
                xyz=None,
                selection=None,
                cmap=None,
                norm=None,
                alpha=1,
                cbar=True,
                cbar_loc='bottom right',
                label='',
                dark_mode=True,
                contour=False,
                contour_fill=True,
                contour_levels=20,
                lw=2,
                zorder=0,
                ):
    if selection is None:
        extent = None
    else:
        if selection.cut_axis_phys == 2:
            extent = selection.range_phys[:2, ::-1].flatten()
        elif selection.cut_axis_phys == 1:
            extent = selection.range_phys[[0,2], ::-1].flatten()
        else:
            extent = selection.range_phys[[1,2], ::-1].flatten()

    """ Plot a 2D field slice """
    if contour:
        x,y = selection.xy_bins(selection.b)
        contour_labels = np.linspace(norm.vmin, norm.vmax, contour_levels+1, endpoint=True)
                                   # color_levels+1, endpoint=True).format('.1f')
        opts = dict(origin='upper',
                    cmap=cmap,
                    norm=norm,
                    extend='both',
                    alpha=alpha,
                    zorder=zorder,
                    )
        if contour_fill:
            im = ax.contourf(x, y, field.T, contour_labels, **opts)
        else:
            opts.update(linewidths=lw)
            im = ax.contour(x, y, field.T, contour_labels, **opts)
    else:
        im = ax.imshow(field.T,
                       cmap=cmap,
                       norm=norm,
                       aspect='equal',
                       # interpolation='lanczos',
                       origin='upper',
                       extent=extent,
                       alpha=alpha,
                       zorder=0,
                       )
    if cbar:
        cbar = colorbar(ax, im=im, label=label, location=cbar_loc, dark_mode=dark_mode)
    return im

def draw_planet(ax=None, wireframe=True, res='low', center=None,
                radius=1, angle=-90, colors=('w','k'), dark_mode=True,
                zorder=-10, clip_on=True,
                **kwargs):
    if ax is None:
        ax = plt.gca()
    if center is None:
        center = np.array([0,0,0])
    if len(center) > 2:
        center = center[:2]

    if  ax.name == "3d":
        if wireframe:
            if dark_mode:
                color1 = 'w'; color2 = 'grey'
            else:
                color1 = 'k'; color2 = 'grey'
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi/2:10j]
            x = np.cos(u)*np.sin(v)
            y = np.sin(u)*np.sin(v)
            z = np.cos(v)
            ax.plot_wireframe(x, y, z, color=color1, alpha=0.5, zorder=zorder, clip_on=clip_on)
            u, v = np.mgrid[0:2*np.pi:20j, np.pi/2:np.pi:10j]
            x = np.cos(u)*np.sin(v)
            y = np.sin(u)*np.sin(v)
            z = np.cos(v)
            artists = ax.plot_wireframe(x, y, z, color=color2, alpha=0.35, zorder=zorder, clip_on=clip_on)
        else:
            from matplotlib.colors import LightSource
            import PIL
            resizes = {'low': 100, 'medium': 200, 'high': 400}
            dres = resizes[res]
            bm = PIL.Image.open('src/bluemarble.jpg')
            bm = np.array(bm.resize([int(d/dres) for d in bm.size]))/256.

            lons = np.linspace(-180, 180, bm.shape[1]) * np.pi/180 
            lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi/180 
            x = np.outer(np.cos(lons), np.cos(lats)).T
            y = np.outer(np.sin(lons), np.cos(lats)).T
            z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T

            artists = ax.plot_surface(x, y, z,
                            rstride=1, cstride=1,
                            linewidth=0,
                            antialiased=True,
                            facecolors=bm,
                            shade=True,
                            lightsource=LightSource(azdeg=90, altdeg=0),
                            zorder=zorder,
                            )
    else:
        from matplotlib.patches import Wedge, Circle
        theta1, theta2 = angle, angle + 180
        w1 = Wedge(center, radius, theta1, theta2, fc=colors[0], **kwargs)
        w2 = Wedge(center, radius, theta2, theta1, fc=colors[1], **kwargs)
        circle = Circle(center, radius, fc='none', ec='black', **kwargs)
        artists = [w1, w2, circle]
        for wedge in artists:
            ax.add_artist(wedge)
    return artists

def plot_surface_3d(ax,
                    X, Y, Z, # meshgrid of coordinates
                    field, # 2D field to plot
                    flat=True, # flat or 3D surface
                    contour=False,
                    contour_fill=True,
                    stride=4, # bin stride for pixel surface plot
                    height_scale=1, # scale of field for 3D surface
                    color='k',
                    alpha=0.5,
                    cmap=None,
                    norm=None,
                    contour_levels=40,
                    vmin=None,
                    vmax=None,
                    zorder=0,
                    lw=2,
                    clip_on=False,
                    path_effects=False,
                    **kwargs):
    """ Plot a 2D surface in pixels or countours in 3D """
    if vmin is None:
        if norm is None:
            # vmin = np.min(field); vmax = np.max(field)
            vmin = 0; vmax = 1
        else:
            vmin = norm.vmin; vmax = norm.vmax
    cmap = plt.cm.get_cmap() if cmap is None else cmap
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax) if norm is None else norm
    if contour:
        color_labels = np.linspace(vmin, vmax, contour_levels+1, endpoint=True)
        if np.all(X == X[0,0]):
            zdir = 'x'
            offset = X[0,0]
            A, B, C = field.T, Y, Z
        elif np.all(Y == Y[0,0]): 
            zdir = 'y'
            offset = Y[0,0]
            A, B, C = X, field.T, Z
        elif np.all(Z == Z[0,0]):
            zdir = 'z'
            offset = Z[0,0]
            A, B, C = X, Y, field.T
        else:
            raise ValueError('X, Y, or Z must have all the same values for a contour plot.')
        options = dict(zdir=zdir,
                       offset=offset,
                       origin='upper',
                       extend='both',
                       levels=color_labels,
                       # nchunk=10,
                       alpha=alpha,
                       cmap=cmap,
                       norm=norm,
                       zorder=zorder,
                       )
        if contour_fill:
            im = ax.contourf(A, B, C, contour_levels, **options)
            im.set_clip_on(clip_on)
        else:
            im = ax.contour(A, B, C, contour_levels, linewidths=lw, **options)
        if path_effects:
            im.set(path_effects=[patheffects.withStroke(linewidth=0.3, alpha=0.3, foreground='black')])
    else:
        if not flat:
            if np.all(X == X[0,0]):
                X = X + field.T*height_scale
            elif np.all(Y == Y[0,0]):
                Y = Y + field.T*height_scale
            elif np.all(Z == Z[0,0]):
                Z = Z + field.T*height_scale
        im = ax.plot_surface(X, Y, Z,
                             rstride=stride,
                             cstride=stride,
                             facecolors=cmap(norm(field.T)),
                             edgecolor='none',
                             # edgecolor='white',
                             # edgecolors='white',
                             # edge
                             linewidth=0,
                             # color='white',
                             antialiased=True,
                             shade=False,
                             cmap=cmap,
                             norm=norm,
                             zorder=zorder,
                             alpha=alpha,
                             )
    return im

def progress_bar(value,
                 range,
                 text=None,
                 ax=None,
                 relative=True,
                 location='top right',
                 dark_mode=True,
                 lw=1,
                 color='orange',
                 bg_color=None,
                 text_color=None,
                 edgecolor='white',
                 alpha=0.7,
                 fontsize=12,
                 zorder=200,
                 margin=None,
                 height_margin=0,
                 extra_height=4,
                 width=0.25,
                 height=0.04,
                 ):
    """ Add a progress bar to the plot """
    if bg_color is None:
        bg_color = 'black' if dark_mode else 'white'
    if text_color is None:
        text_color = 'white' if dark_mode else 'black'

    if range is None:
        min_value = 0
        max_value = 1
    else:
        min_value, max_value = range
    if ax is None:
        ax = plt.gca()

    pad = 1
    if relative:
        if location == 'bottom left':
            loc = [0.05-pad/50, 0.10, width+2*pad/50, height]
        elif location == 'bottom right':
            loc = [0.70-pad/50, 0.10, width+2*pad/50, height]
        elif location == 'top right':
            loc = [0.70-pad/50, 0.87, width+2*pad/50, height]
        elif location == 'top left':
            loc = [0.05-pad/50, 0.87, width+2*pad/50, height]
        if margin is not None:
            loc[0] += 3*margin
            loc[1] -= margin
        ax = ax.inset_axes(loc)
        # ybounds = (ax.get_position().bounds[-1]-0.004)/0.01 * 1.5
        ax.axis('off')

    # from matplotlib.transforms import Affine2D
    # import mpl_toolkits.mplot3d.art3d as art3d
    ax.set_aspect(1)
    width=25
    height=2.+height_margin
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    xpos=0
    ypos=0

    if bg_color is not None:
        bg = mpatches.FancyBboxPatch((xpos-0.2, ypos-0.2), width+0.2*2, height+extra_height, ec="none", fc=bg_color, alpha=alpha,
                                   boxstyle=mpatches.BoxStyle("Round", pad=pad), clip_on=False, zorder=zorder+10)
        ax.add_artist(bg)

    rec_bound = mpatches.FancyBboxPatch((xpos, ypos), width, height, ec=edgecolor, fc=[0,0,0,0], alpha=alpha/2,
                               boxstyle=mpatches.BoxStyle("Round", pad=0), clip_on=False, zorder=zorder+20)
    rec_fill = mpatches.FancyBboxPatch((xpos, ypos), width*value/(max_value-min_value), height, lw=0.2, ec=edgecolor, fc=color, alpha=0.9,
                               boxstyle=mpatches.BoxStyle("Round", pad=0), clip_on=False, zorder=zorder+30)
    ax.add_artist(rec_bound)
    ax.add_artist(rec_fill)
    ax.text(xpos+0.02, ypos+1.2, text,
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=ax.transAxes,
                color=text_color,
                fontsize=fontsize,
                clip_on=False,
                zorder=zorder+30,
                # bbox=dict(facecolor='black', edgecolor='none', alpha=0.3, boxstyle='round,pad=0.3')
            )
    # rec_fill.set_zorder(10000)
    # ax.add_patch(bg)
    # ax.add_patch(rec_bound)
    # ax.add_patch(rec_fill)
    # art3d.pathpatch_2d_to_3d(Rec0, z=1, zdir="z")
    # p = Circle((5, 5), 3)
    # ax.add_patch(p)
    # art3d.pathpatch_2d_to_3d(p, z=0, zdir="x")

def info_bar(text=None,
             ax=None,
             relative=True,
             location='top right',
             dark_mode=True,
             lw=1,
             bg_color='black',
             text_color='white',
             edgecolor='white',
             alpha=0.7,
             fontsize=12,
             zorder=200,
             margin=None,
             ):
    """ Add a progress bar to the plot """
    if not dark_mode:
        bg_color = 'w'
        text_color = 'k'

    if ax is None:
        ax = plt.gca()

    pad = 1
    if relative:
        width=.8
        height=.01
        if location == 'bottom left':
            loc = [0.05-pad/50, 0.10, width+2*pad/50, height]
        elif location == 'bottom right':
            loc = [0.70-pad/50, 0.10, width+2*pad/50, height]
        elif location == 'bottom center':
            loc = [0.50-pad/50, 0.10, width+2*pad/50, height]
        elif location == 'top right':
            loc = [0.70-pad/50, 0.88, width+2*pad/50, height]
        elif location == 'top center':
            loc = [0.50-pad/50, 0.88, width+2*pad/50, height]
        elif location == 'top left':
            loc = [0.05-pad/50, 0.88, width+2*pad/50, height]
        if margin is not None:
            if 'center' in location:
                loc[0] -= width*2/3
                loc[1] -= margin-0.01
            else:
                loc[0] += 3*margin
                loc[1] -= margin
        ax = ax.inset_axes(loc)
        # ybounds = (ax.get_position().bounds[-1]-0.004)/0.01 * 1.5
        ax.axis('off')

    ax.set_aspect(1)
    width=60
    height=0.2
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    xpos=0
    ypos=0

    if bg_color is not None:
        bg = mpatches.FancyBboxPatch((xpos-0.2, ypos-0.2), width+0.2*2, height+3, ec="none", fc=bg_color, alpha=alpha,
                                   boxstyle=mpatches.BoxStyle("Round", pad=pad), clip_on=False, zorder=zorder+10)
        ax.add_artist(bg)

    text = ax.text(xpos+0.02, ypos+1.2, text,
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=ax.transAxes,
                color=text_color,
                fontsize=fontsize,
                clip_on=False,
                zorder=zorder+30,
                # bbox=dict(facecolor='black', edgecolor='none', alpha=0.3, boxstyle='round,pad=0.3')
            )
    return text

def colorbar(ax,
             im=None,
             norm=None,
             cmap=None,
             label=None,
             labelpad=5,
             fontsize=12,
             orientation='horizontal',
             location='bottom left',
             dark_mode=True,
             bg_color='black',
             text_color='white',
             edge_color='k',
             edge_lw=0.2,
             alpha=0.8,
             pad=0.06,
             width=0.25,
             height=0.02,
             ticklabels=None,
             ticks=None,
             margin=None,
             clip_on=False,
             **kwargs):
    """ Add a colorbar to the plot """
    if not dark_mode:
        bg_color = 'w'
        text_color = 'k'
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # cbar = plt.colorbar(im, cax=cax, orientation=orientation, **kwargs)
    # if label is not None:
        # cbar.set_label(label, labelpad=labelpad, fontsize=fontsize)
    if im is None:
        im = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    if location == 'bottom left':
        loc = [0.05, 0.10, width, height]
    elif location == 'bottom right':
        loc = [0.70, 0.10, width, height]
    elif location == 'top right':
        loc = [0.70, 0.90, width, height]
    elif location == 'top left':
        loc = [0.05, 0.90, width, height]
    if margin is not None:
        # if margin is a list of two values, use the first for the x-axis and the second for the y-axis
        if isinstance(margin, (list, tuple)):
            loc[0] += margin[0]
            loc[1] += margin[1]
        else:
            loc[0] += 3*margin
            loc[1] += margin
    # Create an inset axis
    cbar_ax = ax.inset_axes(loc)

    fig = plt.gcf()
    cbar = fig.colorbar(im,
                        cax=cbar_ax,
                        extend='both',
                        pad=0,
                        extendfrac=0.04,
                        ticks=ticks,
                        # fontsize=fontsize,
                        # extendrect=True,
                        # ticks=[],
                        # drawedges=True,
                        # ax=ax,
                        orientation=orientation,
                        label=label,
                        )

    # set cbar fontsize
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.label.set_fontsize(fontsize)
    cbar.ax.xaxis.label.set_fontsize(fontsize)

    cbar.outline.set_color(edge_color)
    cbar.outline.set_linewidth(edge_lw)
    cbar.dividers.set_color(edge_color)
    cbar.dividers.set_linewidth(edge_lw)
    # cbar.solids.set_edgecolor("face")

    # Set colorbar label properties
    cbar.ax.yaxis.label.set_color(text_color)
    cbar.ax.xaxis.label.set_color(text_color)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.labelpad = labelpad
    cbar.ax.tick_params(axis='x', colors=text_color)
    cbar.ax.tick_params(axis='y', colors=text_color)

    # adjust size by looking at the axes bounds
    ybounds = (cbar_ax.get_position().bounds[-1]-0.004)/0.01 * 1.5
    xpos,ypos,width,height = -0.05-pad,-3.7+ybounds,1.1+2*pad,7.9-2*ybounds
    if bg_color is not None:
        bg = mpatches.FancyBboxPatch((xpos, ypos), width, height,
                                     ec="none", fc=bg_color, alpha=alpha,
                                     transform=cbar_ax.transAxes,
                                     boxstyle=mpatches.BoxStyle("Round", pad=pad),
                                     mutation_aspect=20,
                                     clip_on=False,
                                     zorder=0)
        cbar_ax.add_artist(bg)
    return cbar

def annotate_axes(ax,
                  text,
                  fontsize=20,
                  pad=0.2,
                  alpha=0.5,
                  dark_mode=True,
                  bg_color=None,
                  text_color='white',
                  edge_color='w',
                  offset=(0,0),
                  ):
    """ Annotate the axes with a label """
    if not dark_mode:
        text_color = 'k'
        edge_color = 'k'
    if bg_color is None:
        bg_color = 'k' if dark_mode else 'w'
        text_color = 'w' if dark_mode else 'k'

    # Place the label at the top right corner of the plot
    # ax.set_title(label, fontfamily='serif', loc='left', fontsize='medium')

    fig = plt.gcf()
    import matplotlib.transforms as mtransforms
    # trans = mtransforms.ScaledTranslation(5/72-150/72, -5/72, fig.dpi_scale_trans)
    trans = mtransforms.ScaledTranslation(5/72, -5/72, fig.dpi_scale_trans)
    # import matplotlib.patheffects as pe
    opts = dict(size=fontsize,
                rotation=0.,
                transform=ax.transAxes+trans,
                color=text_color,
                ha="left", va="top",
                bbox=dict(boxstyle="round", pad=pad,
                          lw=0.5,
                          ec=edge_color,
                          fc=bg_color,
                          alpha=alpha,
                          ),
            # path_effects=[pe.withStroke(linewidth=1, foreground='black', alpha=0.5)],
                )
    if ax.name == '3d':
        ax.text2D(0.0+offset[0], 1.0+offset[1], text, **opts)
    else:
        ax.text(0.0+offset[0], 1.+offset[1], text, **opts)

def plot_data_dashboard(ax,
                        selection,
                        df,
                        key=None,
                        key2=None,
                        x_key='cycle',
                        edgecolor='white',
                        lw=1,
                        lw2=1,
                        color=None,
                        text_color='white',
                        grid_color=None,
                        alpha=0.8,
                        alpha2=1,
                        fontsize=12,
                        ylim=None,
                        xlim=None,
                        ylim2=None,
                        ylabel=None,
                        ylabel2=None,
                        xlabel=None,
                        x_lines=None,
                        y_lines=None,
                        y_ticks=None,
                        labels=None,
                        legend=False,
                        value_label=False,
                        fill=None,
                        clip_on=True,
                        normalize=False,
                        highlight=None,
                        stream_data=None,
                        ):
    import pypic.colors as cl
    if color is None:
        # colors = [color, cl.elements3, cl.elements4, 'white']
        colors = cl.colors
    else:
        colors = np.atleast_1d(color)
        if len(color) == 1:
            colors = [color, cl.elements3, cl.elements4, 'white']

    if grid_color is None:
        grid_color = text_color
    # if physical_time is not None:
    #     mins = int(physical_time // 60)
    #     secs = int(physical_time - 60*mins)
    #     text += f' ({mins:01d}min:{secs:02d}s)'

    cycle = selection.cycle
    species = selection.species
    sim = selection.sim
    ax2 = None
    if key is None:
        raise ValueError('Dataframe key must be a string or list of strings')
    key = np.atleast_1d(key)
    if labels is not None:
        labels = np.atleast_1d(labels)
        if len(labels) != len(key):
            raise ValueError('Labels must have the same length as key')
    
    # check if df has column time
    if x_key is not None and x_key in df.columns:
        xa = x_key
    elif 'cycle' in df.columns:
        xa = 'cycle'
    elif 'time' in df.columns:
        xa = 'time'
    else:
        raise ValueError('DataFrame must have a column "cycle" or "time"')
    yranges = []
    for k in key:
        if k not in df.columns:
            raise ValueError(f'DataFrame must have a column "{k}"')
        yranges.append(df[k].max() - df[k].min())
    yrange = np.max(yranges)
    if ylim is not None:
        yrange = 1
        yrange = ylim[1] - ylim[0]

    # from scipy.ndimage import gaussian_filter1d
    # jdotE = gaussian_filter1d(jdotE, 1)
    if y_lines is not None:
        y = np.atleast_1d(y_lines)
        for y in y_lines:
            ax.axhline(y=y, color=grid_color, lw=2, ls='--', alpha=0.9, zorder=1)
    if x_lines is not None:
        x = np.atleast_1d(x_lines)
        for x in x_lines:
            ax.axvline(x=x, color=grid_color, lw=2, ls='--', alpha=0.9, zorder=1)

    if stream_data is not None:
        for q in stream_data['q'].unique():
            df_q = stream_data[stream_data['q'] == q]
            ax.plot(df_q[xa], df_q['energy'], color='black', lw=0.3, ls='-', alpha=0.02)
            # ax.plot(df_q[xa], df_q['energy'], color='black', ls='', marker='o', ms=0.5, alpha=0.03)

    if fill is not None:

        colors_dark = [adjust_lightness(cl.colors[0], 0.7),
                       adjust_lightness(cl.colors[1], 0.7)]
        ax.fill_between(df[xa], df['q10'], df['q90'],
                        color=colors_dark[0], alpha=0.5, clip_on=clip_on, label='10th and 90th percentile')
        ax.fill_between(df[xa], df['q1'], df['q3'],
                        color=colors_dark[1], alpha=0.5, clip_on=clip_on, label='first and third quartile')

    for i, k in enumerate(key):
        label = k if labels is None else labels[i]
        c = colors[i]
        if normalize:
            ya = df[k] / abs(df[k]).max()
        else:
            ya = df[k]
        # plot all data
        ax.plot(df[xa], ya, color=c, lw=lw/2, ls='--', alpha=alpha/2, clip_on=clip_on)
        
        # plot data up to cycle
        ax.plot(df[xa][df[xa]<=cycle] , ya[df[xa]<=cycle], color=c, lw=lw, ls='-', alpha=alpha, clip_on=clip_on, label=label)

        # plot current value
        ax.scatter(df[xa][df[xa]==cycle] , ya[df[xa]==cycle], color=c, s=30, alpha=alpha, clip_on=clip_on)

        # plot horizontal line at y=0
        ax.axhline(y=0, color=c, lw=1, ls='--', alpha=0.5)

        # highlight the values requested
        if highlight is not None:
            highlight = np.atleast_1d(highlight)
            for h in highlight:
                ax.plot(df[xa][df[xa]==h] , ya[df[xa]==h],
                        ls='', marker='o', color=c,
                        ms=5, alpha=alpha/2, clip_on=clip_on)
                ax.axvline(x=h, color=text_color, lw=2, ls='--', alpha=0.3)
        
        # plot current value label
        if value_label:
            last_cycle = df[xa][df[xa]==cycle].iloc[0]
            last_value = ya[df[xa]==cycle].iloc[0]
            ax.text(last_cycle, last_value+yrange*0.07, f'{last_value:.0f}',
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    transform=ax.transData,
                    # transform=ax.transAxes,
                    color=text_color,
                    fontsize=fontsize+2,
                    alpha=0.3,
                    clip_on=clip_on,
                    bbox=dict(facecolor='black', edgecolor='none', alpha=0.05, boxstyle='round,pad=0.2')
                )
    if key2 is not None:
        ax2 = ax.twinx()
        key2 = np.atleast_1d(key2)
        for i, k in enumerate(key2):
            label = k if labels is None else labels[i]
            c = colors[i]
            ya = df[k]
            ax2.plot(df[xa], ya, color=text_color, lw=lw2/2, ls='--', alpha=alpha2/2, clip_on=clip_on)

            # plot data up to cycle
            ax2.plot(df[xa][df[xa]<=cycle] , ya[df[xa]<=cycle], color=text_color, lw=lw2, ls='-', alpha=alpha2, clip_on=clip_on, label=label)
            # plot current value
            ax2.scatter(df[xa][df[xa]==cycle] , ya[df[xa]==cycle], color=text_color, s=30, alpha=alpha2, clip_on=clip_on)

            # highlight the values requestes
            if highlight is not None:
                highlight = np.atleast_1d(highlight)
                for h in highlight:
                    ax2.plot(df[xa][df[xa]==h] , ya[df[xa]==h],
                            ls='', marker='o', color=text_color,
                            ms=5, alpha=alpha2, clip_on=clip_on)

    
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=text_color, fontsize=fontsize, alpha=alpha)

    if ylabel is not None:
        ax.set_ylabel(ylabel, color=text_color, fontsize=fontsize, alpha=alpha)
                      # loc='right')
    
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        # ax.set_xlim(df[xa].min(), df[xa].max())
        if x_key == 'cycle':
            # ax.set_xlim(0, df[xa].max())
            ax.set_xlim(sim.cycle_limits[0], sim.cycle_limits[1])
        else:
            ax.set_xlim(df[xa].min(), df[xa].max())

    if ylim is not None:
        ax.set_ylim(ylim)

    if ylim2 is not None:
        ax2.set_ylim(ylim2)

    if y_ticks is not None:
        ax.set_yticks(y_ticks)

    if legend:
        ax.legend(
                  loc='lower right',
                  fontsize=fontsize, frameon=False,
                  facecolor='black',
                  edgecolor='none',
                  labelcolor=text_color,
                  # labelcolor='white',
                  # alpha=0.6,
                  ncol=4,
                  )
        
    # create a legend with labels in one row
    if ylabel2 is not None:
        ax2.set_ylabel(ylabel2, color=text_color, fontsize=fontsize, alpha=alpha)
    elif key2 is not None:
        ax2.set_ylabel(key2[0], color=text_color, fontsize=fontsize, alpha=alpha)
        # ax2.legend(
        #           loc='lower right', fontsize=fontsize, frameon=False,
        #           facecolor='black', edgecolor='none',
        #           labelcolor=text_color,
        #           # alpha=0.6,
        #           ncol=4,
        #           )

    # ax.spines['bottom'].set_color(edgecolor)
    # ax.spines['top'].set_color(edgecolor)
    # ax.spines['right'].set_color(edgecolor)
    # ax.spines['left'].set_color(edgecolor)
    for spine in ['top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)
    if key2 is not None:
        for spine in ['top', 'left', 'bottom']:
            ax2.spines[spine].set_visible(False)
        ax2.tick_params(axis='x', colors=text_color, labelsize=fontsize)
        ax2.tick_params(axis='y', colors=text_color, labelsize=fontsize)
    ax.tick_params(axis='x', colors=text_color, labelsize=fontsize)
    ax.tick_params(axis='y', colors=text_color, labelsize=fontsize)
    ax.yaxis.label.set_color(text_color)
    ax.xaxis.label.set_color(text_color)
    # ax.set_frame_on(False)
    if x_key == 'cycle':
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator(500))
        from matplotlib import ticker
        def my_formatter_fun(x, p):
            return "%dk" % (x/1000)
        ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(my_formatter_fun))

    ax.tick_params(axis='both', which='both', length=3, width=2, direction='in',
                   pad=2, labelsize=fontsize, colors=text_color,
                   labelcolor=text_color)
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_alpha(alpha/2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_alpha(alpha)
    ax.grid(True, alpha=0.3, color=grid_color)
    right_ticks = True if key2 is None else False 
    ax.tick_params(axis='both', which='both', length=5, width=2, direction='in',
                   left=True, right=right_ticks, top=False, bottom=True,
                   labelleft=True, labeltop=False, labelright=right_ticks)
    return ax, ax2


def dataframe_lines(ax,
                    df,
                    x=None,
                    y=None,
                    color=None,
                    cmap=None,
                    norm=None,
                    lw=3,
                    ls='-',
                    alpha=None,
                    label=None,
                    selection=None,
                    legend_ncols=1,
                    legend_loc='upper center',
                    legend_alpha=0.8,
                    dark_mode=False,
                    colored_lines=True,
                    ):
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    if color is None:
        color = cl.colors
    color = np.atleast_1d(color)
    if len(y) > len(x):
        x = np.repeat(x, len(y))
    if len(y) > len(color):
        color = np.repeat(color, len(y))
    if label is None:
        label = y
    label = np.atleast_1d(label)
    cmap = np.atleast_1d(cmap)
    if len(y) > len(cmap):
        cmap = np.repeat(cmap, len(y))
    norm = np.atleast_1d(norm)
    if len(y) > len(norm):
        norm = np.repeat(norm, len(y))
    ls = np.atleast_1d(ls)
    if len(y) > len(ls):
        ls = np.repeat(ls, len(y))
    lcs = []

    for i, (x_, y_, color_, cmap_, norm_, ls_, label_) in enumerate(zip(x, y, color, cmap, norm, ls, label)):
        if selection is not None:
            y_units, _, y_range = pypic.units.info(y_, selection)
        else:
            y_units = ''
            y_range = [df[y_].min(), df[y_].max()]
        if norm_ is None:
            norm_ = color_norm(y_range, log=False)
        if cmap_ is not None:
            color_ = None
        if colored_lines:
            lc = colored_line(ax,
                              df,
                              x=x_, y=y_, c=y_,
                              color=color_,
                              lw=lw,
                              clip_on=True,
                              cmap=cmap_, norm=norm_,
                              path_effects=True,
                              path_effects_color='k',
                              path_effects_alpha=0.7,
                              path_effects_lw=1.,
                              path_effects_type='new-line',
                              zorder=100,
                              alpha=None,
                              )
        else:
            lc = ax.plot(df[x_], df[y_], label=label_,
                         color=color_, lw=lw, ls=ls_, alpha=alpha, zorder=100)
        lcs.append(lc)
        ax.set_ylim(y_range)
        yrange = ax.get_ylim()
        ax.fill_betweenx(yrange, -4, 4, color='gray', alpha=0.05)
        ax.fill_between(df[x_], df[y_], 0, color='gray', alpha=0.2)
        if yrange[0] <= 0:
            ax.axhline(0, color='k', lw=1, ls='--', alpha=0.5)
            ax.fill_between(df[x_], yrange[0], 0, color='gray', alpha=0.1)
        # ax.fill_between(df[x_], df[y_], 0, color='gray', alpha=0.1)
    if colored_lines:
        colored_line_legend(ax, lcs,
                            label,
                            bg_alpha=legend_alpha,
                            bg_color=None,
                            fontsize=None,
                            dark_mode=dark_mode,
                            loc=legend_loc,
                            ncols=legend_ncols)
    else:
        ax.legend(loc=legend_loc,
                  # fontsize=12,
                  # frameon=False,
                  ncol=legend_ncols)

# def dataframe_lines(ax,
#                     df,
#                     x_key=None,
#                     y_key=None,
#                     lw=1,
#                     color=None,
#                     ylim=None,
#                     xlim=None,
#                     xlabel=None,
#                     ylabel=None,
#                     # x_lines=None,
#                     # y_lines=None,
#                     labels=None,
#                     legend=True,
#                     legend_loc='lower left',
#                     fill=None,
#                     clip_on=True,
#                     ):
#     lc0 = colored_line(ax, selection.df,
#                  x='x', y='pperp1', c='pperp1',
#                  color=None, lw=3,
#                  clip_on=True,
#                  cmap=selection.cmap, norm=selection.norm,
#                  path_effects=True,
#                  path_effects_color='k',
#                  path_effects_alpha=0.9,
#                  path_effects_lw=1.5,
#                  path_effects_type='new-line',
#                  zorder=100,
#                  )

def plot_fields_and_particles(ax, fig, data, selection, labels=('x','y'),):
    position_phys = np.asarray(data[0:3])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_xlim(selection.min_phys[0],selection.max_phys[0])
    ax.set_ylim(selection.min_phys[1],selection.max_phys[1])
    # ax.scatter(position_phys[0,:], position_phys[1,:], s=0.1, c=weight, cmap='turbo', norm=norm)
    ax.scatter(position_phys[0,:], position_phys[1,:], s=1, c='red')
    ax.set_aspect('equal', adjustable='box')
    # ax.axhline(y=0.0, ls='--')
    # ax.axvline(x=0.0, ls='--')
    ax.invert_xaxis()
    ax.invert_yaxis()

def stylize_axes(ax):
    ax.tick_params(direction="in")
    ax.tick_params(which='major', width=2)
    ax.tick_params(which='major', length=8)
    ax.set_aspect('equal', 'box')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_edgecolor('white')
    ax.spines['bottom'].set_linestyle('-')
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_edgecolor('white')
    ax.spines['left'].set_linestyle('-')
    ax.set_facecolor('#171717')
    ax.grid(color='white', linestyle='--', linewidth=0.8, alpha=0.25)

def test_axes(fig):
    """ Testing axes """
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)

def dash_text(selection, data, display):
    """ Model and selection info for display """
    return (f'MHD UCLA + iPIC'+"\n\n"
            f'Cycle: {selection.cycle:,}'+"\n\n"
            r'$X_{phys}$'+ f' = {selection.center_phys[0]:.1f} '+r' $R_E$'+"\n"
            r'$Y_{phys}$'+ f' = {selection.center_phys[1]:.1f} '+r' $R_E$'+"\n"
            r'$Z_{phys}$'+ f' = {selection.center_phys[2]:.1f} '+r' $R_E$'+"\n\n"
            r'box'+ f' = {selection.delta_phys[0]:.1f} '+r' $R_E$'+"\n\n"
            r'$N_P = $' + f'{len(data[0]):,}'+"\n"
            )

def show_info(val, ui):
    """ Show details about the model in a new figure """
    fig_info = plt.figure(figsize=(8, 8))
    ax_info = fig_info.add_subplot(111)
    ax_info.axis('off')
    info_text = (f'model size in cells = {ui.selection.sim.size_cell} \n'+
                 f'box lower bound in phys = {ui.selection.sim.min_phys} Re\n'+
                 f'box upper bound in phys = {ui.selection.sim.max_phys} Re\n'+
                 f'box size in phys = {ui.selection.sim.size_phys} Re\n'+
                 f'box size in code units = {ui.selection.sim.size_code} di\n')

    ax_info.text(0.05, 0.95, info_text, fontsize=16, verticalalignment='top')
    plt.show(block=False)

def draw_path_text(ax, text='test'):
    """ Pretty print the selection """
    from matplotlib import patheffects
    text = ax.text2D(0.5, 0.8, text,
                     transform=ax.transAxes,
                     color=cl.bg,
                     ha='center', va='center', size=20)
    text.set_path_effects([patheffects.Stroke(linewidth=3, foreground='white', alpha=0.5),
                           patheffects.Normal()])
    print(ui.selection)

def shape(ax,
          selection,
          shape,
          dark_mode=True,
          cmap = 'gray',
          alpha = 0.6,
          smooth_std=None,
          zorder=0,
          ):
    """ Plot the dipolarizations shapes """
    if smooth_std is not None:
        from scipy.ndimage import gaussian_filter
        shape = gaussian_filter(shape, sigma=smooth_std)
    shape[shape < 0.01] = np.nan
    field_plot_opts = dict(
                           contour=False,
                           contour_fill=True,
                           contour_levels=10,
                           cmap=cmap,
                           norm=mpl.colors.Normalize(vmin=0, vmax=1),
                           alpha=alpha,
                           cbar=False,
                           label=None,
                           dark_mode=dark_mode,
                           )
    im_bg = field_slice(ax,
                        shape,
                        selection=selection,
                        zorder=zorder,
                        **field_plot_opts,
                        )

def plot_intersections(ax,
                       data=None,
                       x=None,
                       y=None,
                       z=None,
                       selection=None,
                       clip_on=True,
                       radius=0.2,
                       alpha=None,
                       lw=1.5,
                       color='white',
                       z_intersect=None,
                       ):
    """ Plot path intersections with the selection planes """
    if alpha is None:
        alpha = 0.4

    positions, zdirs = find_intersections(data=data,
                           x=x,
                           y=y,
                           z=z,
                           selection=selection,
                           z_intersect=z_intersect,
                           )
    for position, zdir in zip(positions, zdirs):
        draw_3d_circle(ax, zdir, radius=radius, center=position, color=color,
                       alpha=alpha, lw=lw, clip_on=clip_on)

def save_figs(val, ui):
    """ Save the figures """
    refine_vdf(0, ui, save=True)
    save_vdf(0, ui, save=True)
    save_vdf(0, ui, bg='white', save=True)
    # print('save figs')

if __name__ == '__main__':
    # new figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    circle = draw_circle(ax=ax, origin=[0,0], radius=5, normal='z')
    # test_imshow()
    # test_axes()
    plt.show()
    # plt.savefig('test.png')
    # pass
