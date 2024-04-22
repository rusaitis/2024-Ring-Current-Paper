import sys
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pypic
import pypic.units as units
import pypic.plot as plot
import pypic.colors as cl
from pypic.fieldline import fieldline

# --- PLOT SETTINGS
show_only = False
publication_quality = True
dark_mode = False
transparent = False
plot_progress_bar = False
LaTeX = True
clip_on = False

sim = pypic.ipic3D()
cycle = 202500
selection = pypic.Selection(sim,
                            species = 1,
                            cycle = cycle,
                            min_phys = [ -30, -10, 0],
                            max_phys = [   0,  10, 0],
                            )

# --- LOAD AND CALCULATE DATA
selection.data_dir = os.path.join(os.path.expanduser('~'), 'DATA')
selection.figures_dir = 'figures'

selection_view = selection.duplicate()
selection_view.min_phys = [-31, -2, -3.5] 
selection_view.max_phys = [  0,  10,  3.5]

# Figure export
selection.figures_dir = 'figures'
figure_name = 'figure10'

import colorcet as cc
cmap = mpl.colormaps['rainbow']
cmap_turbo = mpl.colormaps['turbo']
cmap = cl.bkr_extra_cmap

layout = [['a']
          ]
plots = [
         dict(label='a', cycle=202500, species=1, field='bz', vectors=None, vector_color=None, cbar=True, legend=True,
              scalar_cmap=cmap, vector_cmap=None, vector_label=None),
         ]

trace_centers = [
                 [-25.4, .5, 0],
                 [-23.1, 0., 0],
                 [-20, 2, 0],
                 [-21, 1.6, 0],
                 [-19, 1, 0],
                 [-16, 1, 0],
                ]

def plot_fieldline(ax, selection, center):
    boundary_conditions = {'r_min': 3.,
                           'r_max': 32.,
                           'lower_bound': selection.min_phys,
                           'upper_bound': selection.max_phys,
                           'max_iter': 1e4,
                           'verbose': False,
                           }
    fl = fieldline(r=center,
                   # step=0.1, # 0.05-0.2 for RK4, 0.01-0.05 for Euler
                   step=0.1,
                   method='RK4',
                   backtrace=False,
                   B_func=selection.f_B,
                   boundary_conditions=boundary_conditions)
    fl.trace()
    lc = fl.plot_fieldline(ax=ax,
                      color_key='bz', # 'b', 'bz', 'n', 'errors', 'markers'
                      # color='white', # single color
                      cmap=cmap, # overrides default colors
                      lw=3.2, # line width
                      alpha=0.99, # transparency
                      arrows=True, # add arrows to show field direction
                      end_points=True, # add circles to end points
                      colorbar=False,
                      clip_on=clip_on,
                      selection=selection,
                      intersections=True,
                      intersection_alpha=0.6,
                      zorder=100,
                      draw=True,
                      # arrow_scale=25,
                      # arrow_color='white',
                      arrow_color=None,
                      # arrow_alpha=1,
                      )
    points = fl.r

    return lc, points[:,-1]

field_plot_opts = dict(
                       contour=True,
                       contour_fill=True,
                       contour_levels=10,
                       cmap=None,
                       norm=None,
                       alpha=0.6,
                       cbar=True,
                       cbar_loc='bottom right',
                       label=None,
                       dark_mode=dark_mode,
                       clip_on=clip_on,
                       )

vector_plot_opts = dict(
                        density=1., # float or (float, float)
                        lw=1.5, # linewidth of streamlines
                        lw_min=0.5,
                        lw_max=2.5,
                        # stream_seed_points=stream_seed_points , # array of points to seed streamlines
                        broken_streamlines=True,
                        smooth_field=True,
                        # seed_point_size=25,
                        # seed_point_color='blue',
                        arrowstyle='-|>',
                        # arrowstyle='->',
                        # arrowstyle='-',
                        arrowsize=1.4,
                        average_cells=4,
                        zorder=0,
                        # patheffects_color='w',
                        patheffects_alpha=0.4,
                        patheffects_lw=0.2,
                        color='k',
                        cmap=None,
                        norm=None,
                        alpha=1,
                        cbar=True,
                        cbar_loc='bottom left',
                        cbar_alpha=0.8,
                        label=None,
                        dark_mode=dark_mode,
                        minlength=0.09,
                        maxlength=1.3,
                        )

plot.configure_matplotlib(dark_mode, transparent)

# Size of the figure
fig, axes = plt.subplot_mosaic(layout,
                               figsize=(10, 5),
                               subplot_kw=dict(projection='3d', computed_zorder= False),
                               )

# Annotate axes with axes labels
if len(axes) > 1:
    for label, ax in axes.items():
        plot.annotate_axes(ax, label, fontsize=20, dark_mode=dark_mode)

# Plot each panel
for p in plots:
    print(f'Plotting panel {p["label"]}...')
    ax = axes[p['label']]
    selection = p.get('selection', selection)
    coord = p.get('coord', 'phys')

    # --- LINE PLOTS
    if coord != 'phys' and 'x' in p and 'y' in p:
        x_keys = np.atleast_1d(p.get('x'))
        y_keys = np.atleast_1d(p.get('y'))
        keys = np.unique(np.concatenate([x_keys, y_keys]))
        df = selection.to_dataframe(keys, y_cut=p.get('y_cut', 0))
        plot.dataframe_lines(ax, df, x=p['x'], y=p['y'],
                             color=p.get('color', None),
                             cmap=p.get('cmap', None),
                             norm=p.get('norm', None),
                             label=p.get('labels', None),
                             selection=selection,
                             )
        continue

    # --- FIELD PLOTS
    if selection.get_field(p['field']) is not None:
        skip_calculation = True
    else:
        skip_calculation = False
    if p.get('cycle', None) is not None:
        if p['cycle'] != selection.cycle:
            skip_calculation = False
            selection.cycle = p['cycle']
    if not skip_calculation:
        selection.calculate(quick=True)

    # --- SELECT FIELDS
    scalar_field = selection.get_field(p['field'])

    vector_field = selection.get_field(p['vectors'])
    vector_color = selection.get_field(p['vector_color'])
    if vector_color is None:
        vector_color = p.get('vector_color', None)

    # --- LABELS 
    scalar_label = units.pretty_name(p['field'], LaTeX=LaTeX)
    vector_label = units.pretty_name(p['vectors'], LaTeX=LaTeX)
    vector_color_label = units.pretty_name(p['vector_color'], LaTeX=LaTeX)

    # --- GET UNITS AND LIMITS
    opt = dict(species=selection.species, coord='phys')
    scalar_units, _, scalar_range = units.info(p['field'], selection, **opt)
    vector_units, _, vector_range = units.info(p['vectors'], selection, **opt)

    scalar_label = p.get('scalar_label', scalar_label)
    if scalar_units is not None and scalar_label is not None:
        scalar_label += f' [{scalar_units}]'
    vector_pretty_name = units.pretty_name(p['vectors'], LaTeX=LaTeX)
    vector_range[0] = 0
    if p['label'] != 'b':
        vector_range = [0, 1.5]
    vector_scale = vector_range[1]/2
    vector_label = p['vector_label']

    # --- COLORMAPS
    cmap_scalar = p['scalar_cmap']
    cmap_vector = p['vector_cmap']
    norm_scalar = plot.color_norm(scalar_range, log=False)
    norm_vector = plot.color_norm(vector_range, log=False)
    selection.cmap = cmap_scalar
    selection.norm = norm_scalar

    # --- ADJUST COLORMAP SATURATION
    cmap_scalar.set_over(cl.adjust_lightness(cmap_scalar(1.0), 1.2))
    cmap_scalar.set_under(cl.adjust_lightness(cmap_scalar(0.0), 1.2))

    if scalar_field is not None:
        field_plot_opts['cmap'] = cmap_scalar
        field_plot_opts['norm'] = norm_scalar
        field_plot_opts['label'] = scalar_label

        selection_view.f_B = selection.f_B
        for center in trace_centers:
            lc, _ = plot_fieldline(ax, selection_view, center)

        im_bg = plot.plot_surface_3d(ax, selection.x, selection.y, selection.z, scalar_field, **field_plot_opts)

    if vector_field is not None:
        color = (vector_color if vector_color is not None else vector_plot_opts.get('color', 'black'))
        vector_plot_opts['color'] = color
        vector_plot_opts['cmap'] = cmap_vector
        vector_plot_opts['norm'] = norm_vector
        vector_plot_opts['label'] = vector_label
        vector_plot_opts['broken_streamlines'] = False if p['label'] in ['b'] else True
        vector_plot_opts['density'] = 0.8 if p['label'] in ['b'] else 1
        vector_plot_opts['color'] = 'k' if p['label'] in ['b'] else 'w'
        vector_plot_opts['lw'] = 1.5 if p['label'] in ['b'] else 1
        vector_plot_opts['alpha'] = 0.7 if p['label'] in ['b'] else 0.7
        vector_plot_opts['dark_mode'] = True if p['label'] in ['a', 'c'] else False
        vector_plot_opts['cbar_alpha'] = 0.6 if p['label'] in ['a', 'c'] else 0.85
        im_stream = plot.streamlines(ax,
                                     vector_field,
                                     selection, # selection object for bins and limits
                                     scale=vector_scale*1, # reference size
                                     **vector_plot_opts,
                                     )

        if p['label'] in ['b']:
            from pypic.fields import find_dipolarizations
            shapes = find_dipolarizations(ax, selection,
                                          'bz',
                                          min_length=14,
                                          dbz=2,
                                          x_lims=[-20, -10],
                                          )
            for shape in shapes:
                plot.shape(ax, selection, shape, dark_mode=True,
                           cmap = 'binary', alpha = 0.8, smooth_std=1, zorder=0)
        if p['label'] in ['b']:
            vx = vector_field[0]
            from pypic.fields import find_reversals
            find_reversals(ax, selection, vx, color='w', first_only=False)

axes_opts_phys = dict(
                      draw_radii=[5,8],
                      planet=True,
                      axis_lines=True,
                      grid_lines=True,
                      x_lines=[-30, -25, -20, -15, -10],
                      y_lines=[-10, -5, 0, 5, 10],
                      labels=True,
                      minor_labels=True,
                      dark_mode=dark_mode,
                      transparent=transparent,
                      alpha=0.9,
                      title=None,
                      x_label=r'$\text{X}$',
                      y_label=r'$\text{Y}$',
                      z_label=r'$\text{Z}$',
                      tick_loc = [],
                      label_loc = [],
                      zoom=0.35,
                      zorder=100,
                     )

axes_opts_other = dict(
                   coord='other',
                   dark_mode=False,
                   transparent=True,
                   x_label=r'$\text{x}_{\text{GSM}}$ [R$_E$]',
                   y_label='',
                   tick_loc = ['left'],
                   label_loc = ['left'],
                   tick_marks = ['bottom'],
                   )

# Otherwise configure manually
# axes_opts_phys['tick_loc'] = ['left', 'bottom']
# axes_opts_phys['label_loc'] = ['left', 'bottom']
plot.configure_axes(axes['a'], selection_view, **axes_opts_phys)
axes['a'].view_init(elev=22, azim=40)


def update_cbar(ax, opts):
    cbar = plot.colorbar(ax,
                         norm=opts['norm'],
                         cmap=opts['cmap'],
                         label=opts['label'],
                         labelpad=5,
                         fontsize=15,
                         orientation='horizontal',
                         location='bottom right',
                         dark_mode=True,
                         bg_color='black',
                         text_color='white',
                         edge_color='k',
                         edge_lw=0.2,
                         alpha=0.5,
                         pad=0.1,
                         # margin=-0.15,
                         margin=0.,
                         width=0.32,
                         height=0.032,
                        )
    return cbar

# ui.field_opts = get_field_info(ui.selection, 'bz', ui.field_opts)
cbar_field = update_cbar(ax, field_plot_opts)

if plot_progress_bar:
    labels = np.array(layout).flatten()
    progress_axes = [l for l in labels if np.where(np.array(layout) == l)[0] == 0]
    for p in plots:
        selection = p.get('selection', selection)
        cycle = selection.cycle
        if p.get('cycle', None) is not None:
            cycle = p['cycle']
        if p['label'] in progress_axes:
            plot.progress_bar(cycle, sim.cycle_limits,
                             units.pretty_time(sim.dt_phys*cycle, LaTeX=LaTeX),
                             ax=axes[p['label']])
if show_only:
    # fig.tight_layout()
    plt.show()
    plt.close()
else:
    # fig.tight_layout()
    dpi = 300 if publication_quality else 200
    if not os.path.exists(selection.figures_dir):
        os.makedirs(selection.figures_dir)
    # savepath = os.path.join(selection.figures_dir, f'{figure_name}.png')
    savepath = os.path.join(selection.figures_dir, f'{figure_name}.pdf')
    print(f'Saving figure to {savepath}...')
    plt.savefig(savepath,
                dpi=dpi,
                # bbox_inches='tight',
                pad_inches=0.01, #default 0.1
                # facecolor=ax.get_facecolor(),
                transparent=transparent,
                )
