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

# --- PLOT SETTINGS
show_only = False
publication_quality = True
dark_mode = False
transparent = False
plot_progress_bar = True
LaTeX = True

sim = pypic.ipic3D()
cycle = 1
selection = pypic.Selection(sim,
                            species = 1,
                            cycle = 202500,
                            min_phys = [ -30, -14, 0],
                            max_phys = [  13,  14, 0],
                            )

# --- LOAD AND CALCULATE DATA
selection.data_dir = os.path.join(os.path.expanduser('~'), 'DATA')
selection.figures_dir = 'figures'

# Figure export
selection.figures_dir = 'figures'
figure_name = 'figure5'

import colorcet as cc
cmap = mpl.colormaps['rainbow']
cmap_turbo = mpl.colormaps['turbo']

layout = [['a', 'b'],
          ['c', 'd'],
          ['e', 'f']]
plots = [dict(label='a', cycle=100000, species=1, field='by', vectors=['jtx', 'jty'], vector_color=None, cbar=True, legend=True,
              scalar_cmap=cl.bkr_extra_cmap, vector_cmap=cc.cm.fire, vector_label=r'$\mathbf{j}$ [nA/m$^2$]'),
         dict(label='b', cycle=202500, species=1, field='by', vectors=['jtx', 'jty'], vector_color=None, cbar=True, legend=True,
              scalar_cmap=cl.bkr_extra_cmap, vector_cmap=cc.cm.fire, vector_label=r'$\mathbf{j}$ [nA/m$^2$]'),
         dict(label='c', cycle=100000, species=1, field='vx0', vectors=['vx0', 'vy0'], vector_color=None, cbar=True, legend=True,
              scalar_cmap=cmap, vector_cmap=None, vector_label=r'$\mathbf{v}_{\text{el}}$', scalar_label=r'$\text{v}_{\text{x, el}}$'),
         dict(label='d', cycle=202500, species=1, field='vx0', vectors=['vx0', 'vy0'], vector_color=None, cbar=True, legend=True,
              scalar_cmap=cmap, vector_cmap=None, vector_label=r'$\mathbf{v}_{\text{el}}$', scalar_label=r'$\text{v}_{\text{x, el}}$'),
         dict(label='e', cycle=100000, species=1, field='pperp10', vectors=['jtx', 'jty'], vector_color=None, cbar=True, legend=True,
              scalar_cmap=cmap_turbo, vector_cmap=None, vector_label=r'$\mathbf{j}$ [nA/m$^2$]', scalar_label=r'$\text{p}_{\perp 1, \text{el}}$'),
         dict(label='f', cycle=202500, species=1, field='pperp10', vectors=['jtx', 'jty'], vector_color=None, cbar=True, legend=True,
              scalar_cmap=cmap_turbo, vector_cmap=None, vector_label=r'$\mathbf{j}$ [nA/m$^2$]', scalar_label=r'$\text{p}_{\perp 1, \text{el}}$'),
         ]

field_plot_opts = dict(
                       contour=False,
                       contour_fill=True,
                       contour_levels=10,
                       cmap=None,
                       norm=None,
                       alpha=1.,
                       cbar=True,
                       cbar_loc='bottom right',
                       label=None,
                       dark_mode=dark_mode,
                       )

vector_plot_opts = dict(
                        density=1.2, # float or (float, float)
                        lw=2, # linewidth of streamlines
                        lw_min=0.2,
                        lw_max=2,
                        # stream_seed_points=stream_seed_points , # array of points to seed streamlines
                        broken_streamlines=True,
                        smooth_field=True,
                        # seed_point_size=25,
                        # seed_point_color='blue',
                        arrowstyle='-|>',
                        # arrowstyle='->',
                        arrowsize=1.5,
                        average_cells=5,
                        zorder=0,
                        # patheffects_color='w',
                        patheffects_alpha=0.5,
                        patheffects_lw=1.,
                        color='w',
                        cmap=None,
                        norm=None,
                        alpha=0.8,
                        cbar=True,
                        cbar_loc='bottom left',
                        cbar_alpha=0.8,
                        label=None,
                        dark_mode=dark_mode,
                        minlength=0.11,
                        maxlength=1.5,
                        )

plot.configure_matplotlib(dark_mode, transparent)

# Size of the figure
fig, axes = plt.subplot_mosaic(layout,
                               figsize=(10.5, 10),
                               constrained_layout=True,
                               # height_ratios=[2, 0.8, 0.8],
                               sharex=True,
                               sharey=True,
                               )

# Set small padding between subplots
fig.set_constrained_layout_pads(w_pad=-0./72., h_pad=-0./72., hspace=0, wspace=0)

# Annotate axes with axes labels
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
        selection.calculate()

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
    if p['label'] in ['a', 'b']:
        scalar_range = [-5, 5]
    vector_pretty_name = units.pretty_name(p['vectors'], LaTeX=LaTeX)
    vector_range[0] = 0
    if p['label'] in ['e', 'f']:
        vector_range = [0, 3]
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
        im_bg = plot.field_slice(ax,
                                 scalar_field,
                                 selection=selection,
                                 **field_plot_opts,
                                )

    if vector_field is not None:
        color = (vector_color if vector_color is not None else vector_plot_opts.get('color', 'black'))
        vector_plot_opts['color'] = color
        vector_plot_opts['cmap'] = cmap_vector
        vector_plot_opts['norm'] = norm_vector
        vector_plot_opts['label'] = vector_label
        vector_plot_opts['color'] = 'k' if p['label'] in ['c', 'd'] else 'w'
        if p['label'] in ['a', 'b', 'e', 'f']:
            vector_plot_opts['dark_mode'] = True
            vector_plot_opts['cbar_alpha'] = 0.8
        else:
            vector_plot_opts['dark_mode'] = False
            vector_plot_opts['cbar_alpha'] = 0.8
        # color = [0,0,0,0.3]
        im_stream = plot.streamlines(ax,
                                     vector_field,
                                     selection, # selection object for bins and limits
                                     scale=vector_scale*1, # reference size
                                     **vector_plot_opts,
                                     )

        if p['label'] in ['c', 'd']:
            from pypic.fields import find_dipolarizations
            shapes = find_dipolarizations(ax, selection,
                                          'bz',
                                          min_length=14,
                                          dbz=2,
                                          x_lims=[-20, -10],
                                          )
            for shape in shapes:
                plot.shape(ax, selection, shape, dark_mode=True,
                           cmap = 'binary', alpha = 0.7, smooth_std=1)
        if p['label'] in ['c', 'd']:
            vx = vector_field[0]
            from pypic.fields import find_reversals
            find_reversals(ax, selection, vx, color='w', first_only=False)

axes_opts_phys = dict(draw_radii=[5,8],
                 planet=True,
                 axis_lines=False,
                 grid_lines=True,
                 labels=True,
                 minor_labels=True,
                 dark_mode=dark_mode,
                 transparent=transparent,
                 alpha=0.7,
                 title=None,
                 x_label=r'$\text{x}_{\text{GSM}}$ [R$_E$]',
                 y_label=r'$\text{y}_{\text{GSM}}$ [R$_E$]',
                 tick_loc = [],
                 label_loc = [],
                 tick_marks = ['left', 'bottom', 'top', 'right'],
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

# Automatically configure axes
labels = np.array(layout).flatten()
layout_shape = np.array(layout).shape
for p in plots:
    coord = p.get('coord', 'phys')
    if coord == 'phys':
        opts = axes_opts_phys
    else:
        opts = axes_opts_other
    row = np.where(np.array(layout) == p['label'])[0]
    col = np.where(np.array(layout) == p['label'])[1]
    last_row = row == layout_shape[0]-1
    last_col = col == layout_shape[1]-1
    first_row = row == 0
    first_col = col == 0
    if not first_col and not last_row:
        opts['tick_loc'] = []
        opts['label_loc'] = []
        plot.configure_axes(axes[p['label']], selection, **opts)
    if first_col and not last_row:
        opts['tick_loc'] = ['left']
        opts['label_loc'] = ['left']
        plot.configure_axes(axes[p['label']], selection, **opts)
    if first_col and last_row:
        opts['tick_loc'] = ['left', 'bottom']
        opts['label_loc'] = ['left', 'bottom']
        plot.configure_axes(axes[p['label']], selection, **opts)
    if not first_col and last_row:
        opts['tick_loc'] = ['bottom']
        opts['label_loc'] = ['bottom']
        plot.configure_axes(axes[p['label']], selection, **opts)

# Otherwise configure manually
# axes_opts_phys['tick_loc'] = ['left', 'bottom']
# axes_opts_phys['label_loc'] = ['left', 'bottom']
# plot.configure_axes(axes['a'], selection, **axes_opts_phys)

fig.align_ylabels([axes['a'], axes['b'], axes['c']])

if plot_progress_bar:
    # Automatically plot progress bar at the top of the first row
    labels = np.array(layout).flatten()
    # find the axes in the first row
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
    savepath = os.path.join(selection.figures_dir, f'{figure_name}.pdf')
    print(f'Saving figure to {savepath}...')
    plt.savefig(savepath,
                dpi=dpi,
                bbox_inches='tight',
                pad_inches=0.01, #default 0.1
                # facecolor=ax.get_facecolor(),
                transparent=transparent,
                )
