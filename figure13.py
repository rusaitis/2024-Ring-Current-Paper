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
plot_progress_bar = True
LaTeX = True
clip_on = False

# --- Particle Selection of ~10,000 ions for the 2024 paper was done with the following code
# print(f'unique q = {len(df["q"].unique()):,d}')
# dfpick1 = df[(df['cycle'] == 10000)]
# dfpick1 = dfpick1[(dfpick1['energy'] <= 20)]
# dfpick1 = dfpick1[(dfpick1['x'] < -15)]
# print(f'unique q1 = {len(dfpick1["q"].unique()):,d}')
# dfpick2 = df[(df['cycle'] == 202500)]
# dfpick2 = dfpick2[(dfpick2['x'] > -10)]
# print(f'unique q2 = {len(dfpick2["q"].unique()):,d}')
# q_pick1 = dfpick1['q'].unique()
# q_pick2 = dfpick2['q'].unique()
# q_pick = np.intersect1d(q_pick1, q_pick2)
# print(f'len q_pick = {np.size(q_pick)}')
# dfpick = df[df['q'].isin(q_pick)]
# dfpick.drop_duplicates(subset=['cycle', 'q'], inplace=True)
# dfpick = dfpick.sort_values(by=['cycle'])
# dfpick.to_hdf('ion_trajectories.h5', key='dfpick', mode='w') (the file available in the repository)

# --- Ions analyzed in detail (3 ions chosen for the 2024 ring current paper)
q_picks = [
        # loops
        1.3494941130331822e-07, # 0:
        1.6028904616860857e-07, # 1: loop
        7.2402733192390465e-06, # 2: loop at y=-6
        1.4862602471825898e-05, # 3: loop at -12
        5.121366157779523e-06, # 4: speiser + loop

        # speiser
        6.560770157438584e-06, # 5: speiser, field-aligned
        1.2436856568558564e-07, # 6: speiser  well-behaved
        1.2611442151732697e-07, # 7: speiser well-behaved
        5.654568722021194e-07, # 8: speiser well-behaved
        1.7780243648800112e-06, # 9: speiser well-behaved

        #"cucumber" orbits
        4.1837769706707496e-05, # 10: mirror at midnight
        2.11567834399194e-06, # 11: cucumber orbit
        1.4420594913941532e-05, # 12: mirror at x=-15

        # acc at recon
        3.3215662241919516e-06, # 13: acc at x=-28
        3.3152707939724956e-07, # 14: acc x > -20 ***
        1.282252834724002e-06, # 15: acc x > -20 ***
        1.6887829009941057e-06, # 16: acc x > -20 ***
        1.2103441776070789e-05, # 17: acc x > -20 *

        # acc at far-away recon
        2.473737900941003e-07, # 18: ion 18 acc at recon, field-aligned ***
        1.0501074974982799e-07, # 19: ion 14 big acc at recon
        1.9213097110238348e-07, # 20: ion 17 big acc at recon
        2.838706059161212e-07, # 21: ion 19 acc at recon, field-aligned *
        4.930373790654665e-07, # 22: ion 21 acc at recon
        6.842318677465078e-07, # 23: ion 23 big acc at recon
        9.59932499291523e-07, # 24: ion 24 big acc at recon
        1.0074053157212543e-06, # 25: ion 25 big acc at recon
        3.6590289371712924e-06, # 26: ion 28 big acc at recon **
        6.813941880890898e-06, # 27: ion 29 two acc at recon *
        1.498998813356103e-05, # 28: interesting kink and acc to 100keV ***

        # others
        1.3939649755211447e-06, # 29: circular gyration at 9
        ]

# --- Chosen ion IDs for the 2024 ring current paper
# i_pick = 6 # ion in Figure 11
# i_pick = 15 # ion in Figure 12
i_pick = 29 # ion in Figure 13 (df_paper_205k_all_124368_fields.h5)


sim = pypic.ipic3D()
# cycles_iter = [40000, 151500, 202500] # Ion 6 *
# cycles_plot_field = [40000, 151500] # Ion 6 *

# cycles_iter = [76000, 138000, 165000, 202500] # Ion 15 *
# cycles_plot_field = [76000, 138000, 165000] # Ion 15 *

cycles_iter = [106000, 142500, 202500] # Ion 29 *
cycles_plot_field = [142500]

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
selection_view.min_phys = [-30, -7, -2.5] 
selection_view.max_phys = [  0,  7,  2.5]

selection_fl = selection.duplicate()
selection_fl.min_phys = [-33, -15, -6.5]
selection_fl.max_phys = [ -5,  15,  6.5]

# Figure export
selection.figures_dir = 'figures'
figure_name = 'figure13'


from pypic.input_output import read_particle_dataframe
particle_filename = 'ion_trajectories.h5'
filename = f'{particle_filename.split('.')[0]}_{int(abs(q_picks[i_pick])*1e12)}_fields.h5'
filepath = os.path.join(selection.data_dir, filename)
if not os.path.exists(filepath):
    filepath = os.path.join(selection.sim.local_data_dir, filename)
if not os.path.exists(filepath):
    raise FileNotFoundError(f'File not found: {filepath}')

df = read_particle_dataframe(filepath, read_fields=True)
print(f'Read particles.')
q_pick = q_picks[i_pick]

dfpick = df

from pypic.particles import calculate_derivate_quantities
dfpick = calculate_derivate_quantities(dfpick, selection=selection)

cycles0 = np.arange(10000, 132000, 2000)
cycles1 = np.arange(132000, 202500+500, 500)
cycles = np.concatenate((cycles0, cycles1))

import colorcet as cc
cmap = mpl.colormaps['rainbow']
cmap_turbo = mpl.colormaps['turbo']
cmap = cl.bkr_extra_cmap
cmap_particle_traces = mpl.colormaps['turbo']

particle_trace_key = 'energy'
opt = dict(species=selection_view.species, coord='phys')
traces_units, traces_scale, traces_range = units.info(particle_trace_key,
                                   selection_view, **opt)
cmin_particle_traces, cmax_particle_traces = traces_range
norm_particle_traces = mpl.colors.Normalize(vmin=cmin_particle_traces, vmax=cmax_particle_traces)

axis_label = 'a'
annotation = True

layout = [['a', 'a'],
          ['b', 'c'],
          ['d', 'd'],
          ]
plots = [
         dict(label='a', cycle=cycle, species=1, field='bz', vectors=None, vector_color=None, cbar=True, legend=True,
              scalar_cmap=cmap, vector_cmap=None, vector_label=None, annotation=True, progress_bar=False),
         dict(label='b', cycle=cycle, species=1, field='bz', vectors=None, vector_color=None, cbar=True, legend=True,
              scalar_cmap=cmap, vector_cmap=None, vector_label=None, annotation=True, progress_bar=False),
         dict(label='c', cycle=cycle, species=1, field='bz', vectors=None, vector_color=None, cbar=True, legend=True,
              scalar_cmap=cmap, vector_cmap=None, vector_label=None, annotation=True, progress_bar=False),
         dict(label='d', cycle=cycle, species=1, field=None, vectors=None, vector_color=None, cbar=True, legend=True,
              scalar_cmap=cmap, vector_cmap=None, vector_label=None, annotation=True, progress_bar=False),
         ]

def plot_fieldline(ax, selection, center, alpha=None, color=None):
    boundary_conditions = {'r_min': 3.,
                           'r_max': 33.,
                           'lower_bound': selection.min_phys,
                           'upper_bound': selection.max_phys,
                           'max_iter': 1e4,
                           'verbose': False,
                           }
    fl = fieldline(r=center,
                   # step=0.1, # 0.05-0.2 for RK4, 0.01-0.05 for Euler
                   step=0.03, #0.03
                   method='RK4',
                   backtrace=False,
                   B_func=selection.f_B,
                   boundary_conditions=boundary_conditions)
    fl.trace()
    color = 'white' if dark_mode else 'black'
    if alpha is None:
        alpha = 0.99 if cycle < 160000 else 0.4
        alpha = 0.4 if cycle < 110000 else alpha
    lc = fl.plot_fieldline(ax=ax,
                      color_key='bz', # 'b', 'bz', 'n', 'errors', 'markers'
                      # color='k', # single color
                      # color=fl_color, # single color
                      cmap=cmap, # overrides default colors
                      # norm=norm, # overrides default normalization
                      lw=3.2, # line width
                      alpha=alpha, # transparency
                      arrows=True, # add arrows to show field direction
                      end_points=True, # add circles to end points
                      colorbar=False,
                      clip_on=False,
                      selection=selection,
                      intersections=False,
                      zorder=70,
                      draw=True,
                      # arrow_scale=25,
                      # arrow_color='white',
                      arrow_color=None,
                      arrow_alpha=0.6,
                      path_effects=True,
                      path_effects_alpha=0.8,
                      path_effects_lw=1.3,
                      path_effects_color='white',
                      path_effects_type='stroke',
                      )
    points = fl.r

    return lc, points[:,-1]

particle_trace_opts = dict(cmap=cmap_particle_traces,
                           norm=norm_particle_traces,
                          )

field_plot_opts = dict(
                       contour=True,
                       contour_fill=True,
                       contour_levels=10,
                       cmap=None,
                       norm=None,
                       alpha=0.8,
                       cbar=True,
                       cbar_loc='bottom right',
                       label=None,
                       dark_mode=dark_mode,
                       clip_on=clip_on,
                       )

plot.configure_matplotlib(dark_mode, transparent)

# Size of the figure
fig, axes = plt.subplot_mosaic(layout,
                               figsize=(12, 14),
                               per_subplot_kw={('a', 'b', 'c'): {'projection': '3d', 'computed_zorder': False}},
                               gridspec_kw={'height_ratios': [3, 2, 1],
                                            'wspace': 0.8, 'hspace': 0.1},
                               )

# Plot each panel
for p in plots:
    print(f'Plotting panel {p["label"]}...')
    ax = axes[p['label']]
    selection = p.get('selection', selection)
    coord = p.get('coord', 'phys')
    cycle = p.get('cycle', selection.cycle)


    # for label, ax in axes.items():
    annotate = p.get('annotation', False)
    if annotate:
        offset=(-0.28, 0)
        if p['label'] == 'a':
            offset=(-0.6,-0.01)
        if p['label'] == 'd':
            offset=(-0.1,0.17)
        plot.annotate_axes(ax,
                           p['label'],
                           fontsize=37,
                           dark_mode=dark_mode,
                           bg_color='gray',
                           alpha=0.3,
                           offset=offset,
                           )
    progress_bar = p.get('progress_bar', False)
    if progress_bar:
        plot.progress_bar(cycle,
                          sim.cycle_limits,
                          units.pretty_time(sim.dt_phys*cycle, LaTeX=LaTeX),
                          ax=ax,
                          margin=0.02,
                          fontsize=22,
                          extra_height=6,
                          width=0.4,
                          bg_color='white',
                          text_color='black',
                          alpha=0.5,
                          )

    if p['label'] == 'd':
        x_lines=[t*sim.dt_phys for t in cycles_iter]
        # shift the first and last line to avoid overlap with the axis
        x_lines[0] += 0.2
        x_lines[-1] -= 0.2

        text_color = 'white' if dark_mode else 'black'
        color = cl.colors[0] if dark_mode else cl.colors_dark[0]
        grid_color = 'white' if dark_mode else 'black'

        colors = cl.colors if dark_mode else cl.colors_dark
        info2_ax, info2_ax2 = plot.plot_data_dashboard(ax, selection, dfpick,
                            key=[
                                 'Edotv',
                                 'EdotB',
                                 ],
                            key2=['energy'],
                            x_key='time',
                            edgecolor='k',
                            labels=[r'$\mathbf{E} \cdot {\hat{\mathbf{v}}}$',
                                    r'$\mathbf{E} \cdot {\hat{\mathbf{B}}}$',
                                    ],
                            color=colors,
                            text_color='k',
                            fontsize=16,
                            legend=True,
                            clip_on=False,
                            normalize=False,
                            highlight=np.asarray(cycles_iter)*sim.dt_phys,
                            ylim = [-7,7],
                            ylim2 = [0,100],
                            ylabel=r'[mV/m]',
                            ylabel2='Energy [keV]',
                            xlabel='Time [s]',
                            lw=2.2,
                            lw2=2.4,
                            alpha=0.7,
                            alpha2=0.7,
                            )
        continue

    # --- FIELD PLOTS
    # selection.calculate(quick=True)

    # --- SELECT FIELDS
    scalar_field = selection.get_field(p['field'])

    # --- LABELS 
    scalar_label = units.pretty_name(p['field'], LaTeX=LaTeX)

    # --- GET UNITS AND LIMITS
    opt = dict(species=selection.species, coord='phys')
    scalar_units, _, scalar_range = units.info(p['field'], selection, **opt)

    scalar_label = p.get('scalar_label', scalar_label)
    if scalar_units is not None and scalar_label is not None:
        scalar_label += f' [{scalar_units}]'

    # --- COLORMAPS
    cmap_scalar = p['scalar_cmap']
    norm_scalar = plot.color_norm(scalar_range, log=False)
    selection.cmap = cmap_scalar
    selection.norm = norm_scalar

    # --- ADJUST COLORMAP SATURATION
    cmap_scalar.set_over(cl.adjust_lightness(cmap_scalar(1.0), 1.2))
    cmap_scalar.set_under(cl.adjust_lightness(cmap_scalar(0.0), 1.2))

    if p['label'] in ['a']:
        ax.set_proj_type('persp', focal_length=0.8)
        ax.view_init(elev=22, azim=-45)
    if p['label'] in ['b']:
        ax.set_proj_type('persp', focal_length=0.8)
        ax.view_init(elev=90, azim=-90, roll=0) #xy
    if p['label'] in ['c']:
        ax.set_proj_type('persp', focal_length=0.8)
        ax.view_init(elev=0, azim=-90, roll=0) #xz

    # cycles_iter = [cycle]
    for cycle in cycles_iter:
        print(f'Iterating over cycle = {cycle}')
        df0 = dfpick[dfpick['cycle'] <= cycle]
        last_pt = [df0['x'].iloc[-1], df0['y'].iloc[-1], df0['z'].iloc[-1]]
        selection.center_phys = last_pt
        selection.delta_phys =  [7, 7, 0]
        selection.cycle = cycle
        selection_view.cycle = cycle

        # from pypic.input_output import save_hdf5_slice
        # save_hdf5_slice(selection=selection,
        #                 dtype='float32',
        #                 save_keys=['Bx', 'By', 'Bz'],
        #                 cut_to_selection=False,
        #                 )

        # --- PLOT FIELD
        from pypic.calculate import calculate_fields, calculate_quick_3Dfields
        # selection.field = None
        selection.f_B = None
        calculate_quick_3Dfields(selection)

        # selection.calculate(quick=True)
        scalar_field = selection.get_field(p['field'])

        df_q = dfpick

        # --- Filter by time
        df_history = df_q[df_q['cycle'] <= cycle]
        draw_full_trace = True
        if draw_full_trace:
            seg_x = df_history['x']
            seg_y = df_history['y']
            seg_z = df_history['z']
            seg_c = df_history['energy']
            
            from pypic.geometry import colored_line
            col = colored_line(ax,
                         df_history,
                         # c='energy',
                         lw=3.2,
                         ls='solid',
                         alpha=1.,
                         clip_on=False,
                         cmap=cmap_particle_traces,
                         norm=norm_particle_traces,
                         zdir='y',
                         zorder=100,
                        )
            c = cmap_particle_traces(norm_particle_traces(df_history['energy'].iloc[0]))
            cin = cl.adjust_lightness(c, 0.8)
            cout = cl.adjust_lightness(c, 1.2)
            # Start position
            ax.plot(df_history['x'].iloc[0], df_history['y'].iloc[0], df_history['z'].iloc[0],
                       # c=cmap_particle_traces(norm_particle_traces(df_history['energy'].iloc[0])),
                       c=cin,
                       ms=8, marker='o', ls='', clip_on=False, alpha=0.4, zorder=80,
                       # edgecolor='black',
                       # mec='white',
                       mec=cout,
                       mew=0.7,
                    )

            if p['label'] == 'a':
                step = [0, 0, 0.8]
                line_correction = [0, 0, -0.32]
            elif p['label'] == 'b':
                step = [2.5, -2, 0]
                line_correction = [0, .5, 0]
            elif p['label'] == 'c':
                step = [0, 0, 6]
                line_correction = [0, 0, -.8]

            # annotation text
            ax.text(df_history['x'].iloc[-1]+step[0], df_history['y'].iloc[-1]+step[1], df_history['z'].iloc[-1]+step[2],
                    f'{cycle*sim.dt_phys:.0f} s',
                    zdir=None,
                    color='white',
                    fontsize=14,
                    ha='center',
                    va='center',
                    bbox=dict(facecolor='k', alpha=0.3, edgecolor='w', lw=0.5,
                              boxstyle='round,pad=0.5'),
                    zorder=150,
                    clip_on=False,
                    )
            # draw connecting line
            ax.plot([df_history['x'].iloc[-1], df_history['x'].iloc[-1]+step[0]+line_correction[0]],
                    [df_history['y'].iloc[-1], df_history['y'].iloc[-1]+step[1]+line_correction[1]],
                    [df_history['z'].iloc[-1], df_history['z'].iloc[-1]+step[2]+line_correction[2]],
                    c='k', lw=2.5, ls='-', alpha=0.4, zorder=80, clip_on=False,)

            # End position
            c = cmap_particle_traces(norm_particle_traces(df_history['energy'].iloc[-1]))
            cin = cl.adjust_lightness(c, 0.8)
            cout = cl.adjust_lightness(c, 1.2)
            ax.plot(df_history['x'].iloc[-1], df_history['y'].iloc[-1], df_history['z'].iloc[-1],
                       # c=cmap_particle_traces(norm_particle_traces(df_history['energy'].iloc[-1])),
                       c=cin,
                       # cmap=cmap_particle_traces, norm=norm_particle_traces,
                       ls='', marker='o', clip_on=False,
                       # mec='white',
                       mec=cout,
                       mew=0.7,
                       ms=8, alpha=0.8, zorder=80)

            plot.plot_intersections(ax,
                               df_history,
                               # selection=selection_view,
                               clip_on=False,
                               radius=0.15,
                               alpha=0.1,
                               lw=1.3,
                               color='white',
                               z_intersect=0,
                               )

        draw_particle_vector = True
        if draw_particle_vector:
            from pypic.plot import draw_particle_vectors
            color_B = cl.colors[1] if dark_mode else cl.colors_dark[1]
            color_E = cl.colors[0] if dark_mode else cl.colors_dark[0]
            draw_particle_vectors(ax,
                                  df_history,
                                  all_positions=False,
                                  norm_B=4,
                                  norm_E=2,
                                  color_B=color_B,
                                  color_E=color_E,
                                  lw=4,
                                  zorder=150,
                                  fontsize=16,
                                  )

        # Start position
        ax.scatter(df_history['x'].iloc[0], df_history['y'].iloc[0], df_history['z'].iloc[0],
                   s=19, marker='o', color='white', alpha=0.5, zorder=-20)

        # End position
        ax.scatter(df_history['x'].iloc[-1], df_history['y'].iloc[-1], df_history['z'].iloc[-1],
                   c=df_history['energy'].iloc[-1],
                   cmap=cmap_particle_traces, norm=norm_particle_traces,
                   s=12, alpha=0.5, zorder=20, clip_on=False)

        draw_fieldlines = True
        if draw_fieldlines:
            seed_point = [df_history['x'].iloc[-1], df_history['y'].iloc[-1], df_history['z'].iloc[-1]]
            selection_fl.f_B = selection.f_B
            if p['label'] == 'c':
                alpha = 0.3
            else:
                alpha = 0.99
            lc, _ = plot_fieldline(ax, selection_fl, seed_point, alpha)

        if scalar_field is not None and cycle in cycles_plot_field:
            field_plot_opts['cmap'] = cmap_scalar
            field_plot_opts['norm'] = norm_scalar
            field_plot_opts['label'] = scalar_label

            im_bg = plot.plot_surface_3d(ax, selection.x, selection.y, selection.z, scalar_field, **field_plot_opts)
            # im_bg.set_zorder(-100)


axes_opts_phys = dict(
                      draw_radii=[5,8],
                      planet=True,
                      axis_lines=True,
                      grid_lines=True,
                      x_lines=[-30, -25, -20, -15, -10],
                      y_lines=[-5, 0, 5],
                      labels=True,
                      minor_labels=True,
                      dark_mode=dark_mode,
                      transparent=True,
                      alpha=0.6,
                      title=None,
                      x_label=r'$\text{X}$',
                      y_label=r'$\text{Y}$',
                      z_label=r'$\text{Z}$',
                      tick_loc = [],
                      label_loc = [],
                      zoom=0.5,
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

axes_opts_phys['zoom'] = 0.45
selection_view.min_phys = [-15, -5, -2.5] 
selection_view.max_phys = [  0,  8,  2.5]
plot.configure_axes(axes['a'], selection_view, **axes_opts_phys)
axes_opts_phys['zoom'] = 0.7
axes_opts_phys['z_label'] = None
selection_view.min_phys = [-15,  0, -2.5] 
selection_view.max_phys = [ -5,  6,  2.5]
plot.configure_axes(axes['b'], selection_view, **axes_opts_phys)
axes_opts_phys['zoom'] = 0.67
axes_opts_phys['y_lines'] = []
axes_opts_phys['x_lines'] = [-30, -25, -20, -15, -10, -5]
axes_opts_phys['y_label'] = None
selection_view.min_phys = [-15, -7, -4.5] 
selection_view.max_phys = [  0,  7,  4.5]
plot.configure_axes(axes['c'], selection_view, **axes_opts_phys)
# axes['a'].view_init(elev=22, azim=-45)
# plot.configure_axes(axes['c'], selection_view, **axes_opts_other)

def update_cbar(ax, opts):
    cbar = plot.colorbar(ax,
                         norm=opts['norm'],
                         cmap=opts['cmap'],
                         label=opts['label'],
                         labelpad=-45,
                         fontsize=14,
                         orientation='horizontal',
                         location='top right',
                         dark_mode=True,
                         bg_color='white',
                         text_color='k',
                         edge_color='k',
                         edge_lw=0.2,
                         alpha=0.4,
                         pad=0.05,
                         margin=opts['margin'],
                         width=0.5,
                         height=0.02,
                         ticks=opts['ticks'],
                         clip_on=False,
                        )
    return cbar

particle_trace_opts['label'] = 'Energy [keV]'
particle_trace_opts['margin'] = (-0.4, 0.05)
particle_trace_opts['ticks'] = (0, 20, 40, 60, 80, 100)
cbar_field = update_cbar(axes['a'], particle_trace_opts)
field_plot_opts['margin'] = (0.2, 0.05)
field_plot_opts['ticks'] = (-20,-10,0,10,20)
cbar_field = update_cbar(axes['a'], field_plot_opts)

# fig.align_ylabels([axes['a'], axes['b'], axes['c']])

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
                bbox_inches='tight',
                pad_inches=0.05, #default 0.1
                # facecolor=ax.get_facecolor(),
                # transparent=transparent,
                )
