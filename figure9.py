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


sim = pypic.ipic3D()
# cycle = 10000
show_cycles = [10000, 100000, 155000, 202500] # 9000 ions
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
selection_view.min_phys = [-30, -10, -4.5] 
selection_view.max_phys = [  0,  10,  4.5]

# Figure export
selection.figures_dir = 'figures'
figure_name = 'figure9'

filename = 'ion_trajectories.h5'
filepath = os.path.join(selection.data_dir, filename)
if not os.path.exists(filepath):
    filepath = os.path.join(selection.sim.local_data_dir, filename)
if not os.path.exists(filepath):
    raise FileNotFoundError(f'File not found: {filepath}')

from pypic.input_output import read_particle_dataframe
df = read_particle_dataframe(filepath)
qs = df['q'].unique()
dfpick = df[df['q'].isin(qs[::1])] # Change 1 to 10 to plot only 10% of the ions
print(f'Unique particles = {len(dfpick["q"].unique()):,d}')

print(f'Calculating energy.')
energy_cycle = []
energy_time = []
energy_mean = []
energy_q1 = []
energy_q3 = []
energy_q10 = []
energy_q90 = []
cycles0 = np.arange(2000, 132000, 2000)
cycles1 = np.arange(132000, 202500+500, 500)
cycles = np.concatenate((cycles0, cycles1))
for cycle in cycles:
    dfpick_cycle = dfpick[dfpick['cycle'] == cycle]
    energy = dfpick_cycle['energy']
    energy_cycle.append(cycle)
    energy_time.append(cycle*sim.dt_phys)
    energy_mean.append(energy.mean())
    energy_q1.append(energy.quantile(0.25))
    energy_q3.append(energy.quantile(0.75))
    energy_q10.append(energy.quantile(0.10))
    energy_q90.append(energy.quantile(0.90))

print(f'Done calculating energy.')
df_energy = pd.DataFrame({'cycle': energy_cycle,
                          'time': energy_time,
                          'mean': energy_mean,
                          'q1':   energy_q1,
                          'q3':   energy_q3,
                          'q10':   energy_q10,
                          'q90':   energy_q90,
                          })


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

layout = [['a', 'b'],
          ['c', 'd'],
          ['e', 'f'],
          ['g', 'h'],
          ['i', 'i'],
          ]
plots = [
         dict(label='a', cycle=show_cycles[0], species=1, field=None, vectors=None, vector_color=None, cbar=True, legend=True,
              scalar_cmap=cmap, vector_cmap=None, vector_label=None, annotation=True, progress_bar=True),
         dict(label='b', cycle=show_cycles[0], species=1, field=None, vectors=None, vector_color=None, cbar=True, legend=True,
              scalar_cmap=cmap, vector_cmap=None, vector_label=None, annotation=True, progress_bar=False),
         dict(label='c', cycle=show_cycles[1], species=1, field=None, vectors=None, vector_color=None, cbar=True, legend=True,
              scalar_cmap=cmap, vector_cmap=None, vector_label=None, annotation=True, progress_bar=True),
         dict(label='d', cycle=show_cycles[1], species=1, field=None, vectors=None, vector_color=None, cbar=True, legend=True,
              scalar_cmap=cmap, vector_cmap=None, vector_label=None, annotation=True, progress_bar=False),
         dict(label='e', cycle=show_cycles[2], species=1, field=None, vectors=None, vector_color=None, cbar=True, legend=True,
              scalar_cmap=cmap, vector_cmap=None, vector_label=None, annotation=True, progress_bar=True),
         dict(label='f', cycle=show_cycles[2], species=1, field=None, vectors=None, vector_color=None, cbar=True, legend=True,
              scalar_cmap=cmap, vector_cmap=None, vector_label=None, annotation=True, progress_bar=False),
         dict(label='g', cycle=show_cycles[3], species=1, field=None, vectors=None, vector_color=None, cbar=True, legend=True,
              scalar_cmap=cmap, vector_cmap=None, vector_label=None, annotation=True, progress_bar=True),
         dict(label='h', cycle=show_cycles[3], species=1, field=None, vectors=None, vector_color=None, cbar=True, legend=True,
              scalar_cmap=cmap, vector_cmap=None, vector_label=None, annotation=True, progress_bar=False),
         dict(label='i', cycle=show_cycles[1], species=1, field=None, vectors=None, vector_color=None, cbar=True, legend=True,
              scalar_cmap=cmap, vector_cmap=None, vector_label=None, annotation=True, progress_bar=False),
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
                   step=0.1,
                   method='RK4',
                   backtrace=False,
                   B_func=selection.f_B,
                   boundary_conditions=boundary_conditions)
    fl.trace()
    lc = fl.plot_fieldline(ax=ax,
                      color_key='bz', # 'b', 'bz', 'n', 'errors', 'markers'
                      # color='white', # single color
                      # color=fl_color, # single color
                      cmap=cmap, # overrides default colors
                      # norm=norm, # overrides default normalization
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

particle_trace_opts = dict(cmap=cmap_particle_traces,
                           norm=norm_particle_traces,
                          )

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
                               figsize=(17, 22.5),
                               per_subplot_kw={('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'): {'projection': '3d', 'computed_zorder': False}},
                               gridspec_kw={'height_ratios': [2, 2, 2, 2, 1],
                                            'wspace': 0.1, 'hspace': 0.0},
                               )

# Plot each panel
for p in plots:
    print(f'Plotting panel {p["label"]}...')
    ax = axes[p['label']]
    selection = p.get('selection', selection)
    coord = p.get('coord', 'phys')
    cycle = p.get('cycle', selection.cycle)

    annotate = p.get('annotation', False)
    if annotate:
        offset=(-0.4,-0.01)
        if p['label'] == 'i':
            offset=(-0.02,0.35)
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
    # from pypic.input_output import save_hdf5_slice
    # save_hdf5_slice(selection=selection,
    #                 # filename=path,
    #                 dtype='float16',
    #                 save_keys=['Bx', 'By', 'Bz'],
    #                 cut_to_selection=False,
    #                 )

    if p['label'] == 'i':
        # print(f'Plotting dashboard')
        x_lines=[t*sim.dt_phys for t in show_cycles]
        x_lines[0] += 0.2
        x_lines[-1] -= 0.2
        plot.plot_data_dashboard(ax,
                                 selection,
                                 df_energy,
                                 key='mean',
                                 x_key='time',
                                 ylim=[0, 45],
                                 y_ticks=[0, 10, 20, 30, 40],
                                 x_lines=x_lines,
                                 fill=True,
                                 edgecolor=None,
                                 lw=1.4,
                                 color='white',
                                 text_color='black',
                                 xlabel='Time [s]',
                                 ylabel='Energy [keV]',
                                 grid_color='white',
                                 alpha=0.8,
                                 fontsize=18,
                                 legend=True,
                                 )

        for q in dfpick['q'].unique():
            df_q = dfpick[dfpick['q'] == q]
            ax.plot(df_q['cycle']*sim.dt_phys , df_q['energy'],
                    color='k', ls='-', lw=0.5, alpha=0.012, zorder=1)

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
    skip_calculation = True
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

    if p['label'] in ['a', 'c', 'e', 'g']:
        ax.set_proj_type('persp', focal_length=0.8)
        ax.view_init(elev=22, azim=-45)
    if p['label'] in ['b', 'd', 'f', 'h']:
        ax.set_proj_type('persp', focal_length=0.8)
        ax.view_init(elev=90, azim=-90, roll=0)

    cycles_iter = [cycle]
    for cycle in cycles_iter:
        print(f'Iterating over cycle = {cycle}')
        df0 = dfpick[dfpick['cycle'] <= cycle]
        last_pt = [df0['x'].iloc[-1], df0['y'].iloc[-1], df0['z'].iloc[-1]]
        selection.cycle = cycle
        selection_view.cycle = cycle

        # --- PLOT FIELD
        for q in dfpick['q'].unique():
            df_q = dfpick[dfpick['q'] == q]

            df_history = df_q[df_q['cycle'] <= cycle]
            color = 'white' if dark_mode else 'black'
            ax.plot(df_history['x'], df_history['y'], df_history['z'],
                    color=color, ls='-', lw=0.5, alpha=0.015, zorder=0, clip_on=False)

            draw_particle_vector = False
            if draw_particle_vector:
                color_B = cl.colors[1] if dark_background else cl.colors_dark[1]
                color_E = cl.colors[0] if dark_background else cl.colors_dark[0]
                draw_particle_vectors(ax, df_history,
                                      all_positions=False,
                                      norm_B=4,
                                      norm_E=2,
                                      color_B=color_B,
                                      color_E=color_E,
                                      lw=4,
                                      zorder=110,
                                      )

            guiding_center = False
            if guiding_center:
                df_q, R_gc = pypic.particles.find_guiding_center(df_q, selection_view)
                # ax.plot(*R_gc.T,
                #         color='white', ls='--', lw=2, alpha=0.6, zorder=10)
                ax.plot(df_q['x_gc'], df_q['y_gc'], df_q['z_gc'],
                        color='white', ls='--', lw=1.5, alpha=0.5, zorder=10)
                # draw_Bperp_circle(ax, df_q, selection_view)

                draw_Bperp_circle(ax, df_q, selection_view, spacing=2,
                                  color='yellow', alpha=0.5, lw=1, ls='-',
                                  mean_radius=False, center_EM=False,
                                  num_points=20)
                dp = df_q['dc'].mean()*selection.sim.scaling[0]
                Bpar = df_q['B_par'].mean()
                B = df_q['B_perp'].mean()
                print(f'Bpar = {Bpar} nT | B = {B} nT | dp = {dp} Re')
                title = f'B = {B:.1f} nT | ' + r'$r_{gyro}=$' + f'{dp:.2f}' + r' $R_e$'
                title += ' | Cycle=[185500,185550]'

                # add_particle_projection(info4_ax, selection_view, dfpick,
                #                   edgecolor=grid_color, lw=1, color='orange',
                #                   text_color='white', alpha=0.8,
                #                   fontsize=12)

            # Start position
            # ax.scatter(df_history['x'].iloc[0], df_history['y'].iloc[0], df_history['z'].iloc[0],
            #            s=19, marker='o', color='white', alpha=0.5, zorder=-20)

            # End position
            ax.scatter(df_history['x'].iloc[-1], df_history['y'].iloc[-1], df_history['z'].iloc[-1],
                       c=df_history['energy'].iloc[-1],
                       cmap=cmap_particle_traces, norm=norm_particle_traces,
                       s=12, alpha=0.5, zorder=20, clip_on=False)

            # ax.scatter(df_history['x'].iloc[-1], df_history['y'].iloc[-1], df_history['z'].iloc[-1],
                       # s=15, marker='o', color='y', alpha=0.8, zorder=10)

            draw_fieldlines = False
            dx = 0.2
            if draw_fieldlines:
                boundary_conditions = {'r_min': 3., 'r_max': 33.,
                                       'lower_bound': selection_fl.min_phys,
                                       'upper_bound': selection_fl.max_phys,
                                       'max_iter': 1e5,
                                       'verbose': True,
                                       'z_intersect': 0,
                                       # 'max_intersections': 1,
                                       }

                seed_point = [df_history['x'].iloc[-1], df_history['y'].iloc[-1], df_history['z'].iloc[-1]]
                seed_point1 = [df_history['x'].iloc[-1]+dx, df_history['y'].iloc[-1]+dx, df_history['z'].iloc[-1]+dx]
                seed_point2 = [df_history['x'].iloc[-1]+dx, df_history['y'].iloc[-1]-dx, df_history['z'].iloc[-1]+dx]
                seed_point3 = [df_history['x'].iloc[-1]-dx, df_history['y'].iloc[-1]+dx, df_history['z'].iloc[-1]+dx]
                seed_point4 = [df_history['x'].iloc[-1]-dx, df_history['y'].iloc[-1]-dx, df_history['z'].iloc[-1]+dx]
                seed_point5 = [df_history['x'].iloc[-1]+dx, df_history['y'].iloc[-1]+dx, df_history['z'].iloc[-1]-dx]
                seed_point6 = [df_history['x'].iloc[-1]+dx, df_history['y'].iloc[-1]-dx, df_history['z'].iloc[-1]-dx]
                seed_point7 = [df_history['x'].iloc[-1]-dx, df_history['y'].iloc[-1]+dx, df_history['z'].iloc[-1]-dx]
                seed_point8 = [df_history['x'].iloc[-1]-dx, df_history['y'].iloc[-1]-dx, df_history['z'].iloc[-1]-dx]
                # selection.cycle = int(df_history['cycle'].iloc[-1])
                print(f'cycle = {selection.cycle}')
                # calculate_fields(selection)
                # calculate_quick_3Dfields(selection)
                fl_opts = dict(step=0.03,
                               method='RK4',
                               backtrace=False,
                               B_func=selection.f_B,
                               boundary_conditions=boundary_conditions)
                color = 'white' if dark_background else 'black'
                alpha = 1 if cycle < 160000 else 0.4
                alpha = 0.4 if cycle < 110000 else alpha
                fl_plot_opts = dict(ax=ax,
                                    color_key='bz',
                                    # color=color, # single color
                                    # color=fl_color, # single color
                                    cmap=cmap_fl, # overrides default colors
                                    # norm=norm, # overrides default normalization
                                    lw=3.0, # line width
                                    alpha=alpha, # transparency
                                    arrows=True, # add arrows to show field direction
                                    arrow_alpha=0.5,
                                    arrow_color='white',
                                    end_points=True, # add circles to end points
                                    colorbar=False,
                                    clip_on=False,
                                    selection=selection,
                                    intersections=False,
                                    zorder=70,
                                    )
                fl = fieldline(r=seed_point, **fl_opts)
                fl.trace()
                # fl.plot_errors()
                fl.plot_fieldline(**fl_plot_opts)

    if scalar_field is not None:
        field_plot_opts['cmap'] = cmap_scalar
        field_plot_opts['norm'] = norm_scalar
        field_plot_opts['label'] = scalar_label

        selection_view.f_B = selection.f_B
        # for center in trace_centers:
        #     lc, _ = plot_fieldline(ax, selection_view, center)
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

axes_opts_phys['zoom'] = 0.5
plot.configure_axes(axes['a'], selection_view, **axes_opts_phys)
axes_opts_phys['zoom'] = 0.65
plot.configure_axes(axes['b'], selection_view, **axes_opts_phys)
axes_opts_phys['zoom'] = 0.5
plot.configure_axes(axes['c'], selection_view, **axes_opts_phys)
axes_opts_phys['zoom'] = 0.65
plot.configure_axes(axes['d'], selection_view, **axes_opts_phys)
axes_opts_phys['zoom'] = 0.5
plot.configure_axes(axes['e'], selection_view, **axes_opts_phys)
axes_opts_phys['zoom'] = 0.65
plot.configure_axes(axes['f'], selection_view, **axes_opts_phys)
axes_opts_phys['zoom'] = 0.5
plot.configure_axes(axes['g'], selection_view, **axes_opts_phys)
axes_opts_phys['zoom'] = 0.65
plot.configure_axes(axes['h'], selection_view, **axes_opts_phys)

def update_cbar(ax, opts):
    cbar = plot.colorbar(ax,
                         norm=opts['norm'],
                         cmap=opts['cmap'],
                         label=opts['label'],
                         labelpad=8,
                         fontsize=19,
                         orientation='horizontal',
                         location='bottom right',
                         dark_mode=True,
                         bg_color='gray',
                         text_color='k',
                         edge_color='k',
                         edge_lw=0.2,
                         alpha=0.4,
                         pad=0.05,
                         # margin=-0.15,
                         margin=(0.1,0.05),
                         width=0.8,
                         height=0.03,
                         # ticklabels=None,
                         ticks=[20,40,60,80,100],
                         clip_on=False,
                        )
    return cbar

# ui.field_opts = get_field_info(ui.selection, 'bz', ui.field_opts)
particle_trace_opts['label'] = 'Energy [keV]'
cbar_field = update_cbar(axes['g'], particle_trace_opts)

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
    savepath = os.path.join(selection.figures_dir, f'{figure_name}.jpg')
    print(f'Saving figure to {savepath}...')
    plt.savefig(savepath,
                dpi=dpi,
                bbox_inches='tight',
                pad_inches=0.05, #default 0.1
                # facecolor=ax.get_facecolor(),
                # transparent=transparent,
                )
