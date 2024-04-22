import matplotlib as mpl
import colorcet as cc
import pypic.colors as cl
import seaborn as sns


class ui_ipic3D:
    def __init__(self,
                ):
        self.selection = None
        self.selection_chunk = None
        self.selection_overview = None
        self.data_particle = None
        self.data_fields = None
        self.field_xy = None
        self.field_xz = None
        self.field_overview_xy = None
        self.field_overview_xz = None
        self.field_pick_xz = None
        self.field_pick_xy = None
        self.canvas_xy = None
        self.canvas_xz = None
        self.XYZ_particles = None
        self.rand_particle_indices = None
        self.data_selected = None
        self.fname = None
        self.fnames = None
        self.fcenters = None
        self.choice_ind = None
        # ui widgets
        self.x_slider = None
        self.y_slider = None
        self.z_slider = None
        self.cmap_slider = None
        self.bin_slider = None
        self.box_size_slider = None
        self.vaxes_radio = None
        self.species_radio = None
        self.cycles_radio = None
        self.refine_button = None
        self.info_button = None
        self.save_button = None
        self.im_xy = None
        self.im_xz = None
        self.im_vdf = None
        self.cbar_vdf = None
        self.cbar_field = None
        self.dash = None
        self.vdf_text = None
        self.selection_xy_box = None
        self.selection_xz_box = None
        self.line_xy_vert = None
        self.line_xy_hor = None
        self.line_xz_hor = None
        self.line_xz_vert = None
        self.im_mini_xy = None
        self.im_mini_xz = None
        self.im_pick_xz = None
        self.im_pick_xy = None
        self.scatter_particles = None
        self.scatter_particles_xz = None
        self.annotation_xz = None
        self.annotation_xy = None
        #axes
        self.vdf_ax = None
        self.xz_ax = None
        self.xy_ax = None
        self.xz_mini_ax = None
        self.xy_mini_ax = None
        # blit manager
        self.bm = None

display = dict(
    xlabel = r'$x_{GSM}$',
    ylabel = r'$y_{GSM}$',
    zlabel = r'$z_{GSM}$',
    fontsize = 14,
    figsize = (13, 8),
    dpi = 100,
    chunk_dx = 4,
    chunk_note = None,
    overview_min_GSM = [-33, -16, -6.5],
    overview_max_GSM = [ 13,  16,  6.5],
    pick_min_GSM = [-26, -10, -6],
    pick_max_GSM = [ 14,  10,  6],
    pick_size_GSM = [ 40,  20,  12],
    pick_size_cells = [ 10, 5, 3],
    pick_step = 4,
    position_step = 0.1,
    field_keys = ['Bx', 'By', 'Bz', 'B'],
    field_component = 1,
    field_cmin = 1,
    field_cmax = 10000,
    field_cmin_centered = -50,
    field_cmax_centered = 50,
    # field_cmap_centered = sns.color_palette("icefire", as_cmap=True), # Divergent
    # field_cmap_centered = cc.cm['coolwarm'], # Divergent (2nd fav)
    # field_cmap_centered = mpl.colormaps['twilight_shifted'], # Divergent (good!)
    field_cmap_centered = cc.cm['bkr'], # Divergent (fav)
    # field_cmap_centered = cc.cm['CET_C3'].reversed(),
    # field_cmap_centered = mpl.colormaps['turbo'], # Divergent (fav)
    # field_cmap = mpl.colormaps['turbo']
    # field_cmap = mpl.colormaps['RdBu']
    # field_cmap = mpl.colormaps['jet'].reversed()
    # field_cmap = cc.cm.bmy,
    field_cmap = 'turbo',
    # field_cmap = 'viridis',
    # field_cmap_div = cc.cm.bkr,
    field_label = r'$B_{T}$ [nT]',
    density_keys = ['N0', 'N1', 'rho0', 'rho1'],
    box_di_min = 0.5,
    box_di_max = 4,
    box_di_step = 0.1,
    vdf_bins = 50,
    vdf_bin_min = 20,
    vdf_bin_max = 200,
    vdf_bin_step = 1,
    vdf_cmin = -6-3,
    vdf_cmax = -2+3,
    vdf_crange = [-4,-1],
    vdf_cmap_step = 0.5,
    vdf_mean_color = 'blue',
    vdf_median_color = 'red',
    show_vdf_mean = False,
    show_vdf_median = False,
    vaxes_choices = [[r'$v_x$', r'$v_z$'],
                     [r'$v_y$', r'$v_z$'],
                     [r'$v_x$', r'$v_y$'],
                     [r'$v_{||}$', r'$v_{\perp}$']],
    vaxes_dict = {r'$v_x$': 0,
                  r'$v_y$': 1,
                  r'$v_z$': 2,
                  r'$v_{||}$': 2,
                  r'$v_{\perp}$': 3},
    vaxes_labels = {},
    vaxes_choice = 2,
    # vaxes_lims = [0.06, 0.01, 0.03, 0.04], #0.02
    vaxes_lims = [0.06, 0.005, 0.03, 0.04], #0.02
    # vaxes_lims = [0.02, 0.02, 0.03, 0.04], #0.02
    species_choices = ['electrons', 'ions'],
    species_labels = {'electrons': 0,
                      'ions': 1},
    species_choice = 0,
    cycles_choices = ['130000'],
    cycles_labels = {
                     '130000' : 0,},
    cycles_choice = 0,
    particle_display_number = 10000,
    particle_display_size = 0.5,
    particle_display_alpha = 0.5,
    particle_display_cmap = 'viridis',
    particle_display_color = 'white',
    )

# Generate Radio choice labels and dictionary
display["vaxes_labels"] = {f"{vc[1]} vs. {vc[0]}": i for i, vc in
                           enumerate(display["vaxes_choices"])}

slider_opts = dict(
    handle_style = {'size': 15, 'facecolor': cl.accent, 'edgecolor': '0.8'},
    track_color = cl.elements2,
    initcolor = cl.accent,
    color = cl.elements)

slider_range_opts = dict(
    handle_style = {'size': 15, 'facecolor': cl.accent, 'edgecolor': '0.8'},
    track_color = cl.elements2,
    color = cl.elements)

radio_opts = dict(label_props={'color': [cl.accent for x in display["vaxes_choices"]],
                               'fontsize': [display["fontsize"] for x in display["vaxes_choices"]]},
                  radio_props={
                        'facecolor': [cl.accent for x in display["vaxes_choices"]],
                        'edgecolor': [cl.accent for x in display["vaxes_choices"]],
                        's': [35 for x in display["vaxes_choices"]],})

radio_species_opts = dict(label_props={'color': [cl.accent for x in display["species_choices"]],
                               'fontsize': [display["fontsize"] for x in display["species_choices"]]},
                  radio_props={
                        'facecolor': [cl.accent for x in display["species_choices"]],
                        'edgecolor': [cl.accent for x in display["species_choices"]],
                        's': [35 for x in display["species_choices"]],})

radio_cycles_opts = dict(label_props={'color': [cl.accent for x in display["cycles_choices"]],
                               'fontsize': [display["fontsize"] for x in display["cycles_choices"]]},
                  radio_props={
                        'facecolor': [cl.accent for x in display["cycles_choices"]],
                        'edgecolor': [cl.accent for x in display["cycles_choices"]],
                        's': [35 for x in display["cycles_choices"]],})

# Remove the Toolbar
mpl.rcParams['toolbar'] = 'None'

# Remove the default key bindings
mpl.rcParams['keymap.grid'].remove('g')
mpl.rcParams['keymap.home'].remove('h')
mpl.rcParams['keymap.pan'].remove('p')
mpl.rcParams['keymap.back'].remove('left')
mpl.rcParams['keymap.back'].remove('c')
mpl.rcParams['keymap.forward'].remove('right')
mpl.rcParams['keymap.xscale'].remove('k')
mpl.rcParams['keymap.xscale'].remove('L')
mpl.rcParams['keymap.yscale'].remove('l')
mpl.rcParams['keymap.save'].remove('s')

# Add custom key bindings
def on_key_press(event, ui, update, show_info, refine_vdf, save_figs, save_vdf):
    if event.key in ('1'):
        try:
            ui.display["vaxes_choice"] = 0
            ui.vaxes_radio.set_active(ui.display["vaxes_choice"])
            update(0, ui)
            ui.bm.update()
        except:
            pass
    elif event.key in ('2'):
        try:
            ui.display["vaxes_choice"] = 1
            ui.vaxes_radio.set_active(ui.display["vaxes_choice"])
            update(0, ui)
            ui.bm.update()
        except:
            pass
    elif event.key in ('3'):
        try:
            ui.display["vaxes_choice"] = 2
            ui.vaxes_radio.set_active(ui.display["vaxes_choice"])
            update(0, ui)
            ui.bm.update()
        except:
            pass
    elif event.key in ('4'):
        try:
            ui.display["vaxes_choice"] = 3
            ui.vaxes_radio.set_active(ui.display["vaxes_choice"])
            update(0, ui)
            ui.bm.update()
        except:
            pass
    elif event.key in ('z', 'Z'):
        try:
            ui.display["species_choice"] = 0
            ui.selection.ns = ui.display["species_choice"]
            ui.species_radio.set_active(ui.display["species_choice"])
            # ui.load_all_data(redraw=True)
            update(0, ui)
            # ui.bm.update()
        except:
            pass
    elif event.key in ('x', 'X'):
        try:
            ui.display["species_choice"] = 1
            ui.selection.ns = ui.display["species_choice"]
            ui.species_radio.set_active(ui.display["species_choice"])
            # ui.load_all_data(redraw=True)
            update(0, ui)
            # ui.bm.update()
        except:
            pass
    elif event.key == 'i':
        try:
            ui.display["vdf_crange"][0] -= ui.display["vdf_cmap_step"]
            ui.cmap_slider.set_val(ui.display["vdf_crange"])
            update(0, ui)
        except:
            pass
    elif event.key == 'o':
        try:
            ui.display["vdf_crange"][0] += ui.display["vdf_cmap_step"]
            ui.cmap_slider.set_val(ui.display["vdf_crange"])
            update(0, ui)
        except:
            pass
    elif event.key == 'I':
        try:
            ui.display["vdf_crange"][1] -= ui.display["vdf_cmap_step"]
            ui.cmap_slider.set_val(ui.display["vdf_crange"])
            update(0, ui)
        except:
            pass
    elif event.key == 'O':
        try:
            ui.display["vdf_crange"][1] += ui.display["vdf_cmap_step"]
            ui.cmap_slider.set_val(ui.display["vdf_crange"])
            update(0, ui)
        except:
            pass
    elif event.key in ('j', 'J'):
        try:
            ui.display["vdf_bins"] -= ui.display["vdf_bin_step"]*10
            ui.bin_slider.set_val(ui.display["vdf_bins"])
            update(0, ui)
        except:
            pass
    elif event.key in ('k', 'K'):
        try:
            ui.display["vdf_bins"] += ui.display["vdf_bin_step"]*10
            ui.bin_slider.set_val(ui.display["vdf_bins"])
            update(0, ui)
        except:
            pass
    elif event.key in ('n', 'N'):
        try:
            ui.selection.delta_code = [x - ui.display["box_di_step"] for x in ui.selection.delta_code]
            ui.box_size_slider.set_val(ui.selection.delta_code[0])
            update(0, ui)
        except:
            pass
    elif event.key in ('m', 'M'):
        try:
            ui.selection.delta_code = [x + ui.display["box_di_step"] for x in ui.selection.delta_code]
            ui.box_size_slider.set_val(ui.selection.delta_code[0])
            update(0, ui)
        except:
            pass
    elif event.key == 'W':
        try:
            ui.selection.center_GSM[2] += ui.display["box_di_step"]
            ui.z_slider.set_val(ui.selection.center_GSM[2])
            update(0, ui)
        except:
            pass
    elif event.key == 'S':
        try:
            ui.selection.center_GSM[2] -= ui.display["box_di_step"]
            ui.z_slider.set_val(ui.selection.center_GSM[2])
            update(0, ui)
        except:
            pass
    elif event.key in ('a', 'A', 'left'):
        try:
            ui.selection.center_GSM[0] += ui.display["box_di_step"]
            ui.x_slider.set_val(ui.selection.center_GSM[0])
            update(0, ui)
        except:
            pass
    elif event.key in ('d', 'D', 'right'):
        try:
            ui.selection.center_GSM[0] -= ui.display["box_di_step"]
            ui.x_slider.set_val(ui.selection.center_GSM[0])
            update(0, ui)
        except:
            pass
    elif event.key in ('w', 'up'):
        try:
            ui.selection.center_GSM[1] -= ui.display["box_di_step"]
            ui.y_slider.set_val(ui.selection.center_GSM[1])
            update(0, ui)
        except:
            pass
    elif event.key in ('s', 'down'):
        try:
            ui.selection.center_GSM[1] += ui.display["box_di_step"]
            ui.y_slider.set_val(ui.selection.center_GSM[1])
            update(0, ui)
        except:
            pass
    elif event.key in (','):
        try:
            show_info(0, ui)
        except:
            pass
    elif event.key in ('ctrl+s', 'cmd+s', 'ctrl+S', 'cmd+S'):
        try:
            # save_figs(0, ui)
            refine_vdf(0, ui, save=True)
            save_vdf(0, ui, save=True)
            save_vdf(0, ui, bg='white', save=True)
        except:
            pass
    elif event.key in ('r', 'R'):
        try:
            refine_vdf(0, ui, save=False)
        except:
            pass
    elif event.key in ('e', 'E'):
        try:
            save_vdf(0, ui, save=False)
            # save_vdf(0, ui, bg='white')
        except:
            pass
    # fig.canvas.draw_idle()
