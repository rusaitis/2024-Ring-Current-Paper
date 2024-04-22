import numpy as np
from fast_histogram import histogram2d

def refine_vdf(val, ui, bg='black', v0=None, v1=None, weight=None,
               vel_labels=None, save=True):
    """ Plot a more detailed VDF plot in a new figure """
    import colorcet as cc
    import colors as cl
    import os

    # set mpl rc fontsize to 14
    mpl.rcParams.update({'font.size': 10})

    if bg == 'black':
        plt.style.use(['dark_background'])
        cl_text = 'white'
    else:
        plt.style.use(['default'])
        # plt.style.use(['fivethirtyeight'])
        cl_text = 'black'

    if v0 is None:
        v0, v1, weight, vel_labels = select_velocities(ui.data_selected,
                                                       ui.display,
                                                       vaxes_choice=ui.display["vaxes_choice"])
    # v0 = np.asarray(v0)*ui.selection.sim.c*1e-3
    # v1 = np.asarray(v1)*ui.selection.sim.c*1e-3
    mean_x = np.mean(v0)
    mean_y = np.mean(v1)
    std_x = np.std(v0)
    std_y = np.std(v1)
    median_x = np.median(v0)
    median_y = np.median(v1)

    xlabel=vel_labels[0]
    ylabel=vel_labels[1]
    max_lim = ui.display["vaxes_lims"][ui.display["species_choice"]]

    x = ui.selection.center_phys[0]
    y = ui.selection.center_phys[1]
    z = ui.selection.center_phys[2]
    species = ui.display["species_choices"][ui.selection.ns]
    dx = ui.selection.delta_code[0]
    dx_RE = ui.selection.delta_phys[0]
    N = v0.shape[0]

    # cmin=np.power(10., ui.cmap_slider.val[0])
    # cmax=np.power(10., ui.cmap_slider.val[1]*10)
    cmin=np.power(10., ui.display["vdf_crange"][0])
    cmax=np.power(10., ui.display["vdf_crange"][1])
    # cmin=np.power(10., -1)
    # cmax=np.power(10., 2)
    norm = mpl.colors.LogNorm(vmin=cmin, vmax=cmax)
    # norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)

    fig = plt.figure(figsize=(8.5, 7))
    gs = fig.add_gridspec(2, 3,
                          width_ratios=(4, 1, 0.2),
                          height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.4, hspace=0.15)
    # gs = plt.GridSpec(2, 2, hspace=0.2, wspace=0.2)
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_histx.axvline(x=0, linestyle='--', linewidth=2, color='white')
    ax_histy.axhline(y=0, linestyle='--', linewidth=2, color='white')

    # ax_dash = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[1, 2])

    Hx = histogram2d( v0, v1, bins = ui.display["vdf_bins"],
                              range = [[-max_lim, max_lim],[-max_lim, max_lim]],
                              weights = np.abs(weight),
                              # density = True,
                            )
    # hist2d
    from scipy.ndimage import gaussian_filter
    Hx = gaussian_filter(Hx, sigma=0.8)
    im = ax.imshow(Hx.T,
                   interpolation='nearest',
                   origin='lower',
                    # extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   extent=[-max_lim, max_lim, -max_lim, max_lim], #same result
                    # cmap=cc.cm.fire,
                    cmap=cl.cmap_vdf,
                   norm=norm,
                    # vmin=cmin, vmax=cmax,
                    # vmin=0, vmax=3.5,
                    # norm = norm,
                    # vmin=cmin, vmax=cmax,
                    )
    #contour
    imc = ax.contour(Hx.T,
                    extent=[-max_lim, max_lim, -max_lim, max_lim],
                    linewidths=1,
                    # cmap = 'turbo',
                    colors='white',
                    linestyles='solid',
                    alpha=0.7,
                    # levels = [0.1, 0.5, 1, 2],
                    )
    #contour labels
    ax.clabel(imc, inline=True, fontsize=10, colors='white', fmt='%0.2f')

    ax.text(0.03, 0.03, f'{species}',
            fontsize=15, 
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes,
            alpha=0.7,
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.15))

    ax.text(0.03, 0.97, r'$x_{phys}$' + f' = {x:5.01f} ' + r'$R_E$'+'\n'
                       +r'$y_{phys}$' + f' = {y:5.01f} ' + r'$R_E$'+'\n'
                       +r'$z_{phys}$' + f' = {z:5.01f} ' + r'$R_E$',
            fontsize=11, 
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes,
            alpha=0.7,
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.15))

    ax.text(0.97, 0.97, f'{N:,} particles'+'\n'
                       +f'bin = {dx:.1f} '+ r'$d_i$'
                       +f' ({dx_RE:.1f} ' + r'$R_E$' +')',
            fontsize=11,
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
            alpha=0.7,
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.15))

    ax.set_aspect('equal', 'box')
    ax.set_frame_on(False)
    ax.grid(color='white', linestyle='--', linewidth=0.8, alpha=0.25)
    ax.axvline(x=0, linestyle='--', linewidth=2, color='white')
    ax.axhline(y=0, linestyle='--', linewidth=2, color='white')

    if ui.display["show_vdf_mean"]:
        ax.errorbar(mean_x, mean_y, yerr=std_x/2, xerr=std_y/2, fmt='o',
                    color=ui.display["vdf_mean_color"], capsize=2,
                    label='mean particle velocity', alpha=0.4)
    if ui.display["show_vdf_median"]:
        ax.errorbar(median_x, median_y, yerr=std_x/2, xerr=std_y/2, fmt='o',
                    color=ui.display["vdf_median_color"], capsize=2,
                    label='median particle velocity', alpha=0.4)
    if ui.display["show_vdf_mean"] or ui.display["show_vdf_median"]:
        l = ax.legend(loc='lower right', fontsize=11, frameon=False, labelcolor='white')


    ax.set_xlabel(r'$v_x$', fontsize=14)
    ax.set_ylabel(r'$v_y$', fontsize=14)
    # ax.set_title('fast hist with gaussian smoothing' ,pad=300)
    # ax.set_title(f'Velocity Distribution Function for {species}', pad=20, color=cl_text)
    # ax_histx.text(0.6, 1.2, 'Velocity Distribution Function (electrons)',
    #         verticalalignment='center', horizontalalignment='center',
    #         transform=ax_histx.transAxes,
    #         color='white', fontsize=16)
    plot_marginals(ax_histx, Hx, axis='x', lims=[-max_lim, max_lim])
    plot_marginals(ax_histy, Hx, axis='y', lims=[-max_lim, max_lim])

    cbar = fig.colorbar(im, cax=cax, extend='both', label='counts')
    # ax.set_aspect('equal', adjustable=None, anchor='C', share=True)
    # plt.tight_layout()

    ax.xaxis.label.set_color(cl_text)
    ax.yaxis.label.set_color(cl_text)
    cbar.ax.yaxis.set_tick_params(color=cl_text)
    # change the direction of x tick labels by 45 degrees
    for label in ax.get_xticklabels():
        label.set_rot(45)
    for label in ax_histx.get_xticklabels():
        label.set_rot(45)


    if save:
        # save figure
        if bg == 'black':
            fname = f'VDF_{species}_{x:.01f}X_{y:.01f}Y_{z:.01f}Z_smoothed_white.png'
        else:
            fname = f'VDF_{species}_{x:.01f}X_{y:.01f}Y_{z:.01f}Z_smoothed_black.png'
        # print(fname)
        filename = os.path.join(ui.selection.output_dir, ui.selection.figure_dir, fname)
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        fig.clf()
        plt.close(fig)
    else:
        plt.show()
    return im, Hx

def save_vdf(val, ui, bg='black', save=True):

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import colors as cl
    import os

    if bg == 'black':
        plt.style.use(['dark_background'])
        cl_text = 'white'
    else:
        plt.style.use(['default'])
        # plt.style.use(['fivethirtyeight'])
        cl_text = 'black'


    v0, v1, weight, vel_labels = select_velocities(ui.data_selected, ui.display,
                                               vaxes_choice=ui.display["vaxes_choice"])
    mean_x = np.mean(v0)
    mean_y = np.mean(v1)
    std_x = np.std(v0)
    std_y = np.std(v1)
    median_x = np.median(v0)
    median_y = np.median(v1)
    # vlim=ui.display["vaxes_lims"][ui.display["species_choice"]]
    vlim=ui.display["vaxes_lims"][ui.selection.ns]

    x = ui.selection.center_phys[0]
    y = ui.selection.center_phys[1]
    z = ui.selection.center_phys[2]
    species = ui.display["species_choices"][ui.selection.ns]
    dx = ui.selection.delta_code[0]
    dx_RE = ui.selection.delta_phys[0]
    N = v0.shape[0]

    # set mpl rc fontsize to 14
    mpl.rcParams.update({'font.size': 13})

    # set mpl rc text color to black
    # mpl.rcParams['text.color'] = 'black'

    #new figure
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    if ui.display["show_vdf_mean"]:
        ax.errorbar(mean_x, mean_y, yerr=std_x/2, xerr=std_y/2, fmt='o',
                    color=ui.display["vdf_mean_color"], capsize=2,
                    label='mean particle velocity', alpha=0.6)
    if ui.display["show_vdf_median"]:
        ax.errorbar(median_x, median_y, yerr=std_x/2, xerr=std_y/2, fmt='o',
                    color=ui.display["vdf_median_color"], capsize=2,
                    label='median particle velocity', alpha=0.6)
    if ui.display["show_vdf_mean"] or ui.display["show_vdf_median"]:
        l = ax.legend(loc='lower right', fontsize=11, frameon=False, labelcolor='white')

    # Create a thin axis for a VDF colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
    ax.set_title(f'Velocity Distribution Function for {species}', pad=20, color=cl_text)

    ax.text(0.03, 0.97, r'$x_{phys}$' + f' = {x:5.01f} ' + r'$R_E$'+'\n'
                       +r'$y_{phys}$' + f' = {y:5.01f} ' + r'$R_E$'+'\n'
                       +r'$z_{phys}$' + f' = {z:5.01f} ' + r'$R_E$',
            fontsize=14, 
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes,
            alpha=0.7,
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.15))

    ax.text(0.97, 0.97, f'{N:,} particles'+'\n'
                       +f'bin = {dx:.1f} '+ r'$d_i$'
                       +f' ({dx_RE:.1f} ' + r'$R_E$' +')',
            fontsize=14,
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
            alpha=0.7,
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.15))

    ax.set_aspect('equal', 'box')
    ax.set_frame_on(False)
    ax.grid(color='white', linestyle='--', linewidth=0.8, alpha=0.25)
    ax.axvline(x=0, linestyle='--', linewidth=2, color='white')
    ax.axhline(y=0, linestyle='--', linewidth=2, color='white')
    im, cbar = plot_vdf_2d(ax=ax,
                                   x=v0, y=v1, weight=weight,
                                   bins=ui.display["vdf_bins"],
                                   vlim=vlim,
                                   lognorm=True,
                                   cmin=np.power(10., ui.display["vdf_crange"][0]),
                                   cmax=np.power(10., ui.display["vdf_crange"][1]),
                                   cmap=cl.cmap_vdf,
                                   cax=cax,
                                   xlabel=vel_labels[0] + ' [c]',
                                   ylabel=vel_labels[1] + ' [c]',
                                   )
    ax.xaxis.label.set_color(cl_text)
    ax.yaxis.label.set_color(cl_text)
    cbar.ax.yaxis.set_tick_params(color=cl_text)

    for label in ax.get_xticklabels():
        label.set_rot(45)

    if save:
        # save figure
        if bg == 'black':
            fname = f'VDF_{species}_{x:.01f}X_{y:.01f}Y_{z:.01f}Z_white.png'
        else:
            fname = f'VDF_{species}_{x:.01f}X_{y:.01f}Y_{z:.01f}Z_black.png'
        # print(fname)
        filename = os.path.join(ui.selection.output_dir, ui.selection.figure_dir, fname)
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        fig.clf()
        plt.close(fig)
    else:
        plt.show()

def plot_vdf_2d(ax=None,
                x=None,
                y=None,
                weight=None,
                bins=50,
                cax=None,
                cmin=None,
                cmax=None,
                cbar=None,
                cmap='turbo',
                norm=None,
                im=None,
                xlabel=None,
                ylabel=None,
                vlim=None,
                lognorm=True,
                update_only=False, # only update the data in the plot
                ):
    """ Plot a 2D VDF in a given axis """
    import matplotlib as mpl
    if norm is None:
        if cmin is not None and cmax is not None:
            if lognorm:
                norm = mpl.colors.LogNorm(vmin=cmin, vmax=cmax)
            else:
                norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
        else:
            norm = None

    if vlim is None:
        vlim = np.max(np.abs([x,y]))

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x)
    std_y = np.std(y)
    median_x = np.median(x)
    median_y = np.median(y)


    Hx = histogram2d(x,
                     y,
                     bins = bins,
                     range = [[-vlim, vlim],[-vlim, vlim]],
                     weights = np.abs(weight),
                     # density = True,
                     )

    from scipy.ndimage import gaussian_filter
    Hx = gaussian_filter(Hx, sigma=0.8)

    # ax.set_frame_on(False)
    # ax.grid(color='white', linestyle='--', linewidth=0.8, alpha=0.25)
    # ax.axvline(x=0, linestyle='--', linewidth=2, color='white')
    # ax.axhline(y=0, linestyle='--', linewidth=2, color='white')

    ax.errorbar(mean_x, mean_y, yerr=std_y/2, xerr=std_x/2, fmt='o',
                color='k', capsize=3,
                label='mean ion velocity', alpha=0.7)
    ax.errorbar(median_x, median_y, yerr=std_y/2, xerr=std_x/2, fmt='o',
                color='green', capsize=3,
                label='median ion velocity', alpha=0.7)
    l = ax.legend(loc='lower right', fontsize=12, frameon=True, labelcolor='white')

    if update_only is False:

        im = ax.imshow(Hx.T,
                       interpolation = 'nearest',
                       origin = 'lower',
                       extent = [-vlim, vlim, -vlim, vlim], #same result
                       cmap = cmap,
                       norm = norm,
                       )

        imc = ax.contour(Hx.T,
                        extent=[-vlim, vlim, -vlim, vlim],
                        linewidths=1,
                        # cmap = 'turbo',
                        colors='white',
                        linestyles='solid',
                        alpha=0.7,
                        # levels = [0.1, 0.5, 1, 2],
                        )
        #contour labels
        # ax.clabel(imc, inline=True, fontsize=10, colors='white', fmt='%0.2f')

        if cax is not None:
            cax.cla()
            cax.remove()

        if cax is None:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = mpl.pyplot.colorbar(im,
                                   cax=cax,
                                   label='counts',
                                   orientation='vertical',
                                   pad=0.0,
                                   extend='both',
                                   )
        ax.axhline(y=0.0, ls='--')
        ax.axvline(x=0.0, ls='--')
        ax.tick_params(labelbottom=True)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-vlim, vlim)
        ax.set_xlim(-vlim, vlim)
        # reverse x axis
        ax.invert_xaxis()
        ax.invert_yaxis()
        # print(f'vlim = {vlim}')
    else:
        im.set_data(Hx.T)
        im.set_norm(norm)
        im.set_cmap(cmap)
        im.set_extent([-vlim, vlim, -vlim, vlim])
        if ax is None:
            ax = im.axes
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-vlim, vlim)
        ax.set_xlim(-vlim, vlim)
        # print(f'vlim = {vlim}')
    return im, cbar

def plot_marginals(ax, H, axis='x', lims=[-1,1]):
    if axis == 'x':
        # sum Hx along y axis
        Hx = np.sum(H, axis=1)
        npoints = H.shape[0]
        xaxis = np.linspace(lims[0], lims[1], npoints)
        ax.plot(xaxis, Hx, 'w-')
        ax.set_xlim(lims)
        # ax.set_ylim(0, 1)
        # ax.set_title('marginal histogram')
        # ax.set_xlabel(r'$v_x$')
        ax.set_ylabel('counts')
    else:
        # sum Hx along y axis
        Hy = np.sum(H, axis=0)
        npoints = H.shape[0]
        yaxis = np.linspace(lims[0], lims[1], npoints)
        ax.plot(Hy,yaxis, 'w-')
        ax.set_ylim(lims)
        # ax.set_xlim(0, 1)
        # ax.set_title('marginal histogram')
        # ax.set_ylabel(r'$v_y$')
        ax.set_xlabel('counts')
    return

