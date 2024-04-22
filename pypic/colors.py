import colorcet as cc
# import cmasher as cmr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

# import matplotlib as mpl
accent = 'orange'
accent1 = 'orange'
accent2 = 'red'
accent3 = '#FF9A00'
text = 'white'
grid = 'grey'
bg = '#171717'
# bg = '#2f2c2c'
bg = '#2c2c2c'
elements = '#292929'
# elements = '#363636'
# elements2 = '#121212'
elements2 = '#2e2e2e'
elements3 = '#6AE0BA'
elements4 = '#ff5c98'
colors = [accent, elements3, elements4, 'white']
colors_light = [adjust_lightness(accent, 1.2),
                adjust_lightness(elements3, 1.2),
                adjust_lightness(elements4, 1.2),
                adjust_lightness('white', 1.2)]

# colors_dark = [adjust_lightness(accent, 0.8),
#                adjust_lightness(elements3, 0.8),
#                adjust_lightness(elements4, 0.8),
#                adjust_lightness('white', 0.8)]
colors_dark = ['#f56505', '#0473bd', '#b802af', 'black']

cmap_p = cc.cm["blues"].copy()
# cmap_p = cc.cm["blues"].copy().reversed()
# cmap_p = cc.cm["kbc"].copy()
cmap_p.set_bad(cmap_p.get_under())  # set the color for 0

# cmap_vdf = cc.cm["rainbow"].copy()
# cmap_vdf = mpl.colormaps["rainbow"].copy()
# cmap_vdf = plt.cm.rainbow.copy()
cmap_vdf = plt.cm.turbo.copy()
# cmap_vdf.set_bad(cmap_vdf.get_under())  # set the color for 0
cmap_vdf.set_bad('black')
cmap_vdf.set_under('black')
cmap_vdf.set_over(adjust_lightness(cmap_vdf(1.0), 0.7))
cmap_vdf.set_under(adjust_lightness(cmap_vdf(0.0), 0.2))


import numpy as np
from matplotlib.colors import LinearSegmentedColormap
ncolors = 256
mid_n = 30
max_alpha = 0.9
max_alpha = 1
min_alpha = 0.5

# cmap = mpl.colormaps['twilight_shifted']
# cmap = mpl.colormaps['rainbow']
# cmap = mpl.colormaps['Spectral'].reversed()

# color_array = plt.get_cmap('turbo')(range(ncolors))
color_array = plt.get_cmap('rainbow')(range(ncolors))
# color_array2 = plt.get_cmap('binary')(range(ncolors))
# color_array2 = plt.get_cmap('binary')(np.ones(ncolors))
# color_array = bkr_extra_cmap(range(ncolors))
# cmap_bg = bkr_extra_cmap # divergent (lightblue-blue-black-red-orange)
# alphas = np.linspace(0,1.0,ncolors)
# alphas = np.linspace(0,max_alpha,ncolors)
alphas0 = np.linspace(0,max_alpha,ncolors-mid_n)
alphas_mids = np.zeros(mid_n)
alphas = np.concatenate((alphas_mids, alphas0))
# alphas0 = np.linspace(max_alpha,0,ncolors//2- mid_n//2) #128
# alphas1 = np.linspace(0,max_alpha,ncolors//2- mid_n//2) #128
# alphas_mids = np.zeros(mid_n)
# alphas = np.concatenate((alphas0, alphas_mids, alphas1))
# alphas = np.concatenate((alphas0, alphas1))
# color_array[:,-1] = alphas
mids = 4
offcenter = 7
color_array[124-mids:124+mids+offcenter,-1] = 0.0
# color_array[124-12:124+16,-1] = 0
color_array[124-mids:124+mids+offcenter, 0] = 1
color_array[124-mids:124+mids+offcenter, 1] = 1
color_array[124-mids:124+mids+offcenter, 2] = 1
# map_object = LinearSegmentedColormap.from_list(name='turbo_white',colors=color_array)
map_object = LinearSegmentedColormap.from_list(name='rainbow_white',colors=color_array)
plt.colormaps.register(cmap=map_object)

def cmap_from_color(color=None,
                    cmap:LinearSegmentedColormap=None,
                    alpha:float=None,
                    alpha_range:tuple=None,
                    lightness_range:tuple=None,
                    ncolors:int=256,
                    name:str='custom_cmap',
                    register:bool=False,
                    ):
    """ Create a custom colormap from a list of colors """
    if color is None:
        color = 'black'
    if cmap is not None:
        color_array = cmap(range(ncolors))  
    else:
        color_rgba = mpl.colors.to_rgba(color)
        color_array = np.full((ncolors, 4), color_rgba)

    if alpha_range is not None:
        alphas = np.linspace(alpha_range[0], alpha_range[1], ncolors)
        color_array[:,-1] = alphas

    if alpha is not None:
        color_array[:,-1] = np.full(ncolors, alpha)

    if lightness_range is not None:
        for i, c in enumerate(color_array):
            lightness = lightness_range[0] + (lightness_range[1] - lightness_range[0]) * i / ncolors
            color_array[i][:3] = adjust_lightness(c[:3], lightness)

    map_object = LinearSegmentedColormap.from_list(name=name,colors=color_array)
    if register:
        plt.colormaps.register(cmap=map_object)
    return map_object

turbo_cmap = LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#240C2E-376CF8-26E1A8-94FF2F-F6B02C-DC300D-660006
    (0.000, (0.141, 0.047, 0.180)),
    (0.167, (0.216, 0.424, 0.973)),
    (0.333, (0.149, 0.882, 0.659)),
    (0.500, (0.580, 1.000, 0.184)),
    (0.667, (0.965, 0.690, 0.173)),
    (0.833, (0.863, 0.188, 0.051)),
    (1.000, (0.400, 0.000, 0.024))))

# original
Ray_div_cmap = LinearSegmentedColormap.from_list('Ray_div_cmap', (
    # Edit this gradient at https://eltos.github.io/gradient/#000000-0000BB-0543FF-31B6FE-A1FFFF-FFFFFF-63FF03-FCFF0A-FEB60A-FC5108-FB0006
    (0.000, (0.000, 0.000, 0.000)),
    (0.100, (0.000, 0.000, 0.733)),
    (0.200, (0.020, 0.263, 1.000)),
    (0.300, (0.192, 0.714, 0.996)),
    (0.400, (0.631, 1.000, 1.000)),
    (0.500, (1.000, 1.000, 1.000)),
    (0.600, (0.388, 1.000, 0.012)),
    (0.700, (0.988, 1.000, 0.039)),
    (0.800, (0.996, 0.714, 0.039)),
    (0.900, (0.988, 0.318, 0.031)),
    (1.000, (0.984, 0.000, 0.024))))

Ray_div_cmap = LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:000000-15:0000BB-25:0543FF-35:31B6FE-40:A1FFFF-45:D0FFFF-50:FFFFFF-55:AEFF7C-60:63FF03-65:FCFF0A-75:FEB60A-85:FC5108-100:FB0006
    (0.000, (0.000, 0.000, 0.000)),
    (0.150, (0.000, 0.000, 0.733)),
    (0.250, (0.020, 0.263, 1.000)),
    (0.350, (0.192, 0.714, 0.996)),
    (0.400, (0.631, 1.000, 1.000)),
    (0.450, (0.816, 1.000, 1.000)),
    (0.500, (1.000, 1.000, 1.000)),
    (0.550, (0.682, 1.000, 0.486)),
    (0.600, (0.388, 1.000, 0.012)),
    (0.650, (0.988, 1.000, 0.039)),
    (0.750, (0.996, 0.714, 0.039)),
    (0.850, (0.988, 0.318, 0.031)),
    (1.000, (0.984, 0.000, 0.024))))

Ray_div_cmap = LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:000000-5:1F0026-10:360043-15:32025C-20:300089-25:0000BB-30:0543FF-35:31B6FE-40:A1FFFF-45:D0FFFF-50:FFFFFF-55:AEFF7C-60:63FF03-65:FCFF0A-70:FEB60A-75:FC5108-80:FB0006-85:B80004-90:8C0003-95:6C0002-100:550001
    (0.000, (0.000, 0.000, 0.000)),
    (0.050, (0.122, 0.000, 0.149)),
    (0.100, (0.212, 0.000, 0.263)),
    (0.150, (0.196, 0.008, 0.361)),
    (0.200, (0.188, 0.000, 0.537)),
    (0.250, (0.000, 0.000, 0.733)),
    (0.300, (0.020, 0.263, 1.000)),
    (0.350, (0.192, 0.714, 0.996)),
    (0.400, (0.631, 1.000, 1.000)),
    (0.450, (0.816, 1.000, 1.000)),
    (0.500, (1.000, 1.000, 1.000)),
    (0.550, (0.682, 1.000, 0.486)),
    (0.600, (0.388, 1.000, 0.012)),
    (0.650, (0.988, 1.000, 0.039)),
    (0.700, (0.996, 0.714, 0.039)),
    (0.750, (0.988, 0.318, 0.031)),
    (0.800, (0.984, 0.000, 0.024)),
    (0.850, (0.722, 0.000, 0.016)),
    (0.900, (0.549, 0.000, 0.012)),
    (0.950, (0.424, 0.000, 0.008)),
    (1.000, (0.333, 0.000, 0.004))))

Ray_cmap = LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#0000FF-0000FF-0344FF-0D7CFE-16B6FE-21F6FE-23FF71-2BFF06-CAFF0B-FEAD09-FB4409-FB0006
    (0.000, (0.000, 0.000, 1.000)),
    (0.091, (0.000, 0.000, 1.000)),
    (0.182, (0.012, 0.267, 1.000)),
    (0.273, (0.051, 0.486, 0.996)),
    (0.364, (0.086, 0.714, 0.996)),
    (0.455, (0.129, 0.965, 0.996)),
    (0.545, (0.137, 1.000, 0.443)),
    (0.636, (0.169, 1.000, 0.024)),
    (0.727, (0.792, 1.000, 0.043)),
    (0.818, (0.996, 0.678, 0.035)),
    (0.909, (0.984, 0.267, 0.035)),
    (1.000, (0.984, 0.000, 0.024))))

Mea_div_cmap = LinearSegmentedColormap.from_list('Mea_div_cmap', (
# Edit this gradient at https://eltos.github.io/gradient/#F113FF-1313FF-13FFFF-FFF7F7-27FF13-F9FF13-FF1319
    (0.000, (0.945, 0.075, 1.000)),
    (0.167, (0.075, 0.075, 1.000)),
    (0.333, (0.075, 1.000, 1.000)),
    (0.500, (1.000, 0.969, 0.969)),
    (0.667, (0.153, 1.000, 0.075)),
    (0.833, (0.976, 1.000, 0.075)),
    (1.000, (1.000, 0.075, 0.098))))
# Mea_div_black_cmap = LinearSegmentedColormap.from_list('Mea_div_black_cmap', (

# Mea_div_black_cmap = LinearSegmentedColormap.from_list('my_gradient', (
Mea_div_black_cmap = LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#A700E8-4469FF-30A19E-1E1734-10A22A-E2E800-E8000F
    (0.000, (0.655, 0.000, 0.910)),
    (0.167, (0.267, 0.412, 1.000)),
    (0.333, (0.188, 0.631, 0.620)),
    (0.500, (0.118, 0.090, 0.204)),
    (0.667, (0.063, 0.635, 0.165)),
    (0.833, (0.886, 0.910, 0.000)),
    (1.000, (0.910, 0.000, 0.059))))
Mea_div_black_cmap = LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#BF00E8-4D70FF-24AB9D-181322-57A210-E2E800-E80016
    (0.000, (0.749, 0.000, 0.910)),
    (0.167, (0.302, 0.439, 1.000)),
    (0.333, (0.141, 0.671, 0.616)),
    (0.500, (0.094, 0.075, 0.133)),
    (0.667, (0.341, 0.635, 0.063)),
    (0.833, (0.886, 0.910, 0.000)),
    (1.000, (0.910, 0.000, 0.086))))

Mea_cmap = LinearSegmentedColormap.from_list('Mea_cmap', (
# Edit this gradient at https://eltos.github.io/gradient/#FFFFFF-0B0BFF-0BFFFF-1FFF0B-F8FF0B-FF0B11
    (0.000, (1.000, 1.000, 1.000)),
    (0.200, (0.043, 0.043, 1.000)),
    (0.400, (0.043, 1.000, 1.000)),
    (0.600, (0.122, 1.000, 0.043)),
    (0.800, (0.973, 1.000, 0.043)),
    (1.000, (1.000, 0.043, 0.067))))
bkr_extra_cmap = LinearSegmentedColormap.from_list('bkr_extra_cmap', (
# Edit this gradient at https://eltos.github.io/gradient/#00BEEF-1967F3-1A1719-DB3832-DB9032
    (0.000, (0.369, 0.667, 1.000)),
    (0.250, (0.098, 0.404, 0.953)),
    (0.500, (0.102, 0.090, 0.098)),
    (0.750, (0.859, 0.220, 0.196)),
    (1.000, (0.859, 0.565, 0.196))))
# bkr_extra_cmap = LinearSegmentedColormap.from_list('my_gradient', (
#     # Edit this gradient at https://eltos.github.io/gradient/#00BEEF-0098EF-1967F3-671E81-1A1719-840E4D-DB3832-DC6B14-DB9032
#     (0.000, (0.000, 0.745, 0.937)),
#     (0.125, (0.000, 0.596, 0.937)),
#     (0.250, (0.098, 0.404, 0.953)),
#     (0.375, (0.404, 0.118, 0.506)),
#     (0.500, (0.102, 0.090, 0.098)),
#     (0.625, (0.518, 0.055, 0.302)),
#     (0.750, (0.859, 0.220, 0.196)),
#     (0.875, (0.863, 0.420, 0.078)),
#     (1.000, (0.859, 0.565, 0.196))))

bkr_extra_cmap = LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:00BEEF-25:1967F3-45:1A356B-50:1A1719-55:662423-75:DB3832-100:DB9032
    (0.000, (0.000, 0.745, 0.937)),
    (0.250, (0.098, 0.404, 0.953)),
    (0.450, (0.102, 0.208, 0.420)),
    (0.500, (0.102, 0.090, 0.098)),
    (0.550, (0.400, 0.141, 0.137)),
    (0.750, (0.859, 0.220, 0.196)),
    (1.000, (0.859, 0.565, 0.196))))

turbo_extra_cmap = LinearSegmentedColormap.from_list('turbo_extra_cmap', (
# Edit this gradient at https://eltos.github.io/gradient/#0:211221-6.3:68098B-12.5:9600D0-18.1:7F48F5-25:1967F3-31.3:00A0DE-37.5:21C3CF-43.8:00E6BB-50:44FD6D-56.3:72FE54-62.5:94FF2F-68.8:D3DF00-75:EDBF2D-81.3:DC7A00-87.5:C31C0A-93.8:95180A-100:68140B
    (0.000, (0.129, 0.071, 0.129)),
    (0.063, (0.408, 0.035, 0.545)),
    (0.125, (0.588, 0.000, 0.816)),
    (0.181, (0.498, 0.282, 0.961)),
    (0.250, (0.098, 0.404, 0.953)),
    (0.313, (0.000, 0.627, 0.871)),
    (0.375, (0.129, 0.765, 0.812)),
    (0.438, (0.000, 0.902, 0.733)),
    (0.500, (0.267, 0.992, 0.427)),
    (0.563, (0.447, 0.996, 0.329)),
    (0.625, (0.580, 1.000, 0.184)),
    (0.688, (0.827, 0.875, 0.000)),
    (0.750, (0.929, 0.749, 0.176)),
    (0.813, (0.863, 0.478, 0.000)),
    (0.875, (0.765, 0.110, 0.039)),
    (0.938, (0.584, 0.094, 0.039)),
    (1.000, (0.408, 0.078, 0.043))))
