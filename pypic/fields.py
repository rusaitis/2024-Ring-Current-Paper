import numpy as np
import sys, os
from scipy.interpolate import interp1d
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if path not in sys.path:
    sys.path.append(path)
from pypic.core import *
# import pypic.graphs import as graphs

class Fields:
    """ A class for fields """
    def __init__(self, name):
        self.name = name

def reduce_field(field, avg_cells=5):
    """ Reduce the size of a field by averaging over a block of cells """
    # Size of the block in cells for averaging
    block_size = (avg_cells, avg_cells)
    return block_reduce(field, block_size=block_size, func=np.mean)

def smooth_field(field, std=1, mode='nearest', truncate=1):
    """ Smooth a field with a gaussian filter """
    if isinstance(field, list):
        field = [ndimage.gaussian_filter(f, std, mode=mode, truncate=truncate) for f in field]
    else:
        field = ndimage.gaussian_filter(field, std, mode=mode, truncate=truncate)
    return field

def cut_field(B, selection, transpose=True, clip=True):
    """ Cut 3D array to a 2D slice """
    if transpose:
        B = np.transpose(B, (2, 1, 0)) # (Z,Y,X) to (X,Y,Z)
    
    if B.ndim == 3:
        cut_axis = selection.cut_axis_code
        cut_index = selection.center_cell[cut_axis]
        b2d = np.take(B, indices=cut_index, axis=cut_axis)
    else:
        b2d = B

    if clip:
        min_cell = selection.min_cell
        max_cell = selection.max_cell
        min_cell, max_cell = sort_selection(min_cell, max_cell)
        cut_axis = selection.cut_axis_code
        if cut_axis == 0:   # cut along x phys
            b2d = np.transpose(b2d, (1, 0)) # b2d cut is (Y, Z) phys
            return b2d[min_cell[2]:max_cell[2], # In cell coords, it's (X, Z, Y)
                      min_cell[1]:max_cell[1]]
        elif cut_axis == 1: # cut along z phys
            # b2d = np.transpose(b2d, (1, 0)) # Now it's (X, Y) phys
            return b2d[min_cell[0]:max_cell[0], # In cell coords, it's (X, Z, Y)
                      min_cell[2]:max_cell[2]]
        else: # cut along y phys
            # b2d = np.transpose(b2d, (1, 0)) # Now it's (X, Y) phys
            return b2d[min_cell[0]:max_cell[0], # In cell coords, it's (X, Z, Y)
                      min_cell[1]:max_cell[1]]
    else:
        return b2d

def mask_from_range(data, min_value, max_value):
    """ Return array indices meeting the range criteria """
    # data = np.asarray(data[:])
    return np.array((data < max_value) & (data >= min_value))
    # ind = np.where((data < max_value) & (data >= min_value))
    # return ind

def good_indices(indices, shape):
    """ Check if indices are within the shape """
    return np.all([i >= 0 and i < s for i, s in zip(indices, shape)])

def get_neighbors(index,
                  shape,
                  ):
    """ Get the indices of the neighbors of a cell """
    if len(shape) == 2:
        x, y = index
        neighbor_indices = [(x + dx, y + dy) 
                            for dx in (-1, 0, 1) 
                            for dy in (-1, 0, 1)
                            if not (dx == dy == 0) # exclude the cell itself
                            ]
        # print(neighbor_indices)
        #
        # neighbor_indices = [(x, y) for x, y in product([-1, 0, 1], repeat=2) if (x, y) != (0, 0)]
        # print(neighbor_indices)
        # exit()
    if len(shape) == 3:
        x, y, z = index
        neighbor_indices = [(x + dx, y + dy, z + dz) 
                            for dx in (-1, 0, 1) 
                            for dy in (-1, 0, 1) 
                            for dz in (-1, 0, 1) 
                            if not (dx == dy == dz == 0) # exclude the cell itself
                            ]
    neighbors = []
    for ni in neighbor_indices:
        if good_indices(ni, shape):
            neighbors.append(ni)
    return neighbors

def evolve_cell(world_map,
                shape_map,
                trace_maps,
                index,
                visited_state=1,
                neighbor_state=0.5,
                empty_state=0,
                wall_state=0,
                ):
    """ Evolve a single cell and its neighbors """

    # check if the cell is within the shape
    if not good_indices(index, np.shape(world_map)):
        return shape_map, trace_maps
    # if the cell is a wall, return
    if world_map[index] == wall_state:
        return shape_map, trace_maps

    neighbor_indices = get_neighbors(index, np.shape(shape_map))

    for ni in neighbor_indices:
        # if the neighbor is not a wall and not already in the shape
        if world_map[ni] != wall_state and shape_map[ni] == empty_state:
            shape_map[ni] = neighbor_state

            # store the index of the cell that visited the neighbor
            for i in range(len(trace_maps)):
                # trace_maps[i][ni[i]] = index[i]
                trace_maps[i][ni] = index[i]
    
    shape_map[index] = visited_state # mark the cell as visited
    return shape_map, trace_maps

def evolve_conditions(world_map,
                      shape_map,
                      neighbor_state=0.5,
                      visited_state=1,
                      start_state=-1,
                      end_state=-2,
                      ):
    """ Check if to continue evolving """
    unvisited_neighbors = np.any(shape_map == neighbor_state)
    # find indices of cells in world map that are destinations (-1)
    start_index = np.where(world_map == start_state)
    end_index = np.where(world_map == end_state)
    if len(start_index[0]) > 0:
        start_index = tuple([i[0] for i in start_index])
        start_in_shape = shape_map[start_index] in [neighbor_state, visited_state]
    else:
        start_in_shape = False
    if len(end_index[0]) > 0:
        end_index = tuple([i[0] for i in end_index])
        end_in_shape = shape_map[end_index] in [neighbor_state, visited_state]
    else:
        end_in_shape = False
    return unvisited_neighbors and start_in_shape and not end_in_shape

def evolve_surface(world_map,
                   shape_map,
                   trace_maps,
                   neighbor_state=0.5,
                   ):
    """ Evolve the surface of a shape """
    evolving = evolve_conditions(world_map, shape_map)
    if evolving:
        indices = np.where(shape_map == neighbor_state)
        for index in zip(*indices):
            shape_map, trace_maps = evolve_cell(world_map, shape_map, trace_maps, index)
    return shape_map, trace_maps, evolving

def evolve_shape(a,
                 start_index,
                 end_index=None,
                 neighbor_state=0.5,
                 visited_state=1,
                 start_state=-1,
                 end_state=-2,
                 ):
    """ Evolve a shape from a starting cell """
    world_map = np.zeros_like(a)
    world_map[a > 0] = visited_state
    world_map[a == start_state] = start_state
    world_map[a == end_state] = end_state
    world_map[start_index] = start_state
    if end_index is not None:
        world_map[end_index] = end_state
    shape_map = np.zeros_like(a)
    shape_map[start_index] = neighbor_state
    trace_maps = [np.zeros_like(a) for i in range(len(start_index))]

    evolving = True
    n_iters = 0
    while evolving:
        shape_map, trace_maps, evolving = evolve_surface(world_map, shape_map, trace_maps)
        # if n_iters % 5 == 0:
        #     print(f"Iteration: {n_iters}")
        #     plot_shape(shape_map)
        n_iters += 1
    return shape_map, trace_maps

def identify_shapes(field, min_length=15):
    shapes = []
    visit_map = np.full_like(field, False)
    visit_map[field != 0] = True

    while np.any(visit_map):
        indices = np.where(visit_map > 0)
        index = next(zip(*indices))

        # find the shape that contains the cell
        shape_map, trace_maps = evolve_shape(visit_map, index)

        # remove the shape from the visit map
        visit_map[shape_map > 0] = False

        shape_volume = np.sum(shape_map > 0)
        if shape_volume > 0:
            shape_length_x = np.max(np.sum(shape_map > 0, axis=0))
            shape_length_y = np.max(np.sum(shape_map > 0, axis=1))
            shape_length = np.sqrt(shape_length_x**2 + shape_length_y**2)
            if shape_length >= min_length:
                shapes.append(shape_map)
    return shapes

def trace_shape(trace_maps, start_index, end_index):
    """ Trace the path of a shape """
    path = []
    current_index = end_index
    path.append(current_index)
    while np.any([current_index[i] != start_index[i] for i in range(len(current_index))]):
        current_index = [trace_maps[i][tuple(current_index)] for i in range(len(current_index))]
        current_index = [int(i) for i in current_index]
        path.append(current_index)
    return path

def fill_nearest_neighbors(field, value=1, norm=False, diagonals=True):
    # Initialize the neighbors matrix with the original field
    from itertools import product
    nn = np.copy(field)
    # Define shifts: right, left, up, down, and the four diagonals
    if np.ndim(field) == 2:
        if diagonals:
            shifts = [(x, y) for x, y in product([-1, 0, 1], repeat=2) if (x, y) != (0, 0)]
        else:
            shifts = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if np.ndim(field) == 3:
        if diagonals:
            shifts = [(x, y, z) for x, y, z in product([-1, 0, 1], repeat=3) if (x, y, z) != (0, 0, 0)]
        else:
            shifts = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    # find shifts using permutations allowing for duplicates
    
    for shift in shifts:
        # Apply roll for each direction and add
        if np.ndim(field) == 2:
            nn += np.roll(field, shift=shift, axis=(0, 1))
        if np.ndim(field) == 3:
            nn += np.roll(field, shift=shift, axis=(0, 1, 2))
    if norm:
        nn[nn > 0] = 1
    return nn


def find_dipolarizations(ax,
                         selection,
                         field,
                         dbz=1,
                         distance=10,
                         width=None,
                         smooth_std=3,
                         x_lims=[-20, -10],
                         y_lims=[-10, 10],
                         step=0.1,
                         min_length=10,
                         ):
    """ Find dipolarizations in a field by looking for peaks in Bz """
    from scipy.signal import find_peaks
    if isinstance(field, str):
        field = selection.get_field(field)

    x_bins = selection.x_bins(field)
    y_bins = selection.y_bins(field)
    dips = np.full_like(field, 0)

    slice_bins = y_bins
    fn_x = interp1d(x_bins, np.arange(len(x_bins)), kind='nearest', fill_value='extrapolate')
    fn_y = interp1d(y_bins, np.arange(len(y_bins)), kind='nearest', fill_value='extrapolate')
    slice_fn = fn_y
    slice_axis = 1
    search_bins = x_bins
    cut_coords = np.arange(np.min(slice_bins), np.max(slice_bins)+step, step)

    for a in cut_coords:
        slice = np.take(field, indices=int(slice_fn(a)), axis=slice_axis)
        if smooth_std is not None:
            slice = smooth_field(slice, std=smooth_std)

        peaks, pp = find_peaks(slice,
                               height=dbz,
                               distance=distance,
                               width=width,
                               )
        if len(peaks) == 0:
            continue
        max_height = np.max(pp["peak_heights"])
        for x, y, height in zip(search_bins[peaks], [a]*len(peaks) , pp["peak_heights"]):
            if x >= np.min(x_lims) and x <= np.max(x_lims):
                dips[int(fn_x(x)), int(fn_y(y))] = 1


    nn = fill_nearest_neighbors(dips, value=1, norm=True, diagonals=True)

    shapes = identify_shapes(nn, min_length=min_length)
    return shapes


def find_reversals(ax,
                   selection,
                   field,
                   smooth_std=1,
                   x_lims=[-10,-30],
                   y_lims=None,
                   direction='x',
                   step=0.1,
                   color='#ffed78',
                   ms=3,
                   mew=1,
                   marker='.',
                   alpha=0.7,
                   zorder=0,
                   first_only=True,
                   ):
    x_bins = selection.x_bins(field)
    y_bins = selection.y_bins(field)

    if direction == 'x':
        slice_bins = y_bins
        slice_fn = interp1d(y_bins, np.arange(len(y_bins)), kind='nearest', fill_value='extrapolate')
        slice_axis = 1
        search_bins = x_bins
    else:
        slice_bins = x_bins
        slice_fn = interp1d(x_bins, np.arange(len(x_bins)), kind='nearest', fill_value='extrapolate')
        slice_axis = 0
        search_bins = y_bins
    cut_coords = np.arange(np.min(slice_bins), np.max(slice_bins)+step, step)

    for a in cut_coords:
        n_reversals = 0
        slice = np.take(field, indices=int(slice_fn(a)), axis=slice_axis)
        # smooth the slice of the field
        if smooth_std is not None:
            slice = smooth_field(slice, std=smooth_std)
        # find the reversals (sign changes)
        changes = np.diff(np.sign(slice))
        reversals = search_bins[np.where(changes != 0)]
        if reversals.size > 0:
            for b in reversals:
                if direction == 'x':
                    x,y = b, a
                else:
                    x,y = a, b
                plot_condition = True
                if x_lims is not None and (x < np.min(x_lims) or x > np.max(x_lims)):
                    plot_condition = False
                if y_lims is not None and (y < np.min(y_lims) or y > np.max(y_lims)):
                    plot_condition = False
                if first_only and n_reversals > 0:
                    plot_condition = False
                if plot_condition:
                    n_reversals += 1
                    ax.plot(x, y, ms=ms, mew=mew,
                            color=color, marker=marker, alpha=alpha, zorder=zorder)

def generate_shape(ndims, step_length=3, iters=100):
    """ Generate arbitrary shape using a random walk """
    field = np.zeros(ndims)
    # draw a random walk with 1s in the field
    x = [n//2 for n in ndims]
    field[tuple(x)] = 1
    for i in range(iters):
        dx = np.random.choice([-1, 0, 1], len(ndims))
        for i in range(step_length):
            x_temp = [x[i] + dx[i] for i in range(len(ndims))]
            if good_indices(x_temp, ndims):
                x = x_temp
                field[tuple(x)] = 1
    return field

def remove_edges(field):
    """ Remove the edges of a field """
    for i in range(len(field.shape)):
        field[tuple(slice(0, 1) if j == i else slice(None) for j in range(len(field.shape)))] = 0
        field[tuple(slice(-1, None) if j == i else slice(None) for j in range(len(field.shape)))] = 0
    return field

def mark_shape_destinations(field):
    """ Mark the start and end of a shape """
    field = remove_edges(field)
    # find cells with value 1
    indices = np.where(field == 1)
    # find the first cell that furthest from the center in each direction
    min_ind = [np.min(i) for i in indices]
    max_ind = [np.max(i) for i in indices]

    # find the coordinates for the bottom left and top right corners within the shape
    # start = [min_ind[i] for i in range(len(min_ind))]
    # end = [max_ind[i] for i in range(len(max_ind))]
    # field[tuple(start)] = -1
    # field[tuple(end)] = -2
    ones_indices = np.argwhere(field == 1)
    
    # Sort the indices to find the bottom-left and upper-right
    # Bottom-left: max row index, min column index
    # Upper-right: min row index, max column index
    bottom_left = None
    upper_right = None
    
    if ones_indices.size > 0:
        # For bottom-left, we sort by row in descending order, then by column in ascending order
        bottom_left_candidates = ones_indices[np.lexsort((ones_indices[:, 1], -ones_indices[:, 0]))]
        bottom_left = bottom_left_candidates[0] if bottom_left_candidates.size > 0 else None
        
        # For upper-right, we sort by row in ascending order, then by column in descending order
        upper_right_candidates = ones_indices[np.lexsort((ones_indices[:, 1], ones_indices[:, 0]))]
        upper_right = upper_right_candidates[0] if upper_right_candidates.size > 0 else None
    field[tuple(bottom_left)] = -1 
    field[tuple(upper_right)] = -2
    # return bottom_left, upper_right


    # field[tuple(start_ind)] = -1
    # field[tuple(end_ind)] = -2

    # field[start] = -1
    # field[end] = -2
    return field

def plot_shape(field, path=None, alpha=0.5):
    """ Plot a shape """
    field = np.copy(field)
    field[field == 0] = np.nan
    import matplotlib.colors as colors
    cmap = colors.ListedColormap(['white', 'orange', '#171717', 'gray'])
    norm = colors.BoundaryNorm([-2, -1, 0.5, 1], cmap.N)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(field, cmap=cmap, origin="lower",
              extent=[0, np.shape(field)[0], 0, np.shape(field)[1]],
              alpha=alpha)
    ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.2)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')

    if path is not None:
        # Draw path
        path = np.array(path)
        ax.plot(path[:,1]+0.5,
                path[:,0]+0.5, color='red', linewidth=2, zorder=1)

    # write bin numbers on each cell
    # for i in range(ndim):
        # for j in range(ndim):
            # ax.text(i+0.5, j+0.5, f'{j},{i}', ha='center', va='center', color='green', fontsize=10)
            # ax.text(i+0.5, j+0.5, f'{int(trace_maps[0][j,i])},{int(trace_maps[1][j,i])}',
            #         ha='center', va='center', color='red', fontsize=10)

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    # fix random seed for reproducibility
    np.random.seed(8)

    ndim = 80
    field = generate_shape((ndim, ndim), step_length=np.max([1,ndim//10]), iters=ndim*2)

    # plot field
    # plt use dark background

    field = fill_nearest_neighbors(field, norm=True, diagonals=False)
    field = remove_edges(field)
    field = mark_shape_destinations(field)

    indices = np.where(field == -1)
    start_index = next(zip(*indices)) 
    end_indices = np.where(field == -2)
    end_index = next(zip(*end_indices))
    shape_map, trace_maps = evolve_shape(field, tuple(start_index))
    path = trace_shape(trace_maps, start_index, end_index)
    # print(f"Path: {path}")

    plot_shape(field_map, path=path, alpha=1)
    plot_shape(shape_map, path=path, alpha=0.2)
