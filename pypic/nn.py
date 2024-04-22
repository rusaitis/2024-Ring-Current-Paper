import numpy as np

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
        if n_iters % 40 == 0:
            print(f"Iteration: {n_iters}")
            # plot_shape(shape_map) # plot for gradual development
        n_iters += 1
    return shape_map, trace_maps

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
    # Fill in neighboring cells to enlarge the shape
    from itertools import product
    nn = np.copy(field)
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
    
    for shift in shifts:
        if np.ndim(field) == 2:
            nn += np.roll(field, shift=shift, axis=(0, 1))
        if np.ndim(field) == 3:
            nn += np.roll(field, shift=shift, axis=(0, 1, 2))
    if norm:
        nn[nn > 0] = 1
    return nn

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
    """ Mark the start and end of a shape (top and bottom) """
    field = remove_edges(field)
    # find cells with value 1
    indices = np.where(field == 1)
    # find the first cell that furthest from the center in each direction
    min_ind = [np.min(i) for i in indices]
    max_ind = [np.max(i) for i in indices]
    ones_indices = np.argwhere(field == 1)
    # Sort the indices to find the bottom-left and upper-right
    bottom_left = None
    upper_right = None
    if ones_indices.size > 0:
        # For bottom-left, sort by row in descending order, then by column in ascending order
        bottom_left_candidates = ones_indices[np.lexsort((ones_indices[:, 1], -ones_indices[:, 0]))]
        bottom_left = bottom_left_candidates[0] if bottom_left_candidates.size > 0 else None
        # For upper-right, sort by row in ascending order, then by column in descending order
        upper_right_candidates = ones_indices[np.lexsort((ones_indices[:, 1], ones_indices[:, 0]))]
        upper_right = upper_right_candidates[0] if upper_right_candidates.size > 0 else None
    field[tuple(bottom_left)] = -1 
    field[tuple(upper_right)] = -2
    return field

def plot_shape(field, path=None, path2=None, alpha=0.5, text=None):
    """ Plot a shape """
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    field = np.copy(field)
    field[field == 0] = np.nan
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
        path = np.array(path)
        ax.plot(path[:,1]+0.5,
                path[:,0]+0.5, color='red', linewidth=2.8, zorder=0, alpha=0.8, label='Cellular')

    if path2 is not None:
        path2 = np.array(path2)
        ax.plot(path2[:,1]+0.5,
                path2[:,0]+0.5, color='yellow', linewidth=2, zorder=1, alpha=0.8, label='Mai-Dijstra')

    if text is not None:
        ax.text(0.1, 0.1, text,
                transform=ax.transAxes,
                fontsize=12, ha='center', va='center', color='white',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='black')
                )
    ax.legend()
    fig.tight_layout()
    plt.show()

def Mai_Dijkstra(field, start_index, end_index):
    grid = field
    origin = start_index
    destination = end_index
    grid_l, grid_w = grid.shape

    distance = np.zeros(grid.shape)
    distance[:] = np.inf

    visited = np.zeros(grid.shape)

    # x, y = random.choice(np.argwhere(grid == 1))
    distance[origin] = 0
    indx_map = np.zeros((grid_l,grid_w,2))
    indx_map[:,:,:] = (-1,-1)

    finished = False

    x,y = origin
    # indx_map[x,y,:] = origin

    while finished == False:
        if distance[x,y] == np.inf:
            finished = True
        # if [x,y] == destination:
        #     print ("distance:", distance[x, y])
        #     finished = True
        #     break
        if x < (grid_l - 1):
            if grid[x + 1, y] == 1:
                distance[x + 1, y] = np.min([distance[x, y] + 1, distance[x + 1, y]])
                if np.argmin([distance[x, y] + 1, distance[x + 1, y]]) == 0:
                    indx_map[x + 1,y,:] = (x,y)
            if y < (grid_w - 1):
                if grid[x + 1, y + 1] == 1:
                    distance[x + 1, y + 1] = np.min([distance[x,y] + 1, distance[x+1, y+ 1]])
                    if np.argmin([distance[x, y] + 1, distance[x + 1, y + 1]]) == 0:
                        indx_map[x+1,y+1,:] = (x,y)
            if y > 0:
                if grid[x + 1, y - 1] == 1:
                    distance[x + 1, y - 1] = np.min([distance[x,y] + 1, distance[x+1, y- 1]])
                    if np.argmin([distance[x, y] + 1, distance[x + 1, y - 1]]) == 0:
                        indx_map[x + 1,y - 1,:] = (x,y)
        if x > 0:
            if grid[x - 1, y] == 1:
                distance[x - 1, y] = np.min([distance[x, y] + 1, distance[x - 1, y]])
                if np.argmin([distance[x, y] + 1, distance[x - 1, y]]) == 0:
                    indx_map[x - 1,y,:] = (x,y)
            if y < (grid_w - 1):
                if grid[x - 1, y + 1] == 1:
                    distance[x - 1, y + 1] = np.min([distance[x,y] + 1, distance[x-1, y+ 1]])
                    if np.argmin([distance[x, y] + 1, distance[x - 1, y+1]]) == 0:
                        indx_map[x - 1,y + 1,:] = (x,y)
            if y > 0:
                if grid[x - 1, y - 1] == 1:
                    distance[x - 1, y - 1] = np.min([distance[x,y] + 1, distance[x-1, y-1]])
                    if np.argmin([distance[x, y] + 1, distance[x - 1, y - 1]]) == 0:
                        indx_map[x - 1,y - 1,:] = (x,y)
                
        if y < (grid_w - 1):
            if grid[x, y + 1] == 1:
                distance[x, y + 1] = np.min([distance[x, y] + 1, distance[x, y + 1]])  
                if np.argmin([distance[x, y] + 1, distance[x, y + 1]]) == 0:
                    indx_map[x,y + 1,:] = (x,y)
                
        if  y > 0:
            if grid[x, y - 1] == 1:
                distance[x, y - 1] = np.min([distance[x, y] + 1, distance[x, y - 1]])
                if np.argmin([distance[x, y] + 1, distance[x, y - 1]]) == 0:
                    indx_map[x,y - 1,:] = (x,y)
        
        visited[x,y] = 1
        if distance[visited == 0].size != 0 :
            ind = np.argmin(distance[visited == 0])
            x,y = np.argwhere(visited == 0)[ind]
        else:
            finished = True

    path_exist = False

    if distance[destination] == np.inf:
        print ("no path")
    else:
        print ("path length:", distance[destination])
        path_exist = True

    # Trace path:
    path_end = False

    path = []

    path.append(destination)
    next_ind = destination

    if path_exist == True:
        while path_end == False:
            x,y = next_ind
            x,y = int(x),int(y)
            next_ind = indx_map[x,y,:]
            path.append(tuple(next_ind))
            # path.append(next_ind)
            if tuple(next_ind) == origin:
                path_end = True     
        # print (path)
    else:
        print ("No path")
    return path


if __name__ == "__main__":
    # fix random seed for reproducibility
    np.random.seed(13)

    ndim = 50
    field = generate_shape((ndim, ndim), step_length=np.max([1,ndim//10]), iters=ndim*2)

    field = fill_nearest_neighbors(field, norm=True, diagonals=False)
    # field = np.ones_like(field)
    field = remove_edges(field)
    field = mark_shape_destinations(field)

    indices = np.where(field == -1)
    start_index = next(zip(*indices)) 
    end_indices = np.where(field == -2)
    end_index = next(zip(*end_indices))
    shape_map, trace_maps = evolve_shape(field, tuple(start_index))
    path = trace_shape(trace_maps, start_index, end_index)

    Mai_field = np.copy(field)
    Mai_field[Mai_field != 0] = 1
    path_Mai = Mai_Dijkstra(Mai_field, start_index, end_index)
    length_path = len(path)
    length_path_Mai = len(path_Mai)
    text = f"Cellular: {length_path} cells\nMai-Dijkstra: {length_path_Mai} cells"

    plot_shape(field, path=path, path2=path_Mai, alpha=1, text=text)
    # plot path_Mai
    # plot_shape(shape_map, path=path, alpha=0.2)
