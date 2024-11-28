import numpy as np
import matplotlib.pyplot as plt
import trimesh
import matplotlib.path as mplPath
import numpy.ma as ma
from shapely.affinity import translate


def generate_grid(res, slice_z, path_to_file):
    mesh = trimesh.load(path_to_file)

    plane_origin = [0, 0, slice_z]
    plane_normal = [0, 0, 1]

    slice = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)

    if slice is not None:
        slice_2D, _ = slice.to_planar()
        polygons = slice_2D.polygons_full
    else:
        print(f"No intersection found at z = {slice_z}")
        polygons = []

    x_grid = np.linspace(-35, 35, res)
    y_grid = np.linspace(-35, 35, res)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    grid_points_2d = np.vstack((X_grid.ravel(), Y_grid.ravel())).T
    inside_mask = np.zeros(grid_points_2d.shape[0], dtype=bool)
    translation_vector = (3.1, -1.2)

    for polygon in polygons:
        translated_polygon = translate(polygon, xoff=translation_vector[0], yoff=translation_vector[1])
        x_poly, y_poly = translated_polygon.exterior.xy
        poly_path = mplPath.Path(np.vstack((x_poly, y_poly)).T)
        inside = poly_path.contains_points(grid_points_2d)
        inside_mask = inside_mask | inside

    inside_mask = inside_mask.reshape(X_grid.shape)
    occupancy_grid = np.zeros(X_grid.shape, dtype=float)
    occupancy_grid[inside_mask] = 1


    return occupancy_grid


def visualize_grid():
    slice = 5

    occupancy_grid = generate_grid(140, slice)

    # Mask the -1 values for visualization
    occupancy_grid_ma = ma.masked_equal(occupancy_grid, 1)

    # Create a colormap where masked values will appear in black
    cmap = plt.cm.gray
    cmap.set_bad(color='black')

    # Plot the occupancy grid
    plt.figure(figsize=(10, 8))
    plt.imshow(occupancy_grid_ma, origin='lower', extent=(0, len(occupancy_grid), 0, len(occupancy_grid[0])), cmap=cmap)
    plt.colorbar(label='Occupancy Value')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Occupancy Grid at z = {slice}')
    plt.show()


#visualize_grid()