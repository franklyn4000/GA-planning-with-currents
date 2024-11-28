import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from matplotlib.colors import Normalize
import os


def visualize(index, traveled_path, load_path=False):
    data_folder = 'data_out'
    index_folder = f'path{index}'

    file_path = os.path.join(data_folder, index_folder)
    path = None
    fit_evo = None

    if load_path:
        path = np.loadtxt(os.path.join(file_path, 'path.csv'), delimiter=',')
        fit_evo = np.loadtxt(os.path.join(file_path, 'fit_evo.csv'), delimiter=',')

    res_final = np.loadtxt(os.path.join(file_path, 'res.csv'), delimiter=',')
    grid_final = np.loadtxt(os.path.join(file_path, 'grid.csv'), delimiter=',')
    x_grid_final = np.loadtxt(os.path.join(file_path, 'x_grid.csv'), delimiter=',')
    y_grid_final = np.loadtxt(os.path.join(file_path, 'y_grid.csv'), delimiter=',')
    u_grid_final = np.loadtxt(os.path.join(file_path, 'u_grid.csv'), delimiter=',')
    v_grid_final = np.loadtxt(os.path.join(file_path, 'v_grid.csv'), delimiter=',')
    mag_grid_final = np.loadtxt(os.path.join(file_path, 'mag_grid.csv'), delimiter=',')
    start_final = np.loadtxt(os.path.join(file_path, 'start.csv'), delimiter=',')
    goal_final = np.loadtxt(os.path.join(file_path, 'goal.csv'), delimiter=',')
    goal_uncertainty = np.loadtxt(os.path.join(file_path, 'uncertainty.csv'), delimiter=',')

    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap('jet')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if fit_evo is not None:
        plt.plot(fit_evo, marker='o', linestyle='-', color='b')
        plt.title('Fitness Evolution')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.show()

    fig, ax = plt.subplots(figsize=(14, 14))

    quiver = ax.quiver(x_grid_final, y_grid_final, u_grid_final, v_grid_final,
                       mag_grid_final, cmap=cmap, norm=norm, scale=50)

    occupancy_grid = grid_final
    occupancy_grid_ma = ma.masked_equal(occupancy_grid, -1)
    cmap = plt.cm.gray

    img = ax.imshow(occupancy_grid_ma, origin='lower', extent=(0, res_final, 0, res_final), cmap=cmap)
    ax.set_xlabel('X', fontsize=24)
    ax.set_ylabel('Y', fontsize=24)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)


    if path is not None:
        ax.set_title(f'Path Fitness = {fit_evo[-1]:.3f}', fontsize=42)
        x_coords = [pos[0] for pos in path]
        y_coords = [pos[1] for pos in path]

        ax.plot(x_coords, y_coords, color='red', linewidth=12, label='Future Path')

        start_x = start_final[0]
        start_y = start_final[1]
        goal_x = goal_final[0]
        goal_y = goal_final[1]

        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        width = bbox.width * fig.dpi

        cell_size = width / len(grid_final[0])  # assuming square grid cells
        print(goal_uncertainty)
        marker_size = cell_size * goal_uncertainty

        ax.plot(start_x, start_y, marker='o', color='green', markersize=20, label='Start')
        ax.plot(goal_x, goal_y, marker='o', color=(0.9, 0.1, 0.1, 0.6), markersize=marker_size, label='Goal')

    else:
        print("No path found")

    print("AAAAAAAA", traveled_path)
    if traveled_path is not None:
        x_coords = [pos[0] for pos in traveled_path]
        y_coords = [pos[1] for pos in traveled_path]
        ax.plot(x_coords, y_coords, color='white', linewidth=8, label='Travelled Path')

    if path is not None and traveled_path is not None:
        legend = ax.legend(handles=[
            plt.Line2D([], [], color='red', lw=8, label='Planned Path'),
            plt.Line2D([], [], color='white', lw=4, label='Traveled Path'),
            plt.Line2D([], [], color='green', marker='o', linestyle='None', markersize=15, label='ROV Position'),
            plt.Line2D([], [], color=(0.9, 0.1, 0.1, 0.6), marker='o', linestyle='None', markersize=15, label='Target Area')
        ], loc='lower right', fontsize=23)
        legend.get_frame().set_facecolor('lightgrey')

    plt.tight_layout()
    plt.show()
