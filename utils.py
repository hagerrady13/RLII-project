import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def print_cost(value_function, episode, ax):
    """
    A utility function for plotting the cost-to-go
    Args:
        value_function:
        episode:
        ax:
    Returns:
    """
    grid_size = 40
    POSITION_MIN = -1.2
    POSITION_MAX = 0.5
    VELOCITY_MIN = -0.07
    VELOCITY_MAX = 0.07
    positions = np.linspace(POSITION_MIN, POSITION_MAX, grid_size)

    velocities = np.linspace(VELOCITY_MIN, VELOCITY_MAX, grid_size)
    axis_x = []
    axis_y = []
    axis_z = []
    for position in positions:
        for velocity in velocities:
            axis_x.append(position)
            axis_y.append(velocity)
            axis_z.append(value_function.cost_to_go(position, velocity))

    print("episode: ", np.mean(axis_z), np.std(axis_z))
    ax.scatter(axis_x, axis_y, axis_z)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('cost-to-go')
    ax.set_title('Episode %d' % (episode))

