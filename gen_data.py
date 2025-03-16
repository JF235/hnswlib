# Generate 100 random 2d vectors and plot it

import numpy as np
import matplotlib.pyplot as plt

def generate_random_data(num_points=100, num_dimensions=2):
    """
    Generate random data points in a specified number of dimensions.

    Args:
        num_points (int): Number of data points to generate.
        num_dimensions (int): Number of dimensions for each data point.

    Returns:
        np.ndarray: Generated random data points.
    """
    np.random.seed(42)
    return np.random.rand(num_points, num_dimensions)

def plot_data(data):
    """
    Plot the generated data points.

    Args:
        data (np.ndarray): Data points to plot.
    """
    plt.scatter(data[:, 0], data[:, 1])
    plt.title('Random Data Points')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Generate random data
    random_data = generate_random_data()
    
    # Save random_data to disk
    np.save('random_data.npy', random_data)

    # Plot the generated data
    plot_data(random_data)