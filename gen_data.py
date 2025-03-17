# Generate 100 random 2d vectors and plot it

import numpy as np
import matplotlib.pyplot as plt
import argparse

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

def gen_table(random_data):
    # Generate the table with pre-computed L2 distances
    num_points = random_data.shape[0]
    table = np.zeros((num_points, num_points))
    # Compute pairwise distances using broadcasting
    table = np.sum((random_data[:, np.newaxis, :] - random_data[np.newaxis, :, :])**2, axis=2)
    return table

if __name__ == "__main__":

    # if argument of -g is randomdata generate
    # else if argument of -g is table  
    # Generate random data

    parser = argparse.ArgumentParser(description='Generate random data or table.')
    parser.add_argument('-g', '--generate', type=str, choices=['randomdata', 'table'], required=True,
                        help='Generate random data or table.')
    args = parser.parse_args()
    if args.generate == 'table':
        
        random_data = generate_random_data()
        table = gen_table(random_data)
        # Save table to disk
        np.save('table.npy', table)

        print(table)
    else:

        random_data = generate_random_data()
        
        # Save random_data to disk
        np.save('random_data.npy', random_data)

        # Plot the generated data
        plot_data(random_data)