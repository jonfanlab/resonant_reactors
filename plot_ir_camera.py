"""
Use this to plot any of the ir_camera / ir_image csv files.
Simply run it with python and point it at the file.
Recommend starting with --center, and then adjusting left/top/size to get framing.
"""


import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

two_sc = ''#'2SC_'

def calculate_centroid(arr):
    """
    Calculate the centroid of a 2D NumPy array.
    
    Parameters:
    arr (numpy.ndarray): 2D NumPy array
    
    Returns:
    tuple: (x, y) coordinates of the centroid
    """
    # Get the dimensions of the array
    height, width = arr.shape
    
    # Calculate the sum of x and y coordinates
    x_sum = np.sum(np.arange(width) * np.sum(arr, axis=0))
    y_sum = np.sum(np.arange(height) * np.sum(arr, axis=1))
    
    # Calculate the total mass (sum of all values in the array)
    total_mass = np.sum(arr)
    
    # Avoid division by zero
    if total_mass == 0:
        return None
    
    # Calculate the centroid coordinates
    centroid_x = x_sum / total_mass
    centroid_y = y_sum / total_mass
    
    return (centroid_x, centroid_y)

# Create heatmap
def create_heatmap(data, annot=False, vmax=None, size=300):
    plt.figure(figsize=(8, 8))
    matplotlib.rc('font', size=20)
    cmap = matplotlib.colormaps['inferno']
    #data[data < 60] = np.nan
    cmap.set_bad('white')
    ax = sns.heatmap(data, annot=annot, cmap=cmap, cbar=False, cbar_kws={'location': 'right', 'label': 'Temp [$\degree$C]'}, vmin=22, vmax=vmax)
    ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    plt.savefig('test.png', dpi=size)
    plt.show()

# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='Path to command data file.')
    parser.add_argument('--left', type=int, default=0, help='Where horizontally to slice data.')
    parser.add_argument('--top', type=int, default=0, help='Where vertically to slice data.')
    parser.add_argument('--size', type=int, default=200, help='how many pixels to take')
    parser.add_argument('--vmax', type=int, default=None, help='temperature at top of colorbar')
    parser.add_argument('--center', action='store_true', help='Flag to avoid sending out notifications')

    args = parser.parse_args()

    # load in data
    data = np.loadtxt(args.data_path, delimiter=',')
    print(data)
    vmax = int(args.vmax) if args.vmax else np.max(data)
    #df = df.iloc[int(args.top):int(args.top) + int(args.size), int(args.left):int(args.left) + int(args.size)]
    if not args.center:
        data = data[int(args.top):int(args.top) + int(args.size), int(args.left):int(args.left) + int(args.size)]
    else:
        data_filtered = data * (data > np.max(data) * 0.7)
        x_center, y_center = calculate_centroid(data)
        print(x_center + args.left, y_center + args.top)
        data = data[
           int(y_center + args.top - args.size / 2):int(y_center + args.top + args.size / 2),
           int(x_center + args.left - args.size / 2):int(x_center + args.left + args.size / 2),
        ]
        print(np.max(data))
        print(np.percentile(data, 95))
    create_heatmap(data, vmax=vmax, size=args.size)

if __name__ == "__main__":
    main()