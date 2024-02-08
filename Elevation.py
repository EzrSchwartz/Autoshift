import csv
import math
import pickle
import time
import os

def create_grid(data, precision=1e-5):  # Precision set to 5th decimal place
    grid = {}
    for lat, lon, elev in data:
        # Rounding the coordinates to match the input precision
        lat_rounded = round(lat, 5)
        lon_rounded = round(lon, 5)
        grid[(lat_rounded, lon_rounded)] = elev
    return grid


def find_nearest_elevation(lat, lon, grid):
    start_time = time.time()
    # Start time measurement

    lat_rounded = round(lat, 5)
    lon_rounded = round(lon, 5)

    # Direct lookup
    elevation = grid.get((lat_rounded, lon_rounded), None)

    elapsed_time = time.time() - start_time  # Time measurement ends here
    return elevation, elapsed_time

def load_elevation_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            lat, lon, elev = map(float, row)
            data.append((lat, lon, elev))
    return data

grid_file = 'grid.pkl'

# Check if the grid file exists
if os.path.exists(grid_file):
    # Load the grid from the file
    with open(grid_file, 'rb') as file:
        grid = pickle.load(file)
else:
    # Load data and create grid
    elevation_data = load_elevation_data('extracted_data.csv')
    grid = create_grid(elevation_data)
    # Save the grid to a file
    with open(grid_file, 'wb') as file:
        pickle.dump(grid, file)

# Find nearest elevation
elevation, time_taken = find_nearest_elevation(42.00051,-72.98514, grid)
print(f"Elevation: {elevation} meters" if elevation is not None else "Elevation data not found.")
print(f"Time taken to find nearest elevation: {time_taken:.5f} seconds")
