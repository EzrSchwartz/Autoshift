import pickle
import time
import os

def find_nearest_elevation(lat, lon, grid, search_radius=0.0001):  # search_radius is in degrees
    start_time = time.time()  # Start time measurement

    lat_rounded = lat
    lon_rounded = lon

    # Attempt direct lookup first
    elevation = grid.get((lat_rounded, lon_rounded))
    if elevation is not None:
        elapsed_time = time.time() - start_time
        return elevation, elapsed_time

    # If direct lookup fails, search neighboring cells
    min_distance = float('inf')
    closest_elevation = None

    for lat_offset in [-search_radius, 0, search_radius]:
        for lon_offset in [-search_radius, 0, search_radius]:
            neighbor_lat = round(lat_rounded + lat_offset, 5)
            neighbor_lon = round(lon_rounded + lon_offset, 5)
            if (neighbor_lat, neighbor_lon) in grid:
                distance = (lat - neighbor_lat) ** 2 + (lon - neighbor_lon) ** 2
                if distance < min_distance:
                    min_distance = distance
                    closest_elevation = grid[(neighbor_lat, neighbor_lon)]

    elapsed_time = time.time() - start_time
    return closest_elevation, elapsed_time

# Rest of the script remains the same

grid_file = 'grid.pkl'

# Ensure the grid file exists
#Lat: 41.190399
#Lon: -73.335754
if not os.path.exists(grid_file):
    raise FileNotFoundError("Grid file not found. Please ensure the grid file is in the correct path.")

# Load the grid from the pickle file
with open(grid_file, 'rb') as file:
    grid = pickle.load(file)

# User input for coordinates
input_lat = float(input("Enter latitude: "))
input_lon = float(input("Enter longitude: "))

# Find nearest elevation
elevation, time_taken = find_nearest_elevation(input_lat, input_lon, grid)
print(f"Elevation: {elevation} meters" if elevation is not None else "Elevation data not found.")
print(f"Time taken to find nearest elevation: {time_taken:.5f} seconds")
