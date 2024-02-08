# Import the required packages
import time
import csv
import rasterio
import numpy as np

# Define a function to convert a GeoTIFF file to a CSV file
def geotiff_to_csv(geotiff_path, csv_path):
    # Open the GeoTIFF file as a dataset
    with rasterio.open(geotiff_path) as dataset:
        # Read the dataset's default band (1)
        band1 = dataset.read(1)

        # Get the geographic information
        transform = dataset.transform

        # Prepare CSV file
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Using numpy to iterate over the raster
            for (row, col), value in np.ndenumerate(band1):
                if np.isfinite(value):  # Check for valid elevation data
                    x, y = rasterio.transform.xy(transform, row, col, offset='center')
                    writer.writerow([round(y, 5), round(x, 5), value])
                print(row)

# Usage
# Set the paths of the GeoTIFF file and the CSV file
geotiff_path = r"C:\Users\ezran\Downloads\USGS_13_n42w074_20230307.tif"
csv_path = r"C:\Users\ezran\PycharmProjects\AutoshiftAI__V1.2\elevation_Data.csv"

# Measure the execution time
start_time = time.time()
print(start_time)
# Call the function
geotiff_to_csv(geotiff_path, csv_path)

# Print the execution time
end_time = time.time()
total_time = end_time - start_time
print(total_time)
