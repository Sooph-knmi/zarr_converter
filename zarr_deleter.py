import os
from ecml_tools.data import open_dataset
import numpy as np
# Paths
# nc_files_path = "/home/sophiebuurman/dowa-bucket/dowa_4years/dowa2015"
# zarr_path = "/home/sophiebuurman/data1/data2/zarr_converter/output/dowa2008.zarr"
nc_files_path = "/ec/res4/scratch/ecme5801/dowa2013"
zarr_path = "/ec/res4/scratch/ecme5801/DOWA_zarr/dowa2013.zarr"

# Ensure the Zarr dates folder exists
if not os.path.isdir(zarr_path):
    print(f"Zarr dates folder does not exist: {zarr_path}")
    exit(1)
ds = open_dataset(zarr_path)
# Iterate through each .nc file in the specified directory
for nc_filename in os.listdir(nc_files_path):
    if nc_filename.endswith(".nc"):
        # Extract the date part from the .nc filename (YYYYMMDD before the .nc extension)
        nc_date = nc_filename.split('.')[-2]  # The date is before the last dot and the .nc extension
        date = np.datetime64(nc_date[:4] + "-" + nc_date[4:6] + "-" + nc_date[6:8])
        # Check if a corresponding date file exists in the Zarr archive
        print(date)
        if date in ds.dates:
            # Remove the .nc file if the corresponding date exists
            nc_file_path = os.path.join(nc_files_path, nc_filename)
            print(f"Removing {nc_file_path}")
            os.remove(nc_file_path)
            
        else:
            print(f"Date not found in Zarr archive: {nc_date}. Skipping {nc_filename}")
