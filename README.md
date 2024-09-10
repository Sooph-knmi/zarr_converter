## netcdf-> zarr converter for meps data stored on ppi.
See zconvert.sh for an example of how to run. Uses a singularity container on ppi to run multi_tz.py, and then post_process.py.

### How to use:
nc_config.yaml contains lists of variables to include from meps, and the mapping of their names to the corresponding variables used by ecmwf.
zarr_config.yaml contains the target folder location and name to store the zarr directory.

The meps_to_zarr function in convert.py takes the path of the netcdf file as input, along with an optional argument to downsample the grid.
multi_tz.py is a basic script to run through convert.py multiple times for a range of dates. At the top of this script you can change the:
- start_time: dictionary of year, month, day and hour to start reading data
- end_time: dictionary of year, month, day and hour to stop reading data.
- data frequency: expected frequency of data, default 6 hours.
- nth_point: choose every nth point in the grid along the x and y directions.
- fill_missing: True if missing data should be filled with corresponding lead time of the previous available data. False if skipped.

In the future, we might want to add these parameters to a config file instead.

### Future updates:
- There are still some attributes that should be added to the zarr archive to better correspond with the era zarr files, but it is currently enough to train with aifs-mono.
- Some variables like total column water are not fully implemented.
