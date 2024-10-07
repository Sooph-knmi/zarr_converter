from tozarr import ToZarr
from dataset import Dataset
import yaml
import os
import numpy as np 
from ecml_tools.data import open_dataset

def add_end_data(nc_path, nth_point = 4, last_pass = None, mstepcounter = 0, zarr_config = None, nc_config = None):
    ds = Dataset(
        path=nc_path,  #I probably should have dataset load the config files directly instead of sending them in as function arguments.
        chunks={},  #convert.py is not parallelized, so no need to chunk
        decode_times=True, #we want times in datetime64 format
        config=nc_config,
        )   
    tz = ToZarr(config = zarr_config)
    start_time = {
        "year": 2013,
        "month": 1,
        "day": 1,
        "hour": 00
    }
    # start_time = {
    #     "year": 2020,
    #     "month": 2,
    #     "day": 5,
    #     "hour": 0
    # }

    end_time = {
        "year": 2017,
        "month": 12,
        "day": 31,
        "hour": 00
    }
    last_pass = [start_time, end_time]
    var_labels, map_dict = ds.get_var_names

    print("Aggregating statistics")
    tz.aggregate()
    tz.registry.update_history("Added dataset")

    print("Adding metadata")
    tz.registry.add_basic_metadata(start_date = last_pass[0], end_date = last_pass[1])
    tz.registry.add_attribute(attr_name = "era_to_dowa_mapping", attr_val = map_dict)
    tz.registry.add_attribute(attr_name = "variables", attr_val = var_labels)
    # add lat, lon, proj, x, y
    root = tz.initialise_dataset_backend
    #shift long to [0,360]
    long[long < 0] = long[long < 0] + 360
    tz.add(store = root, name = 'longitudes', data = long, ds_shape = long.shape, add_method = 'overwrite')
    tz.add(store = root, name = 'latitudes', data = lat, ds_shape = lat.shape, add_method = 'overwrite')
    tz.add(store = root, name = 'x', data = ds.x, ds_shape = ds.x.shape, add_method = 'overwrite')
    tz.add(store = root, name = 'y', data = ds.y, ds_shape = ds.y.shape, add_method = 'overwrite')
    # Have to convert np.array to list in proj to store it in zarr archive
    ds.projection
    proj = ds._projection
    proj['standard_parallel'] = list(proj['standard_parallel'])
    tz.registry.add_attribute(attr_name = "projection_lambert", attr_val = proj)
    # tz.add(store = root, name = 'projection_lambert', data = ds._projection)
    print("Finished creating zarr dataset")