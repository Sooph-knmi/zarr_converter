from tozarr import ToZarr
from dataset import Dataset
import yaml
import os
import numpy as np 
from ecml_tools.data import open_dataset
##### TODO: MAKE CHECK IF DATE IS ALREADY IN ZARR ARCHIVE ###### 
def meps_to_zarr(
        nc_path,
        nth_point = 4,
        last_pass = None,
        mstepcounter = 0,
        zarr_config = None,
        nc_config = None,
         ):
    """
    Takes a netcdf file at nc_path as input and adds it to a zarr archive at the path in zarr_config.yaml

    nth_point: only use every nth point in the x and y directions, this allows for downscaling the meps data.
    last_pass: if True, the zarr archive will be aggregated after the dataset is added.
    mstepcounter: uses the lead time of a previous time step if the current time step is missing, this is used to keep track of how many time steps ahead the current time step is missing.

    NOTE: Currently not parallelized, if needed chunking can be added later.
    """
    
    #LEAD TIME AND COORDINATE SELECTION
    lead_length = range(0, 48, 12) #takes every 6th hour from the dataset

    grid_steps = [nth_point, nth_point] #only use every nth point in the x and y directions, this allows for downscaling the meps data.
    for date_index in lead_length:
        # nc_config["include"] = (nc_path.partition("/")[-1]).partition(".")[0]
        ds = Dataset(
        path=nc_path,  #I probably should have dataset load the config files directly instead of sending them in as function arguments.
        chunks={},  #convert.py is not parallelized, so no need to chunk
        decode_times=True, #we want times in datetime64 format
        config=nc_config,
        )   
        config = ds.config
        print("date", ds.dates[date_index])
        if os.path.exists("/ec/res4/scratch/ecme5801/DOWA_zarr/dowa2013.zarr"):
            print("path exists")
            anemoi_zarr = open_dataset("/ec/res4/scratch/ecme5801/DOWA_zarr/dowa2013.zarr")
            if np.array(ds.dataset["time"])[date_index] in anemoi_zarr.dates:
                print("date already exists, skipping")
                pass
            else:
                meps_to_zarr_create(ds, date_index, grid_steps, config, zarr_config)
                print("adding to dataset")
        else:
            meps_to_zarr_create(ds, date_index, grid_steps, config, zarr_config)
                
def meps_to_zarr_create(ds, date_index, grid_steps, config, zarr_config):
    ds.select_coords(grid_steps[0], grid_steps[1])

    print("date", ds.dates[date_index])

    #GET LAT/LON AND PROJECTION
    ds.projection
    long = ds.longitude
    lat = ds.latitude

    ds.leadtime(length=date_index) # index the dates
    # print(ds.dataset.date)
    #CHANGE DATES TO DATETIME64[s]
    ds.dates = ds.dates.astype('datetime64[s]')

    # CONVERT MODEL LEVELS TO PRESSURE LEVELS
    target = np.array([50,100,150,200,250,300,400,500,600,700,850,925,1000])
    ds.model_to_pressure_levels(config, target)
    
    # ROTATE WINDS
    ds.rotate_wind_parallel("uas", "vas")      #rotates 10m x and y wind
    ds.rotate_wind("ua", "va")        #rotates pressure level x and y wind

    #CHANGE UNITS OF UPWARD WIND VELOCITY
    ds.change_units

    #CALCULATE ADDITIONAL PROPERTIES
    ds.dewpoint_from_specific_humidity
    # ds.dewpoint #calculates the 2m dewpoint temperature
    # ds.total_column_water #calculate total column water

    ds.fill_unimplemented #fills in unimplemented variables with zeros mostly.
    
    ds.get_static_properties #calculates sin/cos of lat/lon/day/time and insolation, and adds to dataset
    #GET DATA, VAR_NAMES, STATISTICS:
    data_array, stats = ds.create_data
    # STORE TO ZARR ARCHIVE:
    tz = ToZarr(config = zarr_config)
    tz.create_dataset(
        action= "added new dataset",
        data= data_array,
        dates = np.array(ds.dataset["time"]),
    #    latitudes = lat,
    #    longitudes = long,
        stats = stats,
    )

# if last_pass is not None:
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
        "year": 2013,
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


if __name__ == "__main__":
    pass

