from convert import meps_to_zarr as mtz
from add_end_data_separately import add_end_data
import glob, os
import time

zarr_config = '/ec/res4/hpcperm/ecme5801/DOWA/zarr_converter/zarr_config.yaml'
nc_config= '/ec/res4/hpcperm/ecme5801/DOWA/zarr_converter/nc_config.yaml'
netcdf_folder = "/ec/res4/scratch/ecme5801/dowa2013"
#
frequency = 3
nth_point = 1
fill_missing = False #whether to fill missing time steps with previous time step, or to skip if False

start_time = {
    "year": 2013,
    "month": 1,
    "day": 1,
}

end_time = {
    "year": 2013,
    "month": 1,
    "day": 2,
}

months = range(1, 13)
month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
years = [2013]





mstepcounter = 0
last_pass = None #whether this is the last pass of the loop
start_passed = False #whether the start time has been reached
start_time_tot = time.time()
year=2013



#To do sftof to plot!!!!



for year in years:
    if year % 4 == 0 and year % 100 != 0:
        month_days[1] = 29
    else:
        month_days[1] = 28

    for i, month in enumerate(months):
        for day in range(1, month_days[i]+1, 1):
            # for hour in hours:
            start_time_epoch = time.time()
            if last_pass is None:
                if {"year": year, "month": month, "day": day} == end_time:
                    start_time["hour"]=0
                    end_time["hour"]=23
                    last_pass = [start_time, end_time]  #info to be used for the last pass of the loop

            if not start_passed:
                if {"year": year, "month": month, "day": day} == start_time:
                    start_passed = True
            if start_passed:
                print("start passed")
                varlist = ["hus", "phi", "ps.", "ta.", "ua.", "va.", "w", "psl", "sst", "tas", "uas", "vas", ]
                #ncpath = [os.path.join(netcdf_folder, item) for item in os.listdir(netcdf_folder) if item.endswith(f"{year:04d}{month:02d}{day:02d}.nc")]
            ##modifb
                #varlist = ["hus", "huss", "orog", "phi", "ps.", "ta.", "ua.", "va.", "w", "psl", "sst", "tas", "uas", "vas", ]
                ncpath = [os.path.join(netcdf_folder, item) for item in os.listdir(netcdf_folder) if item.endswith(f"{year:04d}{month:02d}{day:02d}.nc") or item.startswith("orog") or item.startswith("sftof")]
            ### end modifb
                for path in ncpath:
                    for var in varlist:
                        if path.startswith("/ec/res4/scratch/ecme5801/dowa2013/" + var) == True:
                            # print(var)
                            varlist.remove(var)
                if fill_missing:
                    if len(varlist)!=0:
                        mstepcounter += 1
                        ncpath = previous_existing_step
                        print(f"Missing time step at {year:04d}/{month:02d}/{day:02d}, using previous existing time step with lead time {mstepcounter*6} hours")
                    else:
                        previous_existing_step = ncpath
                        mstepcounter = 0
                        mtz(ncpath, nth_point, last_pass, mstepcounter, nc_config=nc_config, zarr_config=zarr_config)
                else:
                    if len(varlist)==0:
                        mtz(ncpath, nth_point, last_pass, mstepcounter, nc_config=nc_config, zarr_config=
                        zarr_config)
                    else:
                        print(f"Missing time step at {year:04d}/{month:02d}/{day:02d}, skipping")
        print(f"{day}/{month}/{year}:--- {time.time() - start_time_epoch} seconds ---")
print(f'total time: {time.time()-start_time_tot}')






###Plot
from aifs.diagnostics.maps import Coastlines
from aifs.diagnostics.plots import EquirectangularProjection
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from textwrap import wrap
import xarray as xr
from ecml_tools.data import open_dataset

def scatter_plot_dowa(fig, ax, lon: np.array, lat: np.array, data: np.array, cmap: str = "viridis", s: Optional[float] = 0.5) -> None:
    """Lat-lon scatter plot: can work with arbitrary grids.

    Parameters
    ----------
    fig : _type_
        Figure object handle
    ax : _type_
        Axis object handle
    lon : np.ndarray
        longitude coordinates array, shape (lon,)
    lat : np.ndarray
        latitude coordinates array, shape (lat,)
    data : _type_
        Data to plot
    cmap : str, optional
        Colormap string from matplotlib, by default "viridis"
    title : _type_, optional
        Title for plot, by default None
    """
    psc = ax.scatter(
        lon,
        lat,
        c=data,
        cmap=cmap,
        s=10,
        alpha=1.0,
        marker='s',
        norm=TwoSlopeNorm(vcenter=0.0) if cmap == "bwr" else None,
        rasterized=True,
    )
    
    ax.set_xlim((-0.3, 0.45))
    ax.set_ylim((0.72, 1.08))
    return psc



def ZarrSinglePlotNoBoundaries(fig, ax, ds, var, time):
    lonlat = np.append(np.expand_dims(ds.latitudes, axis=1),np.expand_dims(ds.longitudes, axis=1), axis = 1)
    pc = EquirectangularProjection()
    lat, lon = lonlat[:, 0], lonlat[:, 1]
    pc_lon, pc_lat = pc(lon, lat)
    index = ds.name_to_index[var]
    title = f'{var} at {ds.dates[time]}'
    ds1 = ds.data[time, index]
    # calculate variable and plot figure
    psc = scatter_plot_dowa(fig, ax, pc_lon, pc_lat, ds1, s=10)
    fig.colorbar(psc, ax=ax)
    ax.set_title("\n".join(wrap(title, 40)))
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_aspect("auto", adjustable=None)
    Coastlines().plot_continents(ax)
    



#from plots import ZarrSinglePlotNoBoundaries
ds = open_dataset("/ec/res4/scratch/ecme5801/DOWA_zarr/dowa2013.zarr")
folder_path = "/ec/res4/hpcperm/ecme5801/DOWA/Plots/"
res = "5"
var = "sst"
var = "2d"
var = "tp"
var = "z"
var = "orog"
var = "z"
var = "lsm"

ds.name_to_index
ds.name_to_index["lsm"]

list_var_surface= ["z"] 

# "sp", "msl", "sst", "10u", "10v",  "2t", "2d", "skt",
#            "tp", "cp", "tcw", "insolation",
#            "cos_latitude", "cos_longitude", "sin_latitude", "sin_longitude", 
#            "cos_julian_day", "sin_julian_day",
#            "cos_local_time", "sin_local_time"]

list_var_pressure_levels= ["z", "u", "v", "w", "q", "t"]
#  hus : q
#   ta : t
#   ua : u
#   va : v
#   w: w
#   phi : z
#   ps : sp
#   psl : msl
#   sftof : lsm
#   orography_stddev: sdor  #note that this one and the next one will be be set to zeros for now to indicate that our grid is fine enough
#   orography_slope: slor
#   uas : 10u
#   vas : 10v
#   tas : 2t
#   2d : 2d
#   sst : sst
#   cp : cp
#   prrain: tp
#   cos_latitude : cos_latitude
#   cos_longitude : cos_longitude
#   sin_latitude : sin_latitude
#   sin_longitude : sin_longitude
#   cos_julian_day : cos_julian_day
#   cos_local_time : cos_local_time
#   sin_julian_day : sin_julian_day
#   sin_local_time : sin_local_time
#   insolation : insolation
#   tcw : tcw
#   tp: tp
#   lsm: lsm
#   z: z
#   ts: skt
#   huss: q_2m
  
os.chdir(folder_path)
for var in list_var_surface:
    print(var)
    time = 0
    fig, ax = plt.subplots(1,1, figsize=(12,8))
    ZarrSinglePlotNoBoundaries(fig, ax, ds, var, time)
    plt.savefig(f'{var}_{ds.dates[time]}.png')
    plt.close()
    
plt.show()

for var in list_var_pressure_levels:
    for press_lvl in ["_50", "_100", "_150", "_200", "_250", "_300", "_400", "_500", "_600", "_700", "_850", "_925", "_1000"]:
        var_press=var+press_lvl
        print(var_press)
        time = 0
        fig, ax = plt.subplots(1,1, figsize=(12,8))
        ZarrSinglePlotNoBoundaries(fig, ax, ds, var_press, time)
        plt.savefig(f'{var_press}_{ds.dates[time]}.png')
        plt.close()


plt.show() 




#Laboratoire:
from dataset import Dataset
ds = Dataset(
    path=ncpath,  #I probably should have dataset load the config files directly instead of sending them in as function arguments.
    chunks={},  #convert.py is not parallelized, so no need to chunk
    decode_times=True, #we want times in datetime64 format
    config=nc_config,
    )  
ds.projection
long = ds.longitude
ds.dataset["prrain"].values
ds.dataset["sst"].values
arr1=ds.dataset["tas"]
arr1.values
arr2=ds.dataset["sst"]
arr2=ds.dataset["huss"]
arr2.values
arr1.shape
import gridpp
from gridpp import dewpoint
import numpy as np

self=ds
ncpath
nth_point
last_pass
mstepcounter
#config=nc_config
zarr_config=zarr_config
grid_steps = [nth_point, nth_point] #only use every nth point in the x and y directions, this allows for downscaling the meps data.
lead_length = range(0, 48, 12) #takes every 6th hour from the dataset
date_index=lead_length[1]
config = ds.config

def mtz(ds, date_index, grid_steps, config, zarr_config):
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
    
mtz(ncpath, nth_point, last_pass, mstepcounter, nc_config, zarr_config)

#End LAboratoire