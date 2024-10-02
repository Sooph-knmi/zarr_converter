from convert import meps_to_zarr as mtz
from add_end_data_separately import add_end_data
import glob, os
import time

#Sophie's path
zarr_config = '/hpcperm/nld1247/zarr_converter/zarr_config.yaml'
nc_config= '/hpcperm/nld1247/zarr_converter//nc_config.yaml'
netcdf_folder = "/ec/res4/scratch/nld1247/dowa0812/"

#Bastien's path
# zarr_config = '/ec/res4/hpcperm/ecme5801/DOWA/zarr_converter/zarr_config.yaml'
# nc_config= '/ec/res4/hpcperm/ecme5801/DOWA/zarr_converter/nc_config.yaml'
# netcdf_folder = "/ec/res4/scratch/ecme5801/dowa2013/"

frequency = 3
nth_point = 1
fill_missing = False #whether to fill missing time steps with previous time step, or to skip if False

start_time = {
    "year": 2008,
    "month": 1,
    "day": 1,
}

end_time = {
    "year": 2012,
    "month": 30,
    "day": 31,
}
months = range(1, 13)
# months = [12]
# months = [7]
month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
years = [2008, 2009, 2010, 2012]


mstepcounter = 0
last_pass = None #whether this is the last pass of the loop
start_passed = True #whether the start time has been reached
start_time_tot = time.time()
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
                ncpath = [os.path.join(netcdf_folder, item) for item in os.listdir(netcdf_folder) if item.endswith(f"{year:04d}{month:02d}{day:02d}.nc")]
                # print(ncpath)
                varlist = ["hus.", "huss", "phi", "ps.", "ta.", "ua.", "va.", "w", "psl", "sst", "tas", "uas", "vas", "ts"]
                #ncpath = [os.path.join(netcdf_folder, item) for item in os.listdir(netcdf_folder) if item.endswith(f"{year:04d}{month:02d}{day:02d}.nc")]
                ncpath = [os.path.join(netcdf_folder, item) for item in os.listdir(netcdf_folder) if item.endswith(f"{year:04d}{month:02d}{day:02d}.nc") or item.startswith("orog") or item.startswith("sftof")]
                for path in ncpath:
                    for var in varlist:
                        if path.startswith(netcdf_folder + var) == True:
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