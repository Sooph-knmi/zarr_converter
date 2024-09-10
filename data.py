import warnings
import netCDF4
from typing import Any, Union, Optional

from functools import cached_property
from pathlib import Path

import xarray as xr
import numpy as np
# import os
import dask

# from dask.distributed import Client
# c = Client(n_workers=os.cpu_count()-2, threads_per_worker=1)


    
def openNetCDF4(path: Union[list[Union[Path, str]], str, Path],
                chunks : Union[tuple, int],
                decode_times : bool = False,
                drop_variables: list = None,
                **kwargs : Any
                ) -> xr.Dataset:
    
    """
        Opens a given netCDF4 file based on xarray API.
        Uses engine netcdf5. If the path is a list of 
        paths the open_mfdataset method is used, if path is not a list
        of paths the regular open_dataset is used. Notice that
        open_mfdataset is not correctly implemented within this project,
        and is still under development.

        args:
            path (list, pathlib.Path, str): Path(s) of netcdf4 file(s)
            chunks (int, tuple): the chunksize
            decode_times (bool): If True, decode times encoded in the standard NetCDF datetime format into datetime objects. 
                                 Otherwise, leave them encoded as numbers. 
                                 This keyword may not be supported by all the backends. 
            
            kwargs (Any): Additional keyword arguments
        
        return:
            A xarray dataset 

    
    """
    dask.config.set(scheduler="single-threaded")
    if isinstance(path, list):
        def preprocessor(file: xr.Dataset):
            f = file.load()
            return f 
    
        try:
            with xr.open_mfdataset(paths = path, 
                                engine="netcdf4",
                                # concat_dim= "time",
                                combine="nested", 
                                # preprocess=preprocessor, 
                                chunks= {"time": 1, "pressure": 1, "x": 198, "y": 198}, 
                                # compat = "broadcast_equals"
                                compat = "override",
                                parallel = True,
                                **kwargs) as dataset:
                ds = dataset

            return ds 
        except Exception as e:
            print(f"An exception occured: {e}") 

    else:
        #warnings.warn("Attention: open_mfdataset is not being since path given is str/Path and not list of paths. This is done sequential")
        path = Path(path)
        #NOTE: add drop vars here later
        try:
            with xr.open_dataset(
                filename_or_obj = path, 
                engine="netcdf4", 
                chunks = chunks, 
                decode_times=decode_times,
                drop_variables=drop_variables,
                **kwargs) as dataset:

                ds = dataset
            return ds 
        except Exception as e:
            print(f"An exception occured: {e}")

class Data:
    def __init__(self, path: Union[str, Path], chunks : Union[str, Path], config: dict, decode_times: bool) -> None:
        """
            
        A super/parent class for the dataset that is being read

        args:
            path (str, pathlib.Path): Path of the dataset
            chunks (int, tuple): The size of the chunk which chunks the dataset
            config: dictionary containing variables to be loaded from the dataset
            decode_times (bool): If True, uses netcdf4 decoding, if false keeping the original format.
        
        return:
            None
        """
        all_meps = []
        for p in path:
            ds_dummy = netCDF4.Dataset(p) #netcdf slightly faster than xr when we just need a list of all variables
            all_meps.extend(list(ds_dummy.variables.keys()))
            ds_dummy.close()
        include_vars = config['include'] #list of variables included in output zarr archive in order
        extract_vars = include_vars + config['drop'] + ['time', 'lat', 'lon', 'Lambert_Conformal'] #additional variables needed for calculations
        exclude_list = list(set(all_meps) - set(extract_vars))
        
        

        self.dataset = openNetCDF4(
            path=path,
            chunks=chunks,
            decode_times=decode_times,
            drop_variables=exclude_list
        )
        print("\nOpened dataset")
        self.pressures = self.dataset["ps"]

    def array_shape(self, varname):
        return self.dataset[varname].shape 
    
    @cached_property
    def longitude(self):
        """
            fetches longitude from the dataset

            args:
                None
            
            return:
                returns a np.ndarray of longitudes
        """
        try:
            return np.array(self.dataset["longitude"]).flatten()
        except KeyError as e:
            print(f"KeyError: {e}.")
            return None


    @cached_property
    def latitude(self):
        """
            fetches latitudes from the dataset

            args:
                None
            
            return:
                returns a np.ndarray of latitudes
        """
        try:
            return np.array(self.dataset["latitude"]).flatten()
        except KeyError as e:
            print(f"KeyError: {e}.")
            return None
    
    
    @cached_property
    def x(self):
        """
            fetches latitudes from the dataset

            args:
                None
            
            return:
                returns a np.ndarray of latitudes
        """
        try:
            return np.array(self.dataset["x"])
        except KeyError as e:
            print(f"KeyError: {e}.")
            return None
    

    @cached_property
    def y(self):
        """
            fetches latitudes from the dataset

            args:
                None
            
            return:
                returns a np.ndarray of latitudes
        """
        try:
            return np.array(self.dataset["y"])
        except KeyError as e:
            print(f"KeyError: {e}.")
            return None
 

    @cached_property
    def dates(self):
        """
            fetches dates from the dataset

            args:
                None
            
            return:
                returns a np.ndarray of dates
        """
        return np.array(self.dataset["time"])
    
    @property
    def projection(self):
        """
            fetches projection that is used for the
            MEPS file.

            args:
                None
            
            return:
                returns a projection string
        """
        try:
            # self._projection = self.dataset.projection_lambert.attrs
            self._projection = self.dataset["Lambert_Conformal"].attrs
            self._projection['earth_radius'] = 6371000.0
            self._projection["standard_parallel"] = [52.500000, 52.500000]
            self._projection['proj4'] = '+proj=lcc +lat_1=52.500000 +lat_2=52.500000 +lat_0=52.500000 +lon_0=.000000 +k_0=1.0 +x_0=649536.512574 +y_0=1032883.739533 +a=6371220.000000 +b=6371220.000000'
        except AttributeError:
            print("AttributeError: projection not found")
            return None
    
    @cached_property
    def variables(self):
        """
            fetches variables that is present within the 
            dataset

            args:
                None
            
            return:
                returns a view of the variables as well their metadata
        """
        return self.dataset.variables
    
    @cached_property
    def pressures(self):
        return self.pressures
    
    @cached_property
    def view_metadata(self):
        """
            View the metadata in the dataset

            args:
                None

            return: 
                None
        
        """
        print(self.dataset) 
    
    @property
    def close(self):
        """
            Closes the dataset that is opened, and the 
            deletes the variable associated to it.

            args:
                None
            
            return:
                None
        
        """
        self.dataset.close()
        del self.dataset


