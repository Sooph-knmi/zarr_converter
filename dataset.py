import yaml

from typing import Any, Optional, Union
from pathlib import Path 
from functools import cached_property

import numpy as np 

from core import Core

class Dataset(Core):
    def __init__(self, path, chunks, decode_times, config = None) -> None:
        # TODO: make this class more chainable (later work)

        # TODO : make this class or in general handle memory better. This setup does not utilize the memory effectively. 
        
        self.configpath = "/lustre/storeB/project/nwp/aifs/code/aifs-support/zarr_converter/nc_config.yaml" if config is None else config
        self.config = self.__open_config

        super().__init__(path, chunks, self.config, decode_times)

    def create_dataset(
            self, 
            **kwargs
            ) -> np.ndarray:
        """
            Creates a numpy ndarray dataset from xarray dataset. 
            # TODO: (under development) find a way to optimize creating the dataset

            args:
                **kwargs: Any
                        -> config bool: if true, using the variable names from the configfile
                        -> varname list/str: a list of or a string containing variable(s)
                        -> concat bool : if true, the dataset is concatenated within a axis
                        -> axis int/tuple: The axis used to concatenate the dataset
                        -> dtype Any: the dtype used in when creating the dataset
            
            return:
                returns a dataset of type np.ndarray with a given shape. 
        
        
        
        """
        if kwargs["config"]:
            self.to_include = self.config["include"]
        else:
            assert isinstance(kwargs["varname"], (str), list), f"varname is not str or list of variable names. Got {kwargs['varname']}"
            self.to_include = kwargs["varname"]

        if kwargs["concat_axis"]:
            self._axis = kwargs["concat_axis"]

        if kwargs["concat"]:
           
            # TODO : think of a way to map variable name to correspondig index'
            
            params = [self.dataset[var].values for var in self.to_include]
        
            self.dataset = self.concatenate_array(params= params, axis= kwargs["axis"], dtype= kwargs["dtype"])
    
            return self.dataset#, self.create_metadata

        else:
            self.dataset = self.dataset[self.to_include].values
            
            return self.dataset#, self.create_metadata
        
    @cached_property
    def create_metadata(self):
        """
            Creates metadata of the variables used to create the dataset. 
            Currently this method is fetching mapping names in the yaml config file.
            Keep in mind that variablenames containing _pl has pressure as coordinates.
            This is added in the metadata

            args:
                None
            
            return:
                a dict of metadata
        
        """
        import copy
        pressure_levels = np.array(self.dataset.coords["pressure"][:])
        metadata = copy.copy(self.config["mapping"])
        for k, v in metadata.items():
            if "_pl" in k:
                metadata[k] = [v + "_" + str(int(pl)) for pl in pressure_levels]
        
        return metadata
    
    @cached_property
    def get_var_names(self):
        """
            Gets the variable names and mappings from the config file.
            Keep in mind that variablenames containing _pl has pressure as coordinates.

            Returns a list of variable names in the same order created in create_data,
            along with a mapping dict that can be used to map the variable names back to the original names.
        """
        # pressure_levels = np.array(self.dataset.coords["pressure"][:])
        var_labels = []                #list to store the variable names
        mapping = self.config['mapping']     #variable name mapping
        map_dict = {}
        # pressure_levels = np.array(self.dataset.coords["pressure"][:])
        pressure_levels = np.array([50,100,150,200,250,300,400,500,600,700,850,925,1000])

        for varname in self.config['include']:
            #FETCH AND STORE VARIABLE NAMES
            # if "_pl" in varname:
            if varname in self.config["pl_var"]:
                for pl in pressure_levels:
                    pl = int(pl)
                    varmap = mapping[varname] + f"_{pl}"
                    var_labels.append(varmap)
                    map_dict[varmap] = varname

            else:
                var_labels.append(mapping[varname])
                map_dict[mapping[varname]] = varname
        #TODO Needs to include a check to ensure all variables are mapped
        return var_labels, map_dict
    
    @cached_property
    def create_data(self):
        """
        Stacks the data into a float32 numpy array.
        Returns this array.
        Also calculates and returns the statistics of the data.
        """

        var_vals = []                      #list to store the data
        # max_shape = self['x_wind_pl'].shape  #shape of the largest variable
        max_shape = self["hus"].shape

        for varname in self.config['include']:

            #RESHAPING FOR DIFFERENT VAR SHAPES:
            # self.dataset = self.dataset.chunk{}
            var_dims = self.dataset[varname].dims   #Shape of the variable

            if len(var_dims) == 4 and set(('time', 'y', 'x')).issubset(var_dims):
                dim1len = self.dataset[varname].shape[1]
                from scipy.ndimage import uniform_filter
                var_val = self[varname].values.reshape((max_shape[0], dim1len, max_shape[2]*max_shape[3]))  #these parameters have the desired shape, with the second axis being either 1 or 13 in length
                #var_val = uniform_filter(self.dataset[varname].values, axes=[2,3]).reshape((max_shape[0], dim1len, max_shape[2]*max_shape[3]))
                
            else:
                if var_dims == ('time', 'y', 'x'):
                    var_val = self.dataset[varname].values
                    var_val = var_val[:,np.newaxis, ...].reshape((max_shape[0], 1, max_shape[2]*max_shape[3]))
                    # var_val = var_val[:,np.newaxis, ...].reshape((max_shape[0], 1, max_shape[2]*max_shape[3]))
                elif 'time' in var_dims:
                    # var_val = self[varname].values
                    var_val = self.dataset[varname].values
                    # var_val = var_val[:,np.newaxis, np.newaxis].repeat(max_shape[2] * max_shape[3], axis=2)
                    var_val = var_val[:,np.newaxis, np.newaxis].repeat(max_shape[2] * max_shape[3], axis=2)
                else: #coordinates y x are alone
                    # var_val = self[varname].values.flatten() #flatten lat long
                    var_val = self.dataset[varname].values.flatten()
                    var_val = var_val[np.newaxis, np.newaxis, ...].repeat(max_shape[0], axis = 0) 
                            
            var_vals.append(var_val)
        #COLLECT DATA AND CALCULATE STATISTICS

        data_array = np.concatenate(var_vals, axis=1)
        stats = self.alt_statistics(data=data_array)  #statistics should be in float64, but the data should be in float32
        data_array = data_array.astype(np.float32)
        data_array = data_array[:, :, np.newaxis, :]   #adds the ensemble dimension to the data

        return data_array, stats



    
    @property
    def statistics(self) -> dict:
        """
            Generates a dict of statistics. It calculates
            sum of squares (sum(x**2)), sum (sum of x), max, min,
            and the length of the array. This must be used in aggregation
            for calculating the global statistics.

            args:
                None
            return:
                dict of statistics
        
        """
        return {
            "squares": self.squares(axis = self._axis),
            "sums" : self.sums(axis=self._axis),
            "max" : self.maxValue(axis= self._axis),
            "min" : self.minValue(axis = self._axis),
            "count" : [self.dataset.shape[0]*self.dataset.shape[2]]*self.dataset.shape[1]
        }
    
    def alt_statistics(self, data: np.ndarray) -> dict:
        """
        Instead uses a numpy array to calculate the satstistics
        Note that this should not sum over time, since it is put inside _build
        """
        axes = (0,2) #axis 0 should be 1 element, or multiple if ensemble, maybe then sum not wanted?
        print("calculation sum of squares")
        return {
            "squares": np.sum(data**2, axis = axes),
            "sums" : np.sum(data, axis = axes),
            "maximum" : np.max(data, axis = axes), #NOTE: Should be named maximum to be consistent with era
            "minimum" : np.min(data, axis = axes), #NOTE: Should be named minimum to be consistent with era
            "count" : np.array([data.shape[0]*data.shape[2]]*data.shape[1], dtype = np.int64)
        }


    def leadtime(self, length):
        """
            Select a given leadtime to filter the dataset.
            keep in mind length is not int or float, should be either
            a list of leadtimes or range(start, end)

            args:
                length: range(start, end)
            
            return:
                updates the object self
        
        """
        self.dataset = self.dataset.isel(time = [length])
        return self
    
    def select_coords(self, ystep, xstep):
        """
            Selects every nth value along the y and x axis on the dataset
            
            return:
                updates the object self
        
        """
        yrange = range(0, len(self.dataset.coords["y"]), ystep)
        xrange = range(0, len(self.dataset.coords["x"]), xstep)
        self.dataset = self.dataset.isel(y = yrange, x = xrange)
        # self.dataset = self.dataset.thin({"x": xstep, "y": ystep})
        return self

    def drop_entry(self, labels: str, axis: Optional[Union[int,tuple]] = None) -> Any:
        """
            Drops an entry within the xarray dataset.
            returns an updated object.

            args:
                labels (str): label to drop
                axis (tuple,int): where to drop, optional param
            
            return:
                updated dataset object
        
        """
        self.dataset = self.dataset.drop(labels=labels,dim=axis)
        return self
    
    def select(self, varname = None , var_in_config: bool = None, key : Optional[str] = None):
        # TODO: find a way to display the selected variable and have chain option
        # for now only chain option is supported, meaning if print(data.select(varname)) is applied then a object is visible

        """
            Selects variables to fetch based on a list or str 
            of variable name(s), however the user can also use a 
            config file (yaml) to give a list of variable names to select. 
            Returns a updated object.

            args:
                varname (str, list, None): list of variable names to be selected
                var_in_config (bool, None): if true, fetches a list of variable names from the config
                key Optional(str): extract or include keynames
            
            return:
                A updated obj
        
        """
        if var_in_config:
            if key in self.config:
                extract = self.config[key]
                self.dataset = self.dataset[extract]

                return self#.dataset[extract]

        elif varname:
            self.dataset = self.dataset[varname]
            return self
        else:
            raise ValueError(f"Please provide a variable to select or set var_in_config to True")


    def __getitem__(self, key) -> np.ndarray | Any:
        """
            Fetches a variable within the dataset. Note KeyError
            is raised if key is not present.

            args:
                key (str): name of a variable or param within the dataset
            return:
                returns the value of the key which is present in the dataset
        """

        try:
            return self.dataset[key]
        except KeyError as e:
            print(f"KeyError: {e}")

            return None
    
    def __setitem__(self, varname: str, value: np.ndarray) -> Any:
        """
            Sets an item within the dataset based on the varname. 
            However if the varname exist, a keyerror is 
            raised and thus exiting the program.
            If not, the key will be placed in the dataset as
            dataset[varname] = (value.shape, value)

            args:
                varname (str): The name of the variable to be set 
                value (np.ndarray): Value of the varname
            
            returns:
                An updated object        
        """

        if varname in list(self.variables):
            raise KeyError(f"The variable name provided already exist, please select another variable name. Got {varname}")

        if isinstance(value, (int, float)):
            self.dataset[varname] = value
        else:
            # If value has a shape, use that shape as dimensions
            try:
                _shape = value.shape
            except AttributeError:
                _shape = (len(value),)

            self.dataset[varname] = (_shape, value)
        
        return self
    
    @cached_property
    def __len__(self):
        """length of the dataset"""

        return len(self.variables)
    
    @cached_property
    def __open_config(self):
        """
            Opens a given yaml file which used for
            the dataset configuration.
            
        """
        try:
            with open(self.configpath, 'r') as file:
                # Load YAML configuration
                config = yaml.safe_load(file)
                return config

        except FileNotFoundError:
            print(f"Error: YAML file not found at {self.configpath}")
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")


if __name__ == "__main__":
    directory = Path("/home/arams/Documents/")
    paths = [Path("/home/arams/Documents/") / "meps_det_2_5km_20230101T12Z.nc", Path("/home/arams/Documents/")  / "meps_det_2_5km_20231115T00Z.nc"]
    filename = "meps_det_2_5km_20230101T12Z.nc"
    ds = Dataset(
        path = paths[0],#directory / filename,
        chunks= {"time" : 10},
        decode_times=False,
        config="/home/arams/Documents/project/met-tools/met_tools/netcdf4_config/config.yaml"
    )
    
    
    ds.leadtime(length=range(0,6))
    ds.projection
    ds.select(var_in_config=True, key="extract")
    long = ds.longitude
    lat = ds.latitude
    dates = ds.dates
    ds.dewpoint
    ds.rotate_wind
    config = ds.config
    
    ds.select(var_in_config=True, key="include")

    ds.create_metadata
    
    for varname in config["include"]:
        old_dim = ds.array_shape(varname=varname)
        new_dim = (old_dim[0], old_dim[1], old_dim[2]*old_dim[3])
        ds.reshape_array(varname,dim=new_dim, inplace=True)

    dataset = ds.create_dataset(
        config = True,
        concat = True, 
        dtype = np.float32,
        concat_axis = 1,
        axis = (0,2)
    )
    stats = ds.statistics
