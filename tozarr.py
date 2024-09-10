import datetime

from typing import Any, Optional, Union, Dict
from functools import cached_property

import zarr
import numpy as np

from zarr_utils import add_data, add_dataset


class Core:
    def __init__(self, *, config = None) -> None:
        self.configpath = "/lustre/storeB/project/nwp/aifs/code/aifs-support/zarr_converter/zarr_config.yaml" if config is None else config
        self.config = self.__open_config
        self.path = self.config['store']

    @cached_property
    def __open_config(self):
        """
            Opens a given yaml file which IS used for
            the dataset configuration.
            
        """
        import yaml
        try:
            with open(self.configpath, 'r') as file:
                # Load YAML configuration
                config = yaml.safe_load(file)
                return config

        except FileNotFoundError:
            print(f"Error: YAML file not found at {self.configpath}")
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")

    @cached_property
    def registry(self):
        """
            initiates the zarr registery class,
            used for updating or adding metadata.

            args:
                None
            
            return:
                an object of ZarrRegistery class
        
        """
        from zarrregistry import ZarrRegistery

        return ZarrRegistery(path = self.path)

    @property
    def initialise_dataset_backend(self)-> zarr.hierarchy.Group: 
        """
            initialises the dataset, and creates a
            hierarchy based on config file. However it
            also return an object which can be used to add
            data, dataset,etc.. to the zarr archive.

            args:
                None
            
            return:
                returns the root of the zarr archive which is opened or 
                ready for modification
        
        """
        return self.__create_or_open_zarr
    
    @property
    def __create_or_open_zarr(self):
        """
            Private function which either open a
            existing zarr archive, if in the case the zarr
            archive do not exist it is created. 

            args:
                None

            return:
                return object which represent the root of the zarr archive, which
                can be modified.
        
        """
        try:
            root = zarr.open_group(store=self.config["store"], mode = "r+")
        except zarr.errors.GroupNotFoundError:
            print("Initializing zarr dataset archive")
            root = zarr.group(store=self.config["store"])
            for grps in self.config["groups"]:
                root.create_group(name = grps)
        return root
    
    def add(self,store: zarr.hierarchy.Group, **kwargs: Any) -> zarr.hierarchy.Group:
        """
            Adds data, values or dataset into the
            zarr archive.

            args:
                store (zarr.hierarchy.Group): the root of the zarr archive where the data is added
            
            return:
                returns an object zarr.hierarchy.Group
        
        """
        #if "values" in kwargs:
        #    return add_data(store_root=store, **kwargs)
        
        return add_dataset(store_root = store, **kwargs)


class ToZarr(Core):
    def __init__(self,**kwargs) -> None:
        super().__init__(**kwargs)
    
    def create_dataset(self, 
                       *,
                       action: str,
                       data : np.ndarray,
                       dates : np.ndarray,
                       #latitudes : np.ndarray,
                       #longitudes : np.ndarray,
                       stats : dict,
                       ) -> None:
        """
            create_dataset is class method which generates 
            zarr dataset. This method adds statistics, dataset
            ,attributes, dates,etc... to the archive. Keep in mind,
            that this method can also be used for keep adding new
            datasets, dates, etc.. to make the archive grow.

            args:
                action (str): A string which describes the action is performed
                data (np.ndarray): Dataset that is added to the archive
                dates (np.ndarray): Dates that is added to the archive
                latitudes (np.ndarray): Latitude coordinates that is added to the archive
                longitudes (np.ndarray): Longitude coordinates that is added to the archive
                stats (dict(np.ndarray)): A dictonary of statistics that is added to the _build archive
                metadata (dict): A dict containing information of variables of the dataset
            
            return None
        """

        root = self.initialise_dataset_backend
        
        # update_metadata = self.registry
        print("Adding data to the archive")
        print("shape of data being added to the archive", data.shape)
        # self.add(store = root, name = "data", data = data, ds_shape = data.shape, chunks = tuple([1] + list(data.shape[1:])), ds_dtype = data.dtype)
        self.add(store = root, name = "data", data = data, ds_shape = data.shape, chunks = tuple([1] + list(data.shape[1:])), ds_dtype = data.dtype , add_method = "append")
        self.add(store = root, name = "dates", data = dates, ds_shape = dates.shape, add_method = "add_to_array")
        #lats and longs should only be added if they are not already in the archive, not appended like the above
        # self.add(store = root, name = "longitudes", data = longitudes, ds_shape = longitudes.shape, add_method = "replace")
        # self.add(store = root, name = "latitudes", data = latitudes, ds_shape = latitudes.shape, add_method = "replace")
        #TODO: moved lats and lons to lastpass in convert.py 
        # print(stats)
        self.add(store = root["_build"], data = stats, add_method = "add_to_array", name ="build_stats")
        # print("root build content", root["_build"])
        #update_metadata.update_history(
        #    action = action
        #)

        #update_metadata.update_others() #adds additional attributes to the zarr archive to better match era.

        # update_metadata.add_variable(variable=metadata)
        # update_metadata.add_attribute(attr_name= "variables", attr_val= metadata)

        
    def aggregate(self) -> None:
        """
            Aggregates the chunks of statistics gathered
            from indivudal files stored in _build archive 
            to calculate the global statistics. The updated
            and aggregated statistics are saved in the root 
            folder.

            args:
                None
            return:
                None
        
        """
        from aggregate import Aggregate

        aggr = Aggregate(path = self.path)
        aggr.update_statistics()

if __name__ == "__main__":
    a = np.random.rand(6,85,100414)

    tz = ToZarr(path = "/home/arams/Documents/project/met-tools/datasets/aram.zarr", config = "/home/arams/Documents/project/met-tools/met_tools/zarr_config/config.yaml")

    tz.create_dataset(data = a, action= "Adding more data to the dataset, no aggregation of statistics performed")
    #aggr = Aggregate(path = "/home/arams/Documents/project/met-tools/datasets/aram.zarr")
    #aggr.update_statistics()
