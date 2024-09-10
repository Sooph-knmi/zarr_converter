import datetime
import warnings

from typing import Any, Union, Optional
from pathlib import Path 

import zarr 


class ZarrRegistery:

    REGISTER = {
        "author": "KNMI",
        "owner" : "KNMI",
        "model" : "DOWA/HA",
        "description" : """AIFS NETCDF4 data converted to Zarr, leadtime: 6 """,
        "num_files_processed": 0,
        "version" : 0.1
        }
    def __init__(self, path: Union[str, Path]) -> None:
        assert Path(path).is_dir(), f"The path given is not a valid file, got: {path}"
        self.path = path
        self.__initialize_register

    @property
    def _open(self) -> zarr.hierarchy.Group:
        """
            Opens a given zarr archive in read mode. Notice, if the path
            given in the constructor is not a valid zarr path,
            an exception is raised.

            args:
                None
            
            return:
                A zarr.hierarchy.Group object which cannot be modifed
        """
        try:
            return zarr.open(self.path, mode = "r")
        except Exception as e:
            print(f"Exception raised: {e}")
            return None

    @property
    def _write(self) -> zarr.hierarchy.Group:
        """
            Opens a given zarr archive in read and write mode. In order
            to use this method, the zarr dir must exist.

            args:
                None
            
            return:
                A zarr.hierarchy.Group object which can be modified
        """
        return zarr.open(self.path, mode = "r+")
    

    @property
    def __initialize_register(self):
        """
            Private class method. Used to intialize 
            the register which contains all metadata in 
            the zarr archive.

            args:
                None

            return:
                None
        
        """
        z = self._write
        if "register" in z.attrs:
            pass 
        else:
            z.attrs["register"] = self.REGISTER

    def update_history(self, action: str, **kwargs: Any) -> None:
        """
            Update the history within the zarr archive attribute.
            History is defined as action that is being performed on 
            the zarr archive

            args:
                action (str) : A string which describes the action is performed
            
            Return:
                None
        
        """
        z = self._write
        new = dict(
                action = action,
                timestamp = datetime.datetime.utcnow().isoformat(),
            )
        
        new.update(**kwargs)
  
        history = z.attrs.get("history", [])
        history.append(new)

        z.attrs["history"] = history

    def add_basic_metadata(self, start_date, end_date) -> None:
        """
        Adds the basic metadata to the zarr.attrs to match the era5 zarr format.
        """
        z = self._write
        z.attrs['frequency'] = 6 #for now hard coded, should be read elswehere later.
        z.attrs['resolution'] = 'O96' #currently needed to merge files in ecml tools.
        z.attrs['ensemble_dimension'] = 2
        z.attrs['flatten_grid'] = True

        sd_str = f"{start_date['year']}-{start_date['month']:02d}-{start_date['day']:02d}T{start_date['hour']:02d}:00:00"
        ed_str = f"{end_date['year']}-{end_date['month']:02d}-{end_date['day']:02d}T{end_date['hour']:02d}:00:00"
        z.attrs['start_date'] = sd_str
        z.attrs['statistics_start_date'] = sd_str
        z.attrs['end_date'] = ed_str
        z.attrs['statistics_end_date'] = ed_str

    def add_attribute(self, attr_name: str, attr_val) -> None:
        """
            Adds an attribute to the zarr.attrs

            args:
                variable (str,dict): str or dict of variable to add in the metadata
            
            returns:
                None
        """
        
        z = self._write
        
        if attr_name in z.attrs:
            var = z.attrs.get(attr_name, {})
            if isinstance(var, str):

                if variable in var:
                    warnings.warn(f"Variables {variable} already exist in the dataset attribute. Ignoring")
                    return None
                #fix this done below
                variable = dict(variable = variable)
                var.update(variable)
                z.attrs[attr_name] = var

            elif isinstance(var, dict):
                import copy

                copy_var = copy.copy(var)
                for k in var.keys():
                    if k in z.attrs[attr_name]:
                        copy_var.pop(k)
                

                if len(copy_var) != 0:
                    var.update(copy_var)
                    z.attrs[attr_name] = var

                    del copy_var

                else:
                    # warnings.warn(f"The zarr attributes contains all the variables provided in input dict. Ignoring")
                    # return None
                    z.attrs[attr_name] = attr_val


            #removed str and dict support for now
                
            # if isinstance(attr_val, list):
            #     return #so far I don't want to update list attrs.
            
            # if isinstance(attr_val, dict):
            #     return #so far I don't want to update dict attrs.

            else:
                # raise TypeError(f"Attribute has not correct type, expected list. Got {type(variable)}")
                z.attrs[attr_name] = attr_val
        else:
            z.attrs[attr_name] = attr_val


    @property
    def update_num_files(self):
        raise NotImplementedError 

    def remove_attrs(self, timestamp: datetime.datetime):
        raise NotImplementedError

    def new_dataset(self):
        raise NotImplementedError 

