import warnings

from pathlib import Path 
from functools import cached_property
from typing import Any, Optional, Union

from zarr_utils import mod_to_pres, rotate_wind_func, change_units_func
from data import openNetCDF4

import numpy as np
import xarray as xr
from data import Data

class Core(Data):
    def __init__(self, path: Union[str, Path], chunks: Union[str, Path], config: dict, decode_times: bool) -> None:

        # TODO: find an efficent way to handle nans, in order to do a summation. np.nansum results in killed message
        # -> this is because np.nansum, etc... copies the data. The data occupies atleast 2.1gb mem, so 4 copies will results in 8gb tmp memory eating up the 
        # -> memory, thus resulting in killed message
        super().__init__(path = path, chunks=chunks, config = config, decode_times=decode_times)
        self.dataset = self.dataset.rename({"lon": "longitude", "lat": "latitude"})
        self.cache = {}
        
    def sums(
        self,
        axis: Optional[Union[int, tuple]] = 0,
        ) -> Union[float, np.ndarray]:

        """
            Calculates the sum of x -> sum(x) for a given 
            axis. NB, this method is still under development.
            Current issues: memory and nans, the solution 
            down below is a temp solution.

            args:
                axis Optional(int, tuple): the axis as to where the sum is happening
            
            return 
                returns the sum of the array for that given axis
        
        """
        # TODO : spør Thomas om man skal ignorere nans og sette de lik null.
        #if "sums" in self.cache and axis == self.cache["sums"]:
        #    return self.cache["sums"][0]
        #_sum = self.dataset.sum(axis = axis)
        #dim1 = axis[0]
        # TODO: temp solution down below

        _sum = np.sum([np.nansum(chunk, axis=1) for chunk in self.dataset], axis= 0)
        
        #self.cache["sums"] = (_sum, axis)
        return _sum

        
    def squares(
            self,
            axis : Optional[Union[int, tuple]] = 0,
        ) -> np.ndarray:
        """
            Calculates the sum of squares -> sum(x**2) for a given 
            axis. NB, this method is still under development.
            Current issues: memory and nans, the solution 
            down below is a temp solution.

            args:
                axis Optional(int, tuple): the axis as to where the sum is happening
            
            return 
                returns the sum of the array squared for that given axis
        
        """
        # TODO : spør Thomas om man skal ignorere nans og sette de lik null.
        #if "squares" in self.cache and axis == self.cache["squares"]:
        #    print(f"Using cache, retrieving squares")
        #    return self.cache["squares"][0]

        #_squares = (np.square(self.dataset)).sum(axis = axis)
        # TODO: temp solution down below
        sqrs = np.square(self.dataset)
        _squares = [np.nansum(chunk, axis=1) for chunk in sqrs]
        _squares =np.sum(_squares, axis=0)

        return _squares
    
    def minValue(
            self,    
            axis : Union[int, tuple] = 0,
            ) -> Union[np.ndarray, float]:
        """
            Calculates the min of x for a given 
            axis. NB, this method is still under development.
            Current issues: memory and nans, the solution 
            down below is a temp solution.

            args:
                axis Optional(int, tuple): the axis as to where the min is happening
            
            return 
                returns the min of the array for that given axis
        
        """
        #if 'min' in self.cache and axis == self.cache["min"][1]:
        #    return self.cache["min"][0]
         
        _min = np.nanmin(self.dataset, axis= axis)
        #self.cache["min"] = (_min, axis)

        return _min

    
    def maxValue(
            self,
            axis : Union[int, tuple] = 0,
            ) -> Union[np.ndarray, float]:
        """
            Calculates the max of x for a given 
            axis. NB, this method is still under development.
            Current issues: memory and nans, the solution 
            down below is a temp solution.

            args:
                axis Optional(int, tuple): the axis as to where the max is happening
            
            return 
                returns the max of the array for that given axis
        
        """
        #if 'max' in self.cache and axis == self.cache["max"][1]:
        #    return self.cache["max"][0]
         
        _max = np.nanmin(self.dataset, axis= axis)
        #self.cache["max"] = (_max, axis)

        return _max
    
    def reshape_array(
            self, 
            varname: str, 
            dim, 
            inplace = False
            ) -> np.ndarray:
        """
            Reshapes a given variable within the dataset, where dim is the new
            shape of that array.

            return:
                An updated object or dataset[varname]
        """
        self.dataset[varname] = (dim, self.dataset[varname].values.reshape(dim))
        if inplace:
            return self
        else:
            return self.dataset[varname]

    @property
    def dewpoint(self) -> None:
        """
            Calculates the dewpoint. This method used
            air_temperature_2m and relative_humidity_2m. 
            Adds the dewpoint to the dataset as the key 2d

            args:
                None
            
            returns:
                    dataset['2d']

        """
        from zarr_utils import calculate_dewpoint

        arr1 = self.dataset["ta"]
        arr2 = self.dataset["relative_humidity_2m"]
        dewpoint = calculate_dewpoint(arr1.values, arr2.values)
        
        self.dataset["2d"] = (arr_tas.dims, dewpoint)
        return self.dataset["2d"]
    
    @property
    def dewpoint_from_specific_humidity(self) -> None:
        """
            Calculates the dewpoint. This method used
            tas, huss and ps variables. 
            Adds the dewpoint to the dataset as the key 2d

            args:
                None
            
            returns:
                    dataset['2d']
        """
        from zarr_utils import calculate_relative_humidity
        from zarr_utils import calculate_dewpoint

        arr_tas = self.dataset["tas"]
        arr_huss = self.dataset["huss"]
        arr_ps = self.dataset["ps"]
        
        print('DEWPOINT FROM SPECIFIC HUMIDITY')
        tmp_RH=calculate_relative_humidity(arr_tas.values, arr_huss.values, arr_ps.values)   
        dewpoint = calculate_dewpoint(arr_tas.values, tmp_RH)
        print('END DEWPOINT')

        self.dataset["2d"] = (arr_tas.dims, dewpoint)
        return self.dataset["2d"]
    
    
    @property
    def get_static_properties(self) -> None:
        """
        This is where properties like insolation, sin(latitude), etc... should be calculated 
        """
        
        long_timeshift = (self.dataset["longitude"]) * np.pi/180 #radians of the phase time shift
        lat = self.dataset["latitude"] * np.pi/180 #convert to radians
        #long = (self.dataset["longitude"] + 180) * np.pi/180 #adjust to era coordinates and convert to radians
        long = self.dataset["longitude"] * np.pi/180 #convert to radians, apparently adjustment not needed

        #sin/cos of longs and lats. Note that by doing cos on the dataframe reference, the values are updated but not extracted in memory.
        self.dataset['cos_latitude'] = np.cos(lat)
        self.dataset['cos_longitude'] = np.cos(long)
        self.dataset['sin_latitude'] = np.sin(lat)
        self.dataset['sin_longitude'] = np.sin(long)

        #sin/cos of time and day
        # second_of_day = self.dates.astype(int)  #this has been converted to a datetime64 object.
        second_of_day = self.dataset["time"].astype(int)
        phase_of_day = (second_of_day * 2 * np.pi) / (24 * 60 * 60) #convert to radians of the day
        phase_of_year = (phase_of_day - np.pi/2)/365.25 #convert to radians of the year, phase shift by .25 days needed.

        #I now want to adjust phase_of_day to local time, which is done by adding the longitude timeshift
        #However, long_timeshift has shape (lat, long), while phase_of_day has shape (time), so I need to broadcast long_timeshift to the time dimension
        local_dims = ('time', 'y', 'x')
        phase_of_day = np.array([phase_of_day])
        local_phase_of_day = np.broadcast_to(phase_of_day[np.newaxis, :], np.array(long_timeshift)[np.newaxis, :].shape).copy()
        phase_of_year = np.array([phase_of_year])
        # phase_of_year = np.broadcast_to(phase_of_year[:, np.newaxis], np.array(long_timeshift).shape).copy()
        local_phase_of_day += np.array(long_timeshift)

        tdims = self.dataset["time"].dims
        # tdims = ("time",)
        # print(tdims)
        #Looking at the era data, it turns out that local time refers to the continous time of day with respect to the sun.
        self.dataset['cos_julian_day'] = (tdims, np.cos(phase_of_year)[0])
        self.dataset['cos_local_time'] = (local_dims, np.cos(local_phase_of_day))
        self.dataset['sin_julian_day'] = (tdims, np.sin(phase_of_year)[0])
        self.dataset['sin_local_time'] = (local_dims, np.sin(local_phase_of_day))
        
        #To calculate insolation, need to use sin local time, sin latitude, and sin julian day.
        #Looking through the era dataset insolation are values between 0 and 1, with 50% of values being 0.
        #indicating that this is the proportion of sun that is shining on the surface.
        #For this we need the local time of day, the yearly axial tilt, and the latitude of the location.
        jan_lat_shift_phase = 79*2*np.pi/365   #79 days from january 1st to the equinox on the 21st of march
        solar_latitude = 23.5 * np.pi/180 * np.sin(phase_of_year[:, np.newaxis, np.newaxis] - jan_lat_shift_phase) #yearly latitude shift of the sun in radians
        latitude_insolation = np.cos(lat.values.reshape(1, lat.shape[0], lat.shape[1]) - solar_latitude) #insolation due to latitude
        longitude_insolation = -np.cos(local_phase_of_day) #insolation due to local time, with peak at 12 hours
        latitude_insolation[latitude_insolation < 0] = 0    #no sun during winter
        longitude_insolation[longitude_insolation < 0] = 0 #no sun at night

        self.dataset['insolation'] = (local_dims, (latitude_insolation * longitude_insolation)[0])
        

        #Does cos(x+pi) = -cos(x) and sin(x+pi) = -sin(x)?
        #answer:
        
    
    @property
    def change_units(self) -> None:
        """
        Changes units and similar for required variables
        """
        from zarr_utils import w2omega

        #Compute the pressure velocity (omega) from the hydrostatic vertical velocity at the different pressure levels
        # wl = 'upward_air_velocity_pl'
        p_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        # w = np.zeros(self.dataset["w"].shape)
        # # print(self.dataset["w"].shape[1])

        # for t in range(self.dataset["w"].shape[0]):
        #     for h in range(self.dataset["w"].shape[1]):
        #         w_pl = self.dataset["w"][t, h, ...].values.flatten()
        #         t_pl = self.dataset['ta'][t, h, ...].values.flatten()
        #         p_pl = p_levels[h] * 100
        #         w[t, h, ...] = np.reshape(w2omega(w_pl, t_pl, p_pl), self.dataset["latitude"].shape)
        w = change_units_func(p_levels, self.dataset["w"], self.dataset["ta"], self.dataset["latitude"])
        # self.dataset["w"] = (self.dataset["w"].dims, w)
        self.dataset["w"] = w
        # self.dataset = self.dataset.chunk({})
        
    @property
    def fill_unimplemented(self) -> None:
        """
            Fills unimplemented variables in the dataset and also masked arrays.
        """
        #warnings.warn("There are still unimplemented variables in the dataset.")
        fill_dims = self.dataset['ps'].dims
        fill_shape = self.dataset['ps'].shape
        #setting orography to 0 because of fine grid, but note that copilot suggests #np.std(self.dataset['surface_geopotential'], axis=0)
        self.dataset['orography_stddev'] = (fill_dims, np.zeros(fill_shape))
        self.dataset['orography_slope'] = (fill_dims, np.zeros(fill_shape))
        self.dataset['tcw'] = (fill_dims, np.zeros(fill_shape))
        self.dataset['cp'] = (fill_dims, np.zeros(fill_shape))
        self.dataset['tp'] = (fill_dims, np.zeros(fill_shape))
        self.dataset['2d'] = (fill_dims, np.zeros(fill_shape))
        self.dataset['skt'] = (fill_dims, np.zeros(fill_shape))
        self.dataset['lsm'] = (fill_dims, np.zeros(fill_shape))
        self.dataset['z'] = (fill_dims, np.zeros(fill_shape))
        #Some of these values may be calculable later, but they are set to 0 for now to test the pipeline.

        #Masked values
        """
        sdvals = self.dataset['SFX_DSN_T_ISBA'].values.flatten()
        sdvals[np.isnan(sdvals)] = 0
        sdvals[sdvals > 10] = 10
        self.dataset['SFX_DSN_T_ISBA'] = (self.dataset['SFX_DSN_T_ISBA'].dims, sdvals.reshape(self.dataset['SFX_DSN_T_ISBA'].shape))
        """
        
    def concatenate_array(self, params: list, axis: int = 0, dtype = np.float32) -> np.ndarray:
        if isinstance(params, list):
            return np.concatenate(params, axis=axis, dtype=dtype)
        else:
            raise AttributeError(f"The provided params is not a sequence of arrays, i.e not a list of arrays. Got type {type(params)}")
    
    #@property
    def rotate_wind(self, xwind: str, ywind: str)-> None:
        """
            Rotates the grid (x,y) in x_wind_10m and y_wind_10m
            to the appropiate rotation. This method uses
            self._projection which is fetched from class data.
            The variables created are named as 10u and 10v

            args:
                None
            
            return:
                the calculated 10u and 10v
        
        """
        import pyproj
        
        from zarr_utils import rotate

        # Get projection parameters from MEPS
        crs = pyproj.CRS.from_cf(self._projection)

        proj4_str = crs.to_proj4()

        #x_comp = "x_wind_10m"
        #y_comp = "y_wind_10m"
        x_comp = xwind
        y_comp = ywind
        
        # Initialize u and v winself.selected_data
        
        x_wind = self.dataset[xwind]
        y_wind = self.dataset[ywind]

        u = np.zeros(x_wind.shape)#, np.float32)
        v = np.zeros(y_wind.shape)#, np.float32)

        for t in range(x_wind.shape[0]):
            for h in range(x_wind.shape[1]):
                x_wind = self.dataset[x_comp]
                x_wind = x_wind[t, h, ...].values.flatten()
                y_wind = self.dataset[y_comp]
                y_wind = y_wind[t, h, ...].values.flatten()
                lats = self.dataset["latitude"].values.flatten()
                lons = self.dataset["longitude"].values.flatten()
                u0, v0 = rotate(x_wind, y_wind, lats, lons, proj4_str)
                u[t, h, ...] = np.reshape(u0, self.dataset["latitude"].shape)
                v[t, h, ...] = np.reshape(v0, self.dataset["latitude"].shape)

        self.dataset[x_comp] = (self.dataset[x_comp].dims, u)
        self.dataset[y_comp] = (self.dataset[y_comp].dims, v)


    def rotate_wind_parallel(self, xwind: str, ywind: str) -> None:
        """
            Rotates the grid (x,y) in x_wind_10m and y_wind_10m
            to the appropiate rotation. This method uses
            self._projection which is fetched from class data.
            The variables created are named as 10u and 10v

            args:
                None
            
            return:
                the calculated 10u and 10v
        
        """
        import pyproj
        
        from zarr_utils import rotate

        # Get projection parameters from MEPS
        crs = pyproj.CRS.from_cf(self._projection)

        proj4_str = crs.to_proj4()

        x_comp = "uas"
        y_comp = "vas"
        # x_comp = xwind
        # y_comp = ywind
        
        # Initialize u and v winself.selected_data
        
        x_wind = self.dataset[xwind]
        y_wind = self.dataset[ywind]

        u = np.zeros(x_wind.shape)#, np.float32)
        v = np.zeros(y_wind.shape)#, np.float32)

        for t in range(x_wind.shape[0]):
            x_wind = self.dataset[x_comp]
            x_wind = x_wind[t, ...].values.flatten()
            y_wind = self.dataset[y_comp]
            y_wind = y_wind[t, ...].values.flatten()
            lats = self.dataset["latitude"].values.flatten()
            lons = self.dataset["longitude"].values.flatten()
            u0, v0 = rotate(x_wind, y_wind, lats, lons, proj4_str)
            u[t, ...] = np.reshape(u0, self.dataset["latitude"].shape)
            v[t, ...] = np.reshape(v0, self.dataset["latitude"].shape)
    # u, v = rotate_wind_func(x_wind, y_wind, self.dataset["latitude"], self.dataset["longitude"], proj4_str)
        # u, v = u.chunk({"time": 1,"pl":1, "y": 198,"x" : 198}), v.chunk({"time": 1,"pl":1, "y": 198,"x" : 198})
        
        # self.dataset[xwind] = (x_wind.dims, u.data)
        # self.dataset[ywind] = (y_wind.dims, v.data)
        self.dataset[x_comp] = (self.dataset[x_comp].dims, u)
        self.dataset[y_comp] = (self.dataset[y_comp].dims, v)


    def model_to_pressure_levels(self, config, target):
        '''
            Converts all pressure level variables (time, model level, x, y)
            from vertical model levels to vertical pressure levels (time, pressure level, x, y)
            This function also preprocesses the other variables to include the height dimension

            args:
                config: specify A and B coefficients and which variables are given at pressure levels
                target: numpy.ndarray with the target pressure levels in hPa
            
            return:
                updated dataset
        
        '''
        print("CONVERTING MODEL TO PRESSURE LEVELS")
        PVAB_path = config["pvab path"]
        variables = config["pl_var"]
        ps = self.dataset["ps"]
        PVAB = np.loadtxt(PVAB_path)
        for var_name in variables:
            print(f'converting {var_name} to pressure levels')
            var = self.dataset[var_name]
            for t in range(var.shape[0]):
                conv_par = mod_to_pres(ps, target, var, t, PVAB)
                conv_par = conv_par.expand_dims("time")
                conv_par = conv_par.transpose("time", "pl", "y", "x")
                self.dataset[var_name] = conv_par

        
        print("PREPROCESSING DOWA")
        for var_name in self.config["include"]:
            if var_name not in self.config["pl_var"]:
                if var_name in self.dataset.keys():
                    var = self.dataset[var_name]
                    self.dataset.update({var_name: var.expand_dims(dim={"height":1}, axis=1)})
    

    

    @property
    def total_column_water(self) -> None:
        '''
        Calculates total column water, which is the total mass of water vapor, cloud condensed water, cloud ice,
        graupel, rain and in a 1x1m column extending from the earth surface to toa. Calculation is done by integrating
        various quantities over all model levels and adding water vapor (allready integrated in MEPS output). 
        '''
        from zarr_utils import get_levels_effective_width

        tcwv = "lwe_thickness_of_atmosphere_mass_content_of_water_vapor"
        mass_frac_fields = ["mass_fraction_of_cloud_condensed_water_in_air_ml",
                            "mass_fraction_of_cloud_ice_in_air_ml",
                            "mass_fraction_of_graupel_in_air_ml",
                            "mass_fraction_of_rain_in_air_ml",
                            "mass_fraction_of_snow_in_air_ml"
                            ]
        af = self.dataset['ap']
        bf = self.dataset['b']

        tcw = np.zeros((self.dataset[tcwv].shape[0], 
                        self.dataset[tcwv].shape[2], 
                        self.dataset[tcwv].shape[3]))

        for t in range(self.dataset[tcwv].shape[0]):
            sp = self.dataset['surface_air_pressure'][t,0,:,:].data
            dp = get_levels_effective_width(af, bf, sp)
            tcw_temp = self.dataset[tcwv][t,0,:,:].data
            for field in mass_frac_fields:
                tcw_temp += np.sum(self.dataset[field][t,:,:,:]*dp, axis=0)
            tcw[t,:,:] = tcw_temp
        
        self.dataset['tcw'] = ((self.dataset[tcwv].dims[0],
                                self.dataset[tcwv].dims[2],
                                self.dataset[tcwv].dims[3]),
                                tcw)
