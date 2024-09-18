from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from numba import jit 
import xarray as xr


######################################### HDF5 related (not used, may be useful later) ###########################################

class HdfFileManager:
    def __init__(self, path: Union[str, Path], if_file_not_found: Optional[dict]) -> None:
        self.path = Path(path)

        if if_file_not_found:
            assert isinstance(if_file_not_found, dict), f"if_file_not_found is not a dict, got {type(if_file_not_found)}"
            self.if_file_not_found = if_file_not_found

    def read(self):
        NotImplementedError 

    def write(self):
        NotImplementedError        



####################################### JSON related  (not used anymore, may be useful later)) #####################################
import json 

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class JsonFileManager:
    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)


    def read(self, if_file_not_found: Optional[dict] = None, convert: str = None) -> Any:
        try:
            with open(file=self.path, mode = "r") as file: 
                data = json.load(file)
            
            if convert: self.__convert(data=data, convertType=convert)
            return data
        
        except FileNotFoundError:
            if if_file_not_found:
                assert isinstance(if_file_not_found, dict), f"The variable if_file_not_found is not a dict. Got type{type(if_file_not_found)}"
            else:
                raise FileNotFoundError
            
            self.write(data=if_file_not_found)
            return if_file_not_found
        
        except Exception as e:
            print(f"Exception raised: {e}")
            return None


    def write(self, data: dict) -> None:
        assert isinstance(data,dict), f"Data is not a dict. Got {type(data)}"

        dumped_data = json.loads(json.dumps(data, cls = NumpyEncoder))
        try:
            with open(file=self.path, mode="w") as file:
                json.dump(dumped_data, file)

        except Exception as e:
            print(f"Error writing dict to file. Exception raised: {e}")
    
    def __convert(self, data, convertType: str):
        if convertType == "numpy":
            for k,v in data.items():
                if isinstance(v, list):
                    data[k] = np.array(v, dtype=np.float32)
        else:
            raise NotImplementedError
        

def jsonHandler(
        path: Union[Path, str], 
        mode: str,  
        data: Optional[dict] = None,
        convertTo: str = None,
        **kwargs: Any
        ) -> None:
    
    """
        Still under development

    """
    path = Path(path)
    modes = ["a", "w", "r"]

    if mode not in modes:
        raise ValueError(f"Given mode is not supported. Supporting a/w/r. Got {mode}")
    
    jsonfilemanager = JsonFileManager(path=path)
    if mode == "r":
        if kwargs["if_file_not_found"]:
            return jsonfilemanager.read(kwargs["if_file_not_found"], convert = convertTo)
        return jsonfilemanager.read(convert = convertTo)
    elif mode == "w":
        assert isinstance(data, dict),f"Data is not provided or not correct type, got {type(data)}"
        jsonfilemanager.write(
                data = data
            )
    else:
        raise NotImplementedError
        



######################################################## calculation utils used in core ##########################################################S########
import pyproj
# import gridpp

@jit(nopython=True)
def w2omega(w, T, p):

    """ Computes the pressure velocity (omega) from the hydrostatic vertical velocity
    for a given temperature (T) and pressure (p).
    
    
    Parameters
    
            w (number or ndarray) - hydrostatic pressure velocity (m/s)
    
            t (number or ndarray) - temperature (K)
    
            p (number or ndarray) - pressure (Pa)
    
    Return same type as w

            omega (number, ndarray or Fieldset) – hydrostatic pressure velocity (Pa/s)

    
    """
    Rd = 287.058  # specific gas constant for dry air 
    g = 9.81  # gravitational acceleration
    
    omega = -w*g*p / (T*Rd)

    return omega
def change_units_func(p_levels, w, ta, lat):
        return xr.apply_ufunc(change_units_numba,
                       w,
                       ta,
                       lat,
                       input_core_dims=[['time',"pl", 'y', 'x'], ['time', 'pl', 'y', 'x'], ["y", "x"]],  # list with one entry per arg
                       output_core_dims=[['time',"pl", 'y', 'x']],  
                       vectorize=True,
                       dask='parallelized',
                       dask_gufunc_kwargs = {"allow_rechunk":True},
                       output_dtypes=[w.dtype],  # one per output
                       kwargs={'plevels': p_levels},
                       )
@jit(nopython=True)
def change_units_numba(w, ta, lat, plevels):
    w_new = np.zeros(w.shape)
    # print(self.dataset["w"].shape[1])

    for t in range(w.shape[0]):
        for h in range(w.shape[1]):
            w_pl = w[t, h, ...].flatten()
            t_pl = ta[t, h, ...].flatten()
            p_pl = plevels[h] * 100
            w_new[t, h, ...] = np.reshape(w2omega(w_pl, t_pl, p_pl), lat.shape)
        return w_new

@jit(nopython=True)
def data_to_zarr_numba(var_val):
    var_dims = var_val.dims
    if len(var_dims) == 4 and set(('time', 'y', 'x')).issubset(var_dims):
        dim1len = self[varname].shape[1]
        var_val = var_val.reshape((max_shape[0], dim1len, max_shape[2]*max_shape[3]))  #these parameters have the desired shape, with the second axis being either 1 or 13 in length
    else:
        if var_dims == ('time', 'y', 'x'):
            var_val = self[varname].values
            # var_val = var_val[:,np.newaxis, ...].reshape((max_shape[0], 1, max_shape[2]*max_shape[3]))
            var_val = var_val[:,np.newaxis, ...].reshape((max_shape[0], 1, max_shape[2]*max_shape[3]))
        elif 'time' in var_dims:
            var_val = self[varname].values
            # var_val = var_val[:,np.newaxis, np.newaxis].repeat(max_shape[2] * max_shape[3], axis=2)
            var_val = var_val[:,np.newaxis, np.newaxis].repeat(max_shape[2] * max_shape[3], axis=2)
        else: #coordinates y x are alone
            var_val = self[varname].values.flatten() #flatten lat long
            var_val = var_val[np.newaxis, np.newaxis, ...].repeat(max_shape[0], axis = 0) 


def calculate_dewpoint(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
        Calculates the dewpoint by using 
        relative_humidity and air_temperature. Notice that function
        uses gridpp to correctly calculate the dewpoint. For more 
        information about gridpp, check: https://github.com/metno/gridpp
        
        args:
            arr1 (np.ndarray): eitehr relative_humidity or air_temperature array
            arr2 (np.ndarray): eitehr relative_humidity or air_temperature array
        
        return:
            the dewpoint reshaped to one of the arrays shape.

    """
    import gridpp
    shape = arr1.shape
    #dewpoint = gridpp.dewpoint(arr1.flatten(), arr2.flatten())
    res_dewpoint = gridpp.dewpoint(arr1.flatten(), arr2.flatten())
    return np.reshape(res_dewpoint, shape)


def calculate_relative_humidity(arr1: np.ndarray, arr2: np.ndarray, arr3: np.ndarray) -> np.ndarray:
        """
            Calculates the relative humidity by using air temperature
            specific humidity, pressure.
            
            args:
                arr1 (np.ndarray): air temperature array
                arr2 (np.ndarray): specific humidity array
                arr3 (np.ndarray): pressure array
            
            return:
                the relative humidity reshaped to one of the arrays shape.

        """
        shape = arr1.shape
        tmp_array_es = np.zeros_like(arr1)
        tmp_array_SHS = np.zeros_like(arr1)
        res_array_RH = np.zeros_like(arr1)
        
        #1. Compute Saturation Vapor Pressure (for ps in Pa, tas in K, see. Lawrence 2004)
        tmp_array_es=610.94*np.exp(17.625*(arr1-273.15)/(arr1-273.15+243.04))
        #2. Compute Specific Humidity at Saturation
        tmp_array_SHS=(0.622*tmp_array_es)/(arr3-(0.378*tmp_array_es))
        #3. Compute relative humidity
        res_array_RH=(arr2/tmp_array_SHS)*100
        return res_array_RH


def rotate(x_wind, y_wind, lats, lons, proj_str_from, proj_str_to="proj+=longlat"):
    """Rotates winds in projection to another projection. Based loosely on
    https://github.com/SciTools/cartopy/blob/main/lib/cartopy/crs.py#L429

    Args:
        x_wind (np.array or list): Array of winds in x-direction [m]
        y_wind (np.array or list): Array of winds in y-direction [m]
        lats (np.array or list): Array of latitudes [degrees]
        lons (np.array or list): Array of longitudes [degrees]
        proj_str_from (str): Projection string to convert from
        proj_str_to (str): Projection string to convert to

    Returns:
        new_x_wind (np.array): Array of winds in x-direction in new projection [m]
        new_y_wind (np.array): Array of winds in y-direction in new projection [m]

    Todo:
        Deal with perturbations that end up outside the domain of the transformation
        Deal with any issues related to directions on the poles
    """

    if np.shape(x_wind) != np.shape(y_wind):
        raise ValueError(
            f"x_wind {np.shape(x_wind)} and y_wind {np.shape(y_wind)} arrays must be the same size"
        )
    if len(lats.shape) != 1:
        raise ValueError(f"lats {np.shape(lats)} must be 1D")

    if np.shape(lats) != np.shape(lons):
        raise ValueError(
            f"lats {np.shape(lats)} and lats {np.shape(lons)} must be the same size"
        )

    if len(np.shape(x_wind)) == 1:
        if np.shape(x_wind) != np.shape(lats):
            raise ValueError(
                f"x_wind {len(x_wind)} and lats {len(lats)} arrays must be the same size"
            )
    elif len(np.shape(x_wind)) == 2:
        if x_wind.shape[1] != len(lats):
            raise ValueError(
                f"Second dimension of x_wind {x_wind.shape[1]} must equal number of lats {len(lats)}"
            )
    else:
        raise ValueError(f"x_wind {np.shape(x_wind)} must be 1D or 2D")

    proj_from = pyproj.Proj(proj_str_from)
    proj_to = pyproj.Proj(proj_str_to)

    # Using a transformer is the correct way to do it in pyproj >= 2.2.0
    transformer = pyproj.transformer.Transformer.from_proj(proj_from, proj_to)

    # To compute the new vector components:
    # 1) perturb each position in the direction of the winds
    # 2) convert the perturbed positions into the new coordinate system
    # 3) measure the new x/y components.
    #
    # A complication occurs when using the longlat "projections", since this is not a cartesian grid
    # (i.e. distances in each direction is not consistent), we need to deal with the fact that the
    # width of a longitude varies with latitude
    orig_speed = np.sqrt(x_wind**2 + y_wind**2)

    x0, y0 = proj_from(lons, lats)
    if proj_from.name != "longlat":
        x1 = x0 + x_wind
        y1 = y0 + y_wind
    else:
        # Reduce the perturbation, since x_wind and y_wind are in meters, which would create
        # large perturbations in lat, lon. Also, deal with the fact that the width of longitude
        # varies with latitude.
        factor = 3600000.0
        x1 = x0 + x_wind / factor / np.cos(lats * 3.14159265 / 180)
        y1 = y0 + y_wind / factor

    X0, Y0 = transformer.transform(x0, y0)
    X1, Y1 = transformer.transform(x1, y1)

    new_x_wind = X1 - X0
    new_y_wind = Y1 - Y0
    if proj_to.name == "longlat":
        new_x_wind *= np.cos(lats * 3.14159265 / 180)

    if proj_to.name == "longlat" or proj_from.name == "longlat":
        # Ensure the wind speed is not changed (which might not the case since the units in longlat
        # is degrees, not meters)
        curr_speed = np.sqrt(new_x_wind**2 + new_y_wind**2)
        new_x_wind *= orig_speed / curr_speed
        new_y_wind *= orig_speed / curr_speed

    return new_x_wind, new_y_wind

def get_half_level_coefficients(af, bf):
    '''
    Calculate half-level coefficients ah and bh from the coefficients a and b for model levels. The half 
    level coefficients can be used to calculate the pressure at the interface between two model levels 
    P = ah + bh*ps (where ps is surface pressure).
    Based on implementation in 'module_integration.f90' in metcoop-harmonie (/util/gl/mod)
    
    Args:
        af (np.ndarray): 1d array of length #(model levels)
        bf (np.ndarray): 1d array of length #(model levels)

    Returns:
        ah (np.ndarray): 1d array of length #(model levels) + 1 
        bh (np.ndarray): 1d array of length #(model levels) + 1
    '''
    nlevels = len(af)
    assert len(af) == len(bf)
    ah, bh = np.zeros(nlevels+1), np.zeros(nlevels+1)

    ah[nlevels] = 0.
    bh[nlevels] = 1.

    for level in range(nlevels-1, 0, -1):
        ah[level] = 2*af[level] - ah[level+1]
        bh[level] = 2*bf[level] - bh[level+1]

    ah[0] = af[0]/np.sqrt(2)
    bh[0] = 0

    return ah, bh

def get_levels_effective_width(af, bf, surface_pressure):
    '''
    Calculate the "effective level width" (height [m] of level * density) of each model level. This can be used to integrate
    quantities over all model levels when quantities are given in fraction_of_mass_per_model_level.
    Based on implementation in 'get_column_integral.f90' in metcoop-harmonie (/util/gl/grb)

    Args:
        af (np.ndarray): 1d array of length #(model levels)
        bf (np.ndarray): 1d array of length #(model levels)
        surface_pressure (np.ndarray): 2d array with surface air pressure for each model grid point

    Returns:
        dp (np.ndarray): 3d array with shape ( len(af), surface_pressure.shape ) with effective level width for each
        model level / grid point. 
    '''
    g = 9.80665 # Taken from metcoop-harmonie
    nlevels = len(af)

    ah, bh = get_half_level_coefficients(af, bf)
    da = ah[1:nlevels+1] - ah[0:nlevels]
    db = bh[1:nlevels+1] - bh[0:nlevels]
    dp = (da[:, np.newaxis, np.newaxis] + surface_pressure*db[:, np.newaxis, np.newaxis]) / g

    return dp


@jit(nopython=True)
def model_to_pressure_levels_numba(ps, target_pressure_levels, par, t, PVAB):
    '''
    Interpolates model levels to pressure levels using for loops
    for HARMONIE-AROME netcdf output (as in DOWA), given in numpy arrays
    Interpolation per time step! (Otherwise it gets too slow)
    
    args:
    ps (np.ndarray):
        pressure level at ground level (surface air pressure), shape (time, n_model_levels, x, y)
    target_pressure_levels (np.ndarray): 
        pressure levels to interpolate to, shape (n_pres_levels,)
    par (np.ndarray): 
        data to be interpolated, shape (time, n_model_levels, x, y)
    pvab_path (str): 
        path of model level coëfficients A and B (.txt) 
    t (int):
        time step of interpolation

    return:
    parout (np.ndarray): interpolated data at target pressure levels, shape (x, y, n_model_levels)
    '''
    
    par_shape = par.shape
    n_model_levels = par_shape[1]
    n_pres_levels = len(target_pressure_levels)
    
    VBH = PVAB[:,0]
    VAH = PVAB[:,1]
    
    VAF = [0.5*(VAH[n] + VAH[n+1]) for n in range(n_model_levels)]
    VBF = [0.5*(VBH[n] + VBH[n+1]) for n in range(n_model_levels)]
    parout = np.zeros((par_shape[2], par_shape[3], n_pres_levels))
    for j in range(par_shape[2]):
        for k in range(par_shape[3]):
            pressures = np.array([VAF[l] + VBF[l] * ps[t, j, k] for l in range(len(VAF))])
            array = pressures/100
            for p, pl in enumerate(target_pressure_levels):
                idx = np.searchsorted(array, pl, side="left")
                if idx == 0:
                    alpha = (pl - array[0]) / (array[20] - array[0])
                    #print(f"Extrapolating left: pl={pl}, alpha={alpha}, par0={par[t, 0, j, k]}, par1={par[t, 1, j, k]}")
                    parout[j, k, p] = 0#par[t, 0, j, k] + alpha * (par[t, 20, j, k] - par[t, 0, j, k])
                elif idx == len(array):
                    alpha = (pl - array[-20]) / (array[-1] - array[-20])
                    #print(f"Extrapolating right: pl={pl}, alpha={alpha}, parN-2={par[t, -2, j, k]}, parN-1={par[t, -1, j, k]}")
                    parout[j, k, p] = 0#par[t, -20, j, k] + alpha * (par[t, -1, j, k] - par[t, -20, j, k])
                else:
                        alpha = (pl - array[idx-1]) / (array[idx] - array[idx-1])
                        #print(f"Interpolating: pl={pl}, alpha={alpha}, parIdx-1={par[t, idx-1, j, k]}, parIdx={par[t, idx, j, k]}")
                        parout[j, k, p] = par[t, idx-1, j, k] + alpha * (par[t, idx, j, k] - par[t, idx-1, j, k])
    
    return parout

def mod_to_pres(ps, target, dowa, t, PVAB):
    return xr.apply_ufunc(model_to_pressure_levels_numba, 
                   ps, 
                   target,
                   dowa, 
                   input_core_dims=[['time', 'x', 'y'], ["pl"], ['time', 'lev', 'x', 'y']],  # list with one entry per arg
                   output_core_dims=[['x', 'y', 'pl']],  # returned data has one dimension less
                #    exclude_dims=set(("lev",)),
                   vectorize=True,
                   dask='parallelized',
                   dask_gufunc_kwargs = {"allow_rechunk":True},
                   output_dtypes=[dowa.dtype],  # one per output
                   kwargs={'t': t, 'PVAB': PVAB },
    )

def rotate_wind_func(xwind, ywind, lats, lons, proj):
        return xr.apply_ufunc(rotate_wind_internal,
                       xwind,
                       ywind,
                       lats,
                       lons,
                       input_core_dims=[['time',"pl", 'y', 'x'], ['time', 'pl', 'y', 'x'], ["y", "x"], ["y", "x"]],  # list with one entry per arg
                       output_core_dims=[['time',"pl", 'y', 'x'], ['time',"pl", 'y', 'x']],  # returned data has one dimension less
                       vectorize=True,
                       dask='parallelized',
                       dask_gufunc_kwargs = {"allow_rechunk":True},
                       output_dtypes=[xwind.dtype, ywind.dtype],  # one per output
                       kwargs={'proj': proj},
                       )

# @jit(nopython=True)
def rotate_wind_internal(x_wind, y_wind, lats, lons, proj):
    u = np.zeros(x_wind.shape)#, np.float32)
    v = np.zeros(y_wind.shape)#, np.float32)
    for t in range(x_wind.shape[0]):
        for h in range(x_wind.shape[1]):
            x_wind = x_wind[t][h]
            x_wind = x_wind.flatten()
            y_wind = y_wind[t][h]
            y_wind = y_wind.flatten()
            f_lat = lats.flatten()
            f_lon = lons.flatten()
            u0, v0 = rotate(x_wind, y_wind, f_lat, f_lon, proj)
            u[t, h, ...] = np.reshape(u0, lats.shape)
            v[t, h, ...] = np.reshape(v0, lons.shape)

################################################### Zarr utils #########################################################

import zarr 

def add_dataset(
    *,
    name: str, 
    store_root: zarr.hierarchy.Group,
    data: Union[dict, np.ndarray], 
    chunks: Union[dict, int] = None,
    ds_dtype: Any = None, 
    ds_shape: tuple = None,
    add_method: str = "append",
    **kwargs: Any
    ) -> zarr.hierarchy.Group:

    """
        Adds a dataset into zarr archive, how the archive looks depends 
        on the shape of ds_shape and chunks. For example for a given 
        dataset with dimension (6,85,1001461), and you wish the chunking to be 
        0.0 1.0 2.0, etc.. for each time dimension then it is wise to have 
        shape = ds_shape and chunks = tuple([1] + list(data.shape[1:]).

        args:
            name (str): name of the folder you wish to save under
            root (zarr.hierarchy.Group): The root store of the zarr archive
            data (np.ndarray) : numpy data array which contains the dataset
            chunks (int, tuple): how the data is chunked in the archive
            ds_dtype (Any): The datatype to be used when saving
            ds_shape (tuple) : The shape of the dataset
            **kwargs (Any) : Additional keywords
        
        return:
            Stores the dataset in the archive, and returns root of the store.
    """
    import errors  

    if isinstance(data, dict):
        for n, value in data.items():
            value = value[None,...]  #adds time dimension.
            store_root = add_dataset(
                name = n,
                store_root = store_root,
                data = value,
                chunks = chunks,
                ds_dtype = ds_dtype,
                ds_shape = ds_shape,
                add_method = add_method
            )
        return store_root
    
    if ds_dtype is None:
        assert data is not None, (name, ds_shape, data, ds_dtype, store_root)
        ds_dtype = data.dtype
    
    if ds_shape is None:
        ds_shape = data.shape

    if data is not None:
        assert data.shape == ds_shape, (data.shape, ds_shape)

        arr = zarr.array(
            data= data,
            chunks = chunks,
            dtype = ds_dtype,
        )
        try:
            if add_method == "append":
                store_root[name].append(data = arr, axis=0)

            elif add_method == "add_to_array":
                #there's probably a zarr native way to do this, but for now this will do.
                vals = store_root[name]
                vals = np.array(vals, dtype = ds_dtype)
                #Assumes that the first dimension is time, works for all 1D->nD arrays
                new_vals = np.zeros((vals.shape[0] + data.shape[0], *data.shape[1:]), dtype=ds_dtype)
                new_vals[:vals.shape[0]] = vals
                new_vals[vals.shape[0]:] = data

                arr = zarr.array(data=new_vals, chunks = chunks, dtype = ds_dtype)
            
                store_root[name] = arr #i couldn't really find a way to test if an initialized variable but not array existed without appending so for now, all non appends will be overwritten each time
            elif add_method == "overwrite":
                store_root[name] = arr
        except (AttributeError, KeyError):
            # if data array does not exist within the group, key error will occur within _build.
            store_root[name] = arr

        #how do I test if store_root[name] exists without using try/except?
        #Answer: use if name in store_root

        return store_root
    
    else:
        raise errors.ArrayNotFoundError(f"Data array not found or not given. Got {data}")

def add_data(
        *,
        store_root: zarr.hierarchy.Group,
        data: Union[dict, np.ndarray],
        chunks: Union[dict, int] = None,
        d_dtype: Any = None,
        **kwargs: Any
    ) -> zarr.hierarchy.Group:
    """
        Adds a data/values into zarr archive, how the archive looks depends 
        on chunks. 

        args:
            store_root (zarr.hierarchy.Group): The root store of the zarr archive
            data (np.ndarray) : numpy data array which contains the dataset
            chunks (int, tuple): how the data is chunked in the archive
            d_dtype (Any): The datatype to be used when saving
            **kwargs (Any) : Additional keywords
        
        return:
            Stores the data in the archive, and returns root of the store.
    """
    assert isinstance(store_root, zarr.hierarchy.Group), f"The store root is not zarr.hierarchy.Group. Got {store_root}"
    
    if chunks:
        assert isinstance(chunks, (int, tuple)), f"Chunks is not int or tuple, got {type(chunks)}"

    if data is not None:
        assert isinstance(data, (int,float, dict))
    
        if isinstance(data, dict):
            for name, value in data.items():
                value = np.array(value, dtype=d_dtype)[None,...]

                
                arr = zarr.array(
                 data=value,
                 chunks = value.shape if chunks is None else chunks,
                 dtype = d_dtype
                )

                try:
                    # if data array group exist, append to existing group
                    store_root[name].append(data = value, axis=0) 
                
                
                except KeyError:
                    # if data array does not exist within the group
                    store_root[name] = arr
            
            return store_root
        
        else:
            data = np.array(data, dtype=d_dtype)[None,...]
            arr = zarr.array(
                 data=np.array(data, dtype=d_dtype),
                 chunks = data.shape,
                 dtype = d_dtype
            )

            try:
                # if data array group exist, append to existing group
                store_root[kwargs["name"]].append(data = arr, axis=0) 
            except AttributeError:
                # if data array does not exist within the group
                store_root[kwargs["name"]] = arr

            return store_root
    else:
        raise ValueError(f"The given data is not found or None")
        
