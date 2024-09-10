import numpy as np
from tozarr import ToZarr

tz = ToZarr()
root = tz.initialise_dataset_backend

#FIX LONGS
new_longs = np.array(root['longitudes'])# - 180
new_longs[new_longs < 0] = new_longs[new_longs < 0] + 360
tz.add(store = root, name = "longitudes", data = new_longs, ds_shape = new_longs.shape, append = False)
