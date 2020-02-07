from functools import reduce
from operator import mul

import numpy as np
import numpy.matlib as npm
import xarray as xr


def create_dataset(
    has_time_bounds=True,
    time_units='days since 1980-01-01',
    calendar='noleap',
    use_cftime=True,
    decode_times=True,
    nyrs=3,
    var_const=True,
):
    """return an example xarray.Dataset object, useful for testing functions"""

    # set up values for Dataset, 4 yrs of analytic monthly values
    days_1yr = np.array(
        [31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0], dtype='float32'
    )
    time_edges = np.insert(np.cumsum(npm.repmat(days_1yr, nyrs, 1)), 0, 0)
    time_bounds_vals = np.stack((time_edges[:-1], time_edges[1:]), axis=1)
    time_vals = np.mean(time_bounds_vals, axis=1)
    time_vals_yr = time_vals / 365.0
    lats = np.array([20.0, 30.0], dtype='float32')
    lons = np.array([100.0, 120.0], dtype='float32')

    shape = (time_vals_yr.size, lats.size, lons.size)
    if var_const:
        var_vals = np.ones(shape=shape, dtype='float32')
    else:
        num = reduce(mul, shape)
        vals = np.linspace(time_vals_yr.min(), time_vals_yr.max(), num=num)
        var_vals = np.sin(np.pi * vals) * np.exp(-0.1 * vals)
        var_vals = var_vals.reshape(shape).astype('float32')

    # create Dataset, including time_bounds
    time_var = xr.DataArray(
        time_vals,
        name='time',
        dims='time',
        coords={'time': time_vals},
        attrs={'units': time_units, 'calendar': calendar},
    )

    time_bounds = xr.DataArray(
        time_bounds_vals, name='time_bounds', dims=('time', 'd2'), coords={'time': time_var}
    )
    var = xr.DataArray(
        var_vals,
        name='var_ex',
        dims=['time', 'lat', 'lon'],
        coords={'time': time_var, 'lat': lats, 'lon': lons},
    )
    ds = var.to_dataset()

    if has_time_bounds:
        ds.time.attrs['bounds'] = 'time_bounds'
        ds = xr.merge((ds, time_bounds))

    if decode_times:
        ds = xr.decode_cf(ds, use_cftime=use_cftime)

    return ds
