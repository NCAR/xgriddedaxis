from functools import reduce
from operator import mul

import numpy as np
import pandas as pd
import xarray as xr


def create_dataset(
    start='2018-01',
    end='2020-12',
    freq='MS',
    calendar='standard',
    units='days since 1980-01-01',
    use_cftime=True,
    decode_times=True,
    nlats=1,
    nlons=1,
    var_const=None,
):
    """ Utility function for creating test data """

    if use_cftime:
        offset = xr.coding.cftime_offsets.to_offset(freq)
        end = xr.coding.cftime_offsets.to_cftime_datetime(end, calendar=calendar)
        dates = xr.cftime_range(start=start, end=end + offset, freq=freq, calendar=calendar)

    else:
        offset = pd.tseries.frequencies.to_offset(freq)
        dates = pd.date_range(
            start=pd.to_datetime(start), end=pd.to_datetime(end) + offset, freq=freq
        )

    bnds = np.vstack((dates[:-1], dates[1:])).T
    bnds_encoded = xr.coding.times.encode_cf_datetime(bnds, units=units, calendar=calendar)[0]

    times = xr.DataArray(
        bnds_encoded.mean(axis=1),
        dims=('time'),
        name='time',
        attrs={'units': units, 'calendar': calendar},
    )

    if decode_times:
        times = xr.coding.times.decode_cf_datetime(
            times, units=units, calendar=calendar, use_cftime=use_cftime
        )
        time_bounds = xr.DataArray(
            bnds, name='time_bounds', dims=('time', 'd2'), coords={'time': times}
        )

    else:
        time_bounds = xr.DataArray(
            bnds_encoded, name='time_bounds', dims=('time', 'd2'), coords={'time': times}
        )

    lats = np.linspace(start=-90, stop=90, num=nlats, dtype='float32')
    lons = np.linspace(start=-180, stop=180, num=nlons, dtype='float32')

    shape = (times.size, lats.size, lons.size)
    num = reduce(mul, shape)

    if var_const is None:
        data = np.arange(1, num + 1, dtype='float32').reshape(shape)
    elif var_const:
        data = np.ones(shape=shape, dtype='float32')

    else:
        vals = np.linspace(250.0, 350.0, num=num)
        var_vals = np.sin(np.pi * vals) * np.exp(-0.1 * vals)
        data = var_vals.reshape(shape).astype('float32')

    var = xr.DataArray(
        data,
        name='var_ex',
        dims=['time', 'lat', 'lon'],
        coords={'time': times, 'lat': lats, 'lon': lons},
    )

    ds = var.to_dataset()

    ds = xr.merge((ds, time_bounds))
    ds.time.attrs['bounds'] = 'time_bounds'
    ds.time.attrs['units'] = units
    ds.time.attrs['calendar'] = calendar

    return ds
