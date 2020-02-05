import numpy as np
import numpy.matlib as npm
import pytest
import xarray as xr

from xtimeutil import Axis


def create_dataset(
    has_time_bounds=True,
    time_units='days since 0001-01-01',
    calendar='noleap',
    use_cftime=True,
    decode_times=True,
    nyrs=3,
    var_const=True,
):
    """return an example xarray.Dataset object, useful for testing functions"""

    # set up values for Dataset, 4 yrs of analytic monthly values
    days_1yr = np.array([31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0])
    time_edges = np.insert(np.cumsum(npm.repmat(days_1yr, nyrs, 1)), 0, 0)
    time_bounds_vals = np.stack((time_edges[:-1], time_edges[1:]), axis=1)
    time_vals = np.mean(time_bounds_vals, axis=1)
    time_vals_yr = time_vals / 365.0
    if var_const:
        var_vals = np.ones_like(time_vals_yr)
    else:
        var_vals = np.sin(np.pi * time_vals_yr) * np.exp(-0.1 * time_vals_yr)

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
    var = xr.DataArray(var_vals, name='var_ex', dims='time', coords={'time': time_var})
    ds = var.to_dataset()

    if has_time_bounds:
        ds.time.attrs['bounds'] = 'time_bounds'
        ds = xr.merge((ds, time_bounds))

    if decode_times:
        ds.time.data = xr.coding.times.decode_cf_datetime(
            ds.time, time_units, calendar, use_cftime=use_cftime
        )

        if has_time_bounds:
            ds.time_bounds.data = xr.coding.times.decode_cf_datetime(
                ds.time_bounds, time_units, calendar, use_cftime=use_cftime
            )

    return ds


@pytest.mark.parametrize(
    'has_time_bounds, time_units, calendar, use_cftime, decode_times',
    [
        (True, 'days since 0001-01-01', 'noleap', True, True),
        (True, 'days since 0001-01-01', 'noleap', True, False),
        (False, 'days since 0001-01-01', 'noleap', True, True),
        (True, 'hours since 1800-01-01', 'standard', False, True),
        (False, 'hours since 1800-01-01', 'standard', False, True),
        (True, 'hours since 1800-01-01', 'standard', False, False),
    ],
)
def test_init_axis(has_time_bounds, time_units, calendar, use_cftime, decode_times):
    ds = create_dataset(
        has_time_bounds=has_time_bounds,
        time_units=time_units,
        calendar=calendar,
        use_cftime=use_cftime,
        decode_times=decode_times,
    )
    axis = Axis(ds)
    assert axis.metadata['is_time_decoded'] == decode_times
    assert axis.decoded_times.shape == ds.time.shape


@pytest.mark.parametrize(
    'dataset, decode_times',
    [('rasm', False), ('rasm', True), ('air_temperature', False), ('air_temperature', True)],
)
def test_init_with_real_data(dataset, decode_times):
    ds = xr.tutorial.open_dataset(dataset, decode_times=decode_times)
    axis = Axis(ds)
    assert axis.metadata['is_time_decoded'] == decode_times
    assert axis.decoded_times.shape == ds.time.shape


def test_validate_time_coord():
    ds = create_dataset()
    with pytest.raises(ValueError):
        _ = Axis(ds, 'times')
