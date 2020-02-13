import itertools

import numpy as np
import pytest
import sparse
import xarray as xr

from xtimeutil import Remapper
from xtimeutil.remapper import _validate_freq
from xtimeutil.testing import create_dataset

freqs = (
    '12000S',
    '200T',
    'H',
    '23H',
    'D',
    '85D',
    '7M',
    'Q',
    '11Q-JUN',
    'A',
    '9YS',
)

# cftime config
time_units = ('days since 1800-01-01',)
calendars = ('noleap',)
decode_times = (False, True)
inputs1 = [time_units, calendars, decode_times, (True,), freqs, ('middle', 'right', 'left')]

# pandas datetime config
inputs2 = [
    ('hours since 1800-01-01',),
    ('standard',),
    (True, False),
    (False,),
    freqs,
    ('middle', 'right', 'left'),
]

combs1 = [element for element in itertools.product(*inputs1)]
combs2 = [element for element in itertools.product(*inputs2)]

parameters = combs1 + combs2


def xarray_weighted_resample(ds, group):
    wgt = ds.time_bounds.diff('d2').squeeze()
    wgt_grouped = wgt.groupby(group)
    wgt = wgt_grouped / wgt_grouped.sum(dim=xr.ALL_DIMS)
    periods = len(wgt_grouped.groups)
    np.testing.assert_almost_equal(wgt.groupby(group).sum(dim=xr.ALL_DIMS), np.ones(periods))
    ds_resample = (ds[['var_ex']] * wgt).groupby(group).sum(dim='time')
    return ds_resample


@pytest.mark.parametrize(
    'time_units, calendar, decode_times, use_cftime, freq, binding', parameters,
)
def test_init_remapper(time_units, calendar, decode_times, use_cftime, freq, binding):
    ds = create_dataset(
        units=time_units, calendar=calendar, use_cftime=use_cftime, decode_times=decode_times,
    )

    remapper = Remapper(ds, freq=freq, binding=binding)
    assert isinstance(remapper.info, xr.Dataset)
    assert {'outgoing_time_bounds', 'weights'}.issubset(set(remapper.info.variables))
    assert isinstance(remapper.info.weights.data, sparse._coo.core.COO)
    assert set(
        [
            'freq',
            'binding',
            'is_time_decoded',
            'time_coord_name',
            'units',
            'calendar',
            'time_bounds_varname',
            'time_bounds_dim',
            'time_bounds_dim_axis_num',
            'use_cftime',
        ]
    ) == set(remapper.info.attrs.keys())


@pytest.mark.parametrize('use_cftime', [True, False])
def test_remapper_out_freq_warnings(use_cftime):
    ds = create_dataset(start='2018-01-01', end='2018-08-01', freq='MS', use_cftime=use_cftime)
    with pytest.warns(UserWarning):
        _ = Remapper(ds, freq='A')


def test_invalid_out_freq():
    with pytest.raises(ValueError):
        _validate_freq(freq='QM')


@pytest.mark.parametrize(
    'start, end, in_freq, out_freq, nlats, nlons, group',
    [
        ('2018-01-01', '2020-12-31', 'MS', 'A', 2, 2, 'time.year'),
        ('2018-01-01', '2018-01-31', 'D', 'M', 2, 2, 'time.month'),
        ('2018-01-01', '2018-12-31', 'D', 'M', 2, 2, 'time.month'),
        ('2018-01-01', '2018-01-07', '24H', 'D', 2, 2, 'time.day'),
    ],
)
def test_remapper_average(start, end, in_freq, out_freq, nlats, nlons, group):
    ds = create_dataset(start=start, end=end, freq=in_freq, nlats=nlats, nlons=nlons)
    remapper = Remapper(ds, freq=out_freq)
    results = remapper.average(ds.var_ex).data
    expected = xarray_weighted_resample(ds, group).var_ex.data
    np.testing.assert_array_equal(expected, results)


def test_remapper_input_time_axis_mismatch():
    ds = create_dataset(start='2018-01-01', end='2018-01-07', freq='D')
    remapper = Remapper(ds, freq='7D')

    ds2 = create_dataset(start='2018-01-01', end='2018-01-08', freq='D')
    with pytest.raises(ValueError):
        _ = remapper.average(ds2.var_ex)


@pytest.mark.parametrize(
    'start, end, in_freq, out_freq, nlats, nlons, group',
    [
        ('2018-01-01', '2020-12-31', 'MS', 'A', 2, 2, 'time.year'),
        ('2018-01-01', '2018-01-31', 'D', 'M', 2, 2, 'time.month'),
        ('2018-01-01', '2018-12-31', 'D', 'M', 2, 2, 'time.month'),
        ('2018-01-01', '2018-01-07', '24H', 'D', 2, 2, 'time.day'),
    ],
)
def test_remapper_average_w_transposed_dims(start, end, in_freq, out_freq, nlats, nlons, group):
    ds = create_dataset(start=start, end=end, freq=in_freq, nlats=nlats, nlons=nlons)
    ds = ds.transpose('lat', 'lon', 'd2', 'time', ...)
    remapper = Remapper(ds, freq=out_freq)
    results = remapper.average(ds.var_ex).data
    expected = (
        xarray_weighted_resample(ds, group).transpose('lat', 'lon', 'd2', 'time', ...).var_ex.data
    )
    np.testing.assert_array_equal(expected, results)
