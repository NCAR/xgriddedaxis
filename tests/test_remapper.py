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
    ds_resample = (ds[['tmin']] * wgt).groupby(group).sum(dim='time')
    return ds_resample


@pytest.mark.parametrize(
    'time_units, calendar, decode_times, use_cftime, freq, binding', parameters,
)
def test_init_remapper(time_units, calendar, decode_times, use_cftime, freq, binding):
    ds = create_dataset(
        units=time_units, calendar=calendar, use_cftime=use_cftime, decode_times=decode_times,
    )

    remapper = Remapper(ds, freq=freq, binding=binding)
    assert isinstance(remapper.info, dict)
    assert remapper.outgoing['encoded_times'].attrs == ds.time.attrs
    assert isinstance(remapper.weights, sparse._coo.core.COO)


@pytest.mark.parametrize('use_cftime', [True, False])
def test_remapper_out_freq_warnings(use_cftime):
    ds = create_dataset(start='2018-01-01', end='2018-08-01', freq='MS', use_cftime=use_cftime)
    with pytest.warns(UserWarning):
        _ = Remapper(ds, freq='A')


def test_invalid_out_freq():
    with pytest.raises(ValueError):
        _validate_freq(freq='QM')


@pytest.mark.xfail(reason='Will be supported at a later time')
@pytest.mark.parametrize(
    'start, end, in_freq, out_freq, nlats, nlons, group',
    [
        ('2018-01-01', '2021-01-01', 'MS', 'A', 2, 2, 'time.year'),
        ('2018-01-01', '2018-02-01', 'D', 'M', 2, 2, 'time.month'),
        ('2018-01-01', '2019-01-01', 'D', 'M', 2, 2, 'time.month'),
        ('2018-01-01', '2018-01-08', '24H', 'D', 2, 2, 'time.day'),
    ],
)
def test_remapper_average(start, end, in_freq, out_freq, nlats, nlons, group):
    ds = create_dataset(
        start=start, end=end, freq=in_freq, nlats=nlats, nlons=nlons, var_const=False
    )
    remapper = Remapper(ds, freq=out_freq)
    results = remapper.average(ds.tmin).data
    expected = xarray_weighted_resample(ds, group).tmin.data
    np.testing.assert_almost_equal(expected, results, verbose=True)


@pytest.mark.xfail(reason='Will be supported at a later time')
@pytest.mark.parametrize(
    'start, end, in_freq, out_freq, nlats, nlons, group',
    [
        ('2018-01-01', '2021-01-01', 'MS', 'A', 2, 2, 'time.year'),
        ('2018-01-01', '2018-02-01', 'D', 'M', 2, 2, 'time.month'),
        ('2018-01-01', '2019-01-01', 'D', 'M', 2, 2, 'time.month'),
        ('2018-01-01', '2018-01-08', '24H', 'D', 2, 2, 'time.day'),
    ],
)
def test_remapper_average_w_transposed_dims(start, end, in_freq, out_freq, nlats, nlons, group):
    ds = create_dataset(
        start=start, end=end, freq=in_freq, nlats=nlats, nlons=nlons, var_const=False
    )
    ds = ds.transpose('lat', 'lon', 'd2', 'time', ...)
    remapper = Remapper(ds, freq=out_freq)
    results = remapper.average(ds.tmin).data
    expected = (
        xarray_weighted_resample(ds, group).transpose('lat', 'lon', 'd2', 'time', ...).tmin.data
    )
    np.testing.assert_almost_equal(expected, results)


@pytest.mark.xfail(reason='Will be supported at a later time')
def test_remapper_input_time_axis_mismatch():
    ds = create_dataset(start='2018-01-01', end='2018-01-07', freq='D')
    remapper = Remapper(ds, freq='7D')

    ds2 = create_dataset(start='2018-01-01', end='2018-01-08', freq='D')
    with pytest.raises(ValueError):
        _ = remapper.average(ds2.tmin)
