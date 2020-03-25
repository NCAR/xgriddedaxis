import itertools

import numpy as np
import pytest

from xtimeutil.axis import _infer_time_data_tick_binding, get_time_axis_info
from xtimeutil.testing import create_dataset

# cftime config
time_units = ('days since 1800-01-01', 'hours since 1800-01-01')
calendars = ('standard', 'noleap', '360_day', 'all_leap', '365_day', 'proleptic_gregorian')
decode_times = (False, True)
inputs1 = [time_units, calendars, decode_times, (True,)]

# pandas datetime config
inputs2 = [('days since 1800-01-01',), ('standard',), (True, False), (False,)]

combs1 = [element for element in itertools.product(*inputs1)]
combs2 = [element for element in itertools.product(*inputs2)]

parameters = combs1 + combs2


@pytest.mark.parametrize(
    'time_units, calendar, decode_times, use_cftime', parameters,
)
def test_init_axis(time_units, calendar, decode_times, use_cftime):
    ds = create_dataset(
        units=time_units, calendar=calendar, use_cftime=use_cftime, decode_times=decode_times,
    )
    info = get_time_axis_info(ds)
    assert info['is_time_decoded'] == decode_times
    assert info['decoded_times'].shape == ds.time.shape


def test_init_axis_with_attrs_from_encoding():
    ds = create_dataset()
    ds.time.encoding = ds.time.attrs
    ds.time.attrs = {}
    info = get_time_axis_info(ds)
    assert isinstance(info, dict)


def test_init_missing_bounds():
    ds = create_dataset()
    ds = ds.drop_vars(['time_bounds'])
    del ds.time.attrs['bounds']
    with pytest.raises(RuntimeError):
        _ = get_time_axis_info(ds)


def test_validate_time_coord():
    ds = create_dataset(var_const=False)
    with pytest.raises(KeyError):
        _ = get_time_axis_info(ds, 'times')


def test_invalid_binding():
    ds = create_dataset(var_const=True)
    with pytest.raises(KeyError):
        _ = get_time_axis_info(ds, 'time', 'center')


def test_infer_time_data_tick_binding():
    ds = create_dataset(decode_times=False)
    sample_time_bound = ds.time_bounds[0]
    sample_time_data_tick = ds.time[0]

    assert 'middle' == _infer_time_data_tick_binding(
        sample_time_bound.data, sample_time_data_tick.data
    )
    assert 'left' == _infer_time_data_tick_binding(
        sample_time_bound.data, np.min(sample_time_bound).data
    )
    assert 'right' == _infer_time_data_tick_binding(
        sample_time_bound.data, np.max(sample_time_bound).data
    )

    with pytest.raises(RuntimeError):
        _infer_time_data_tick_binding(sample_time_bound.data, np.std(sample_time_bound).data)
