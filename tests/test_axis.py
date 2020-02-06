import pytest
import xarray as xr

from xtimeutil import Axis
from xtimeutil.testing import create_dataset


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
