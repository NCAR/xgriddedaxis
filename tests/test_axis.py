import itertools

import pytest
import xarray as xr

from xtimeutil import Axis
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
    axis = Axis(ds)
    assert axis.metadata['is_time_decoded'] == decode_times
    assert axis.decoded_times.shape == ds.time.shape


@pytest.mark.parametrize(
    'dataset, decode_times',
    [('rasm', False), ('rasm', True), ('air_temperature', False), ('air_temperature', True)],
)
def test_init_missing_bounds(dataset, decode_times):
    ds = xr.tutorial.open_dataset(dataset, decode_times=decode_times)
    with pytest.raises(RuntimeError):
        _ = Axis(ds)


def test_validate_time_coord():
    ds = create_dataset()
    with pytest.raises(KeyError):
        _ = Axis(ds, 'times')


def test_invalid_binding():
    ds = create_dataset()
    with pytest.raises(KeyError):
        _ = Axis(ds, 'time', 'center')
