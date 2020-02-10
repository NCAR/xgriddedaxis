import itertools

import pytest
import scipy
import xarray as xr

from xtimeutil import Remapper
from xtimeutil.testing import create_dataset

freqs = (
    '4000S',
    '200T',
    'H',
    '23H',
    'D',
    '7D',
    '85D',
    'M',
    'MS',
    '7M',
    'Q',
    'QS',
    '11Q-JUN',
    'A',
    '9YS',
)

# cftime config
time_units = ('days since 1800-01-01',)
calendars = ('standard', 'noleap', '360_day', 'all_leap', '365_day')
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


@pytest.mark.parametrize(
    'time_units, calendar, decode_times, use_cftime, freq, binding', parameters[:10],
)
def test_init_remapper(time_units, calendar, decode_times, use_cftime, freq, binding):
    ds = create_dataset(
        units=time_units, calendar=calendar, use_cftime=use_cftime, decode_times=decode_times,
    )

    remapper = Remapper(ds, freq=freq, binding=binding)
    assert isinstance(remapper.outgoing_time_bounds, xr.DataArray)
    assert isinstance(remapper.weights, scipy.sparse.csr.csr_matrix)


@pytest.mark.parametrize(
    'time_units, calendar, decode_times, use_cftime, freq, binding', parameters,
)
def test_remapper_average(time_units, calendar, decode_times, use_cftime, freq, binding):
    ds = create_dataset(
        units=time_units, calendar=calendar, use_cftime=use_cftime, decode_times=decode_times,
    )

    remapper = Remapper(ds, freq=freq, binding=binding)
    data = remapper.average(ds['var_ex'])
    assert isinstance(data, xr.DataArray)
