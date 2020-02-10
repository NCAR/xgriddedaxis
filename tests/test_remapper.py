import itertools

import pytest
import scipy
import xarray as xr

from xtimeutil import Remapper
from xtimeutil.testing import create_dataset

freqs = (
    '12000S',
    '200T',
    'H',
    '23H',
    'D',
    '85D',
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
calendars = ('noleap', '360_day', 'all_leap')
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
    'time_units, calendar, decode_times, use_cftime, freq, binding', parameters,
)
def test_init_remapper(time_units, calendar, decode_times, use_cftime, freq, binding):
    ds = create_dataset(
        units=time_units, calendar=calendar, use_cftime=use_cftime, decode_times=decode_times,
    )

    remapper = Remapper(ds, freq=freq, binding=binding)
    assert isinstance(remapper.outgoing_time_bounds, xr.DataArray)
    assert isinstance(remapper.weights, scipy.sparse.csr.csr_matrix)
    assert isinstance(remapper.coverage, dict)


@pytest.mark.parametrize(
    'start, end, in_freq, out_freq, weights, row_idx, col_idx',
    [
        ('2018-01-01', '2018-01-07', 'D', '7D', [1.0] * 7, [0] * 7, list(range(7))),
        ('2018-01-01', '2018-01-07', '7D', 'D', [1.0 / 7] * 7, list(range(7)), [0] * 7),
    ],
)
def test_remapper_coverage(start, end, in_freq, out_freq, weights, row_idx, col_idx):
    ds = create_dataset(start=start, end=end, freq=in_freq)
    remapper = Remapper(ds, freq=out_freq)
    assert weights == remapper.coverage['weights']
    assert row_idx == remapper.coverage['row_idx']
    assert col_idx == remapper.coverage['col_idx']


@pytest.mark.parametrize(
    'time_units, calendar, decode_times, use_cftime, freq, binding', combs1[:5] + combs2[:3],
)
def test_remapper_average(time_units, calendar, decode_times, use_cftime, freq, binding):
    ds = create_dataset(
        units=time_units, calendar=calendar, use_cftime=use_cftime, decode_times=decode_times,
    )

    remapper = Remapper(ds, freq=freq, binding=binding)
    data = remapper.average(ds['var_ex'])
    assert isinstance(data, xr.DataArray)
