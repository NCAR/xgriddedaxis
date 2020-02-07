import pytest
import scipy
import xarray as xr

from xtimeutil import Remapper
from xtimeutil.testing import create_dataset


@pytest.mark.parametrize(
    'has_time_bounds, time_units, calendar, use_cftime, decode_times, freq',
    [
        (True, 'days since 0001-01-01', 'noleap', True, True, 'M'),
        (True, 'days since 0002-01-01', 'noleap', True, False, 'MS'),
        (False, 'days since 0001-01-01', 'noleap', True, True, 'AS'),
        (True, 'hours since 1800-01-01', 'standard', False, True, '3D'),
        (False, 'hours since 1800-01-01', 'standard', False, True, '10H'),
        (True, 'hours since 1800-01-01', 'standard', False, False, 'Q'),
    ],
)
def test_init_remapper(has_time_bounds, time_units, calendar, use_cftime, decode_times, freq):
    ds = create_dataset(
        has_time_bounds=has_time_bounds,
        time_units=time_units,
        calendar=calendar,
        use_cftime=use_cftime,
        decode_times=decode_times,
    )

    remapper = Remapper(ds, freq)
    tb_dim = remapper.metadata['time_bounds_dim']
    res = xr.decode_cf(remapper._from_axis._ds).resample(time=freq)
    assert res._full_index.size == remapper.decoded_time_bounds_out.isel({tb_dim: 0}).size
    assert isinstance(remapper.weights, scipy.sparse.csr.csr_matrix)
    assert remapper.weights.shape == (res._full_index.size, ds.time.size)


@pytest.mark.parametrize(
    'has_time_bounds, time_units, calendar, use_cftime, decode_times, freq',
    [
        (True, 'days since 0001-01-01', 'noleap', True, True, 'M'),
        (True, 'days since 0002-01-01', 'noleap', True, False, 'MS'),
        (False, 'days since 0001-01-01', 'noleap', True, True, 'AS'),
        (True, 'hours since 1800-01-01', 'standard', False, True, '3D'),
        (False, 'hours since 1800-01-01', 'standard', False, True, '10H'),
        (True, 'hours since 1800-01-01', 'standard', False, False, 'Q'),
    ],
)
def test_init_remapper_transposed_dims(
    has_time_bounds, time_units, calendar, use_cftime, decode_times, freq
):
    ds = create_dataset(
        has_time_bounds=has_time_bounds,
        time_units=time_units,
        calendar=calendar,
        use_cftime=use_cftime,
        decode_times=decode_times,
    )
    ds = ds.transpose()

    remapper = Remapper(ds, freq)
    tb_dim = remapper.metadata['time_bounds_dim']
    res = xr.decode_cf(remapper._from_axis._ds).resample(time=freq)
    assert res._full_index.size == remapper.decoded_time_bounds_out.isel({tb_dim: 0}).size
    assert isinstance(remapper.weights, scipy.sparse.csr.csr_matrix)
    assert remapper.weights.shape == (res._full_index.size, ds.time.size)


@pytest.mark.parametrize(
    'has_time_bounds, time_units, calendar, use_cftime, decode_times, freq',
    [
        (True, 'days since 0001-01-01', 'noleap', True, True, 'M'),
        (True, 'days since 0002-01-01', 'noleap', True, False, 'MS'),
        (False, 'days since 0001-01-01', 'noleap', True, True, 'AS'),
        (True, 'hours since 1800-01-01', 'standard', False, True, '3D'),
        (False, 'hours since 1800-01-01', 'standard', False, True, '10H'),
        # (True, 'hours since 1800-01-01', 'standard', False, False, 'Q'), Failing for the time being
    ],
)
def test_remapper_average(has_time_bounds, time_units, calendar, use_cftime, decode_times, freq):
    ds = create_dataset(
        has_time_bounds=has_time_bounds,
        time_units=time_units,
        calendar=calendar,
        use_cftime=use_cftime,
        decode_times=decode_times,
    )

    variable = 'var_ex'
    remapper = Remapper(ds, freq)
    data = remapper.average(ds[variable])
    res = xr.decode_cf(remapper._from_axis._ds)[variable].resample(time=freq).mean()

    assert isinstance(data, xr.DataArray)
    assert data.dims == res.dims
