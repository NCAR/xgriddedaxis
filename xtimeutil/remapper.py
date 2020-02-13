import re
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix

from .axis import Axis, _get_time_bounds_dims

_FREQUENCIES = {
    'A': 'AS',
    'AS': 'AS',
    'Y': 'YS',
    'YS': 'YS',
    'Q': 'QS',
    'QS': 'QS',
    'M': 'MS',
    'MS': 'MS',
    'D': 'D',
    'H': 'H',
    'T': 'T',
    'min': 'min',
    'S': 'S',
    'AS-JAN': 'AS-JAN',
    'AS-FEB': 'AS-FEB',
    'AS-MAR': 'AS-MAR',
    'AS-APR': 'AS-APR',
    'AS-MAY': 'AS-MAY',
    'AS-JUN': 'AS-JUN',
    'AS-JUL': 'AS-JUL',
    'AS-AUG': 'AS-AUG',
    'AS-SEP': 'AS-SEP',
    'AS-OCT': 'AS-OCT',
    'AS-NOV': 'AS-NOV',
    'AS-DEC': 'AS-DEC',
    'A-JAN': 'AS-JAN',
    'A-FEB': 'AS-FEB',
    'A-MAR': 'AS-MAR',
    'A-APR': 'AS-APR',
    'A-MAY': 'AS-MAY',
    'A-JUN': 'AS-JUN',
    'A-JUL': 'AS-JUL',
    'A-AUG': 'AS-AUG',
    'A-SEP': 'AS-SEP',
    'A-OCT': 'AS-OCT',
    'A-NOV': 'AS-NOV',
    'A-DEC': 'AS-DEC',
    'QS-JAN': 'QS-JAN',
    'QS-FEB': 'QS-FEB',
    'QS-MAR': 'QS-MAR',
    'QS-APR': 'QS-APR',
    'QS-MAY': 'QS-MAY',
    'QS-JUN': 'QS-JUN',
    'QS-JUL': 'QS-JUL',
    'QS-AUG': 'QS-AUG',
    'QS-SEP': 'QS-SEP',
    'QS-OCT': 'QS-OCT',
    'QS-NOV': 'QS-NOV',
    'QS-DEC': 'QS-DEC',
    'Q-JAN': 'QS-JAN',
    'Q-FEB': 'QS-FEB',
    'Q-MAR': 'QS-MAR',
    'Q-APR': 'QS-APR',
    'Q-MAY': 'QS-MAY',
    'Q-JUN': 'QS-JUN',
    'Q-JUL': 'QS-JUL',
    'Q-AUG': 'QS-AUG',
    'Q-SEP': 'QS-SEP',
    'Q-OCT': 'QS-OCT',
    'Q-NOV': 'QS-NOV',
    'Q-DEC': 'QS-DEC',
}

# Copied from xarray
# https://github.com/pydata/xarray/blob/52ee5df/xarray/coding/cftime_offsets.py
_FREQUENCY_CONDITION = '|'.join(_FREQUENCIES.keys())
_PATTERN = fr'^((?P<multiple>\d+)|())(?P<freq>({_FREQUENCY_CONDITION}))$'


def _validate_freq(freq):
    try:
        freq_data = re.match(_PATTERN, freq).groupdict()
    except AttributeError:
        valid_frequencies = sorted(list(set(_FREQUENCIES.keys()) - set(_FREQUENCIES.values())))
        raise ValueError(
            f'Invalid frequency string provided. Valid frequencies (plus their multiples) include: {valid_frequencies}'
        )
    freq = freq_data['freq']
    multiples = freq_data['multiple']
    if multiples is None:
        multiples = 1
    else:
        multiples = int(multiples)
    return f'{multiples}{_FREQUENCIES[freq]}'


class Remapper:
    """ An object that facilitates conversion between two time axis.
    """

    def __init__(self, ds, freq, time_coord_name='time', binding='middle'):
        """
        Create a new Remapper object that facilitates conversion between two time axis.

        Parameters
        ----------
        ds : xarray.Dataset
            Contains the relevant time coordinate information.
        freq : str / frequency object
            The offset object representing target conversion also known as resampling
            frequency (e.g., 'MS', '2D', 'H', or '3T' For full specification of available
            frequencies, please see `here
            <https://xarray.pydata.org/en/stable/generated/xarray.cftime_range.html>`_.
        time_coord_name : str, optional
            Name of the time coordinate, by default 'time'
        binding : {'left', 'right', 'middle'}, optional
            Defines different ways a data tick could be bound to an interval.

            - `left`: means that the data tick is bound to the left/beginning of
              the interval or the lower time bound.

            - `right`: means that the data tick is bound to the right/end of the
               interval or the upper time bound.

            - `middle`: means that the data tick is bound half-way through between
               lower time bound and upper time bound.

        """
        self._ds = ds
        self.binding = binding
        self._from_axis = Axis(ds, time_coord_name, binding)
        self.ti = self._from_axis.decoded_time_bounds.values.flatten().min()
        self.tf = self._from_axis.decoded_time_bounds.values.flatten().max()
        self.freq = _validate_freq(freq)
        self.coverage = {}
        self.incoming_time_bounds = self._from_axis.decoded_time_bounds
        self.outgoing_time_bounds = self._generate_outgoing_time_bounds()
        self.weights = self._get_coverage_matrix(
            self.incoming_time_bounds, self.outgoing_time_bounds
        )

    def _generate_outgoing_time_bounds(self):

        warning_message = f'Resample frequency is greater than extent of incoming time axis. Doubling time axis interval.'

        if xr.core.common.is_np_datetime_like(self.incoming_time_bounds.dtype):
            # Use to_offset() function to compute offset that allows us to generate
            # time range that includes the end of the incoming time bounds.
            offset = pd.tseries.frequencies.to_offset(self.freq)

            time_bounds = pd.date_range(
                start=pd.to_datetime(self.ti), end=pd.to_datetime(self.tf), freq=self.freq
            )

            if (len(time_bounds) == 1) or (time_bounds[-1] < self.tf):
                # this should be rare
                if len(time_bounds) == 1:
                    warnings.warn(warning_message)

                time_bounds = pd.date_range(
                    start=pd.to_datetime(self.ti),
                    end=pd.to_datetime(self.tf) + offset,
                    freq=self.freq,
                )

        else:
            offset = xr.coding.cftime_offsets.to_offset(self.freq)
            time_bounds = xr.cftime_range(
                start=self.ti,
                end=self.tf,
                freq=self.freq,
                calendar=self._from_axis.metadata['calendar'],
            )

            if (len(time_bounds) == 1) or (time_bounds[-1] < self.tf):
                # this should be rare
                if len(time_bounds) == 1:
                    warnings.warn(warning_message)

                time_bounds = xr.cftime_range(
                    start=self.ti,
                    end=self.tf + offset,
                    freq=self.freq,
                    calendar=self._from_axis.metadata['calendar'],
                )

        msg = f"""{self.tf} upper bound from the incoming time axis is not covered in the outgoing
        time axis which has {time_bounds[-1]} as the upper bound."""

        assert time_bounds[-1] >= self.tf, msg
        outgoing_time_bounds = np.vstack((time_bounds[:-1], time_bounds[1:])).T
        dims = _get_time_bounds_dims(self._from_axis.metadata)

        if self._from_axis.metadata['time_bounds_dim_axis_num'] == 0:
            outgoing_time_bounds = outgoing_time_bounds.T

        return xr.DataArray(dims=dims, data=outgoing_time_bounds)

    def _get_coverage_matrix(self, incoming_time_bounds, outgoing_time_bounds):
        encoded_time_bounds_in, _, _ = xr.coding.times.encode_cf_datetime(incoming_time_bounds)
        encoded_time_bounds_out, _, _ = xr.coding.times.encode_cf_datetime(outgoing_time_bounds)

        if self._from_axis.metadata['time_bounds_dim_axis_num'] == 1:
            from_lower_bound = encoded_time_bounds_in[:, 0]
            from_upper_bound = encoded_time_bounds_in[:, 1]
            to_lower_bound = encoded_time_bounds_out[:, 0]
            to_upper_bound = encoded_time_bounds_out[:, 1]

        else:
            from_lower_bound = encoded_time_bounds_in[0, :]
            from_upper_bound = encoded_time_bounds_in[1, :]
            to_lower_bound = encoded_time_bounds_out[0, :]
            to_upper_bound = encoded_time_bounds_out[1, :]

        m = to_lower_bound.size
        n = from_lower_bound.size

        row_idx = []
        col_idx = []
        weights = []
        for r in range(m):
            toLB = to_lower_bound[r]
            toUB = to_upper_bound[r]
            for c in range(n):
                fromLB = from_lower_bound[c]
                fromUB = from_upper_bound[c]
                fromLength = fromUB - fromLB

                if (fromUB <= toLB) or (fromLB >= toUB):  # No coverage
                    continue
                elif (fromLB <= toLB) and (fromUB >= toLB) and (fromUB <= toUB):
                    row_idx.append(r)
                    col_idx.append(c)
                    weights.append((fromUB - toLB) / fromLength)
                elif (fromLB >= toLB) and (fromLB < toUB) and (fromUB >= toUB):
                    row_idx.append(r)
                    col_idx.append(c)
                    weights.append((toUB - fromLB) / fromLength)
                elif (fromLB >= toLB) and (fromUB <= toUB):
                    row_idx.append(r)
                    col_idx.append(c)
                    weights.append(1.0)
                elif (fromLB <= toLB) and (fromUB >= toUB):
                    row_idx.append(r)
                    col_idx.append(c)
                    weights.append((toUB - toLB) / fromLength)

        self.coverage['weights'] = weights
        self.coverage['col_idx'] = col_idx
        self.coverage['row_idx'] = row_idx

        wgts = csr_matrix((weights, (row_idx, col_idx)), shape=(m, n)).tolil()
        mask = np.asarray(wgts.sum(axis=1)).flatten() == 0
        wgts[mask, 0] = np.nan
        return wgts.tocsr()

    def _get_time_axis_dim_num(self, da):
        """
        Return the dimension number of the time axis coordinate in a DataArray.
        """
        time_coord_name = self._from_axis.metadata['time_coord_name']
        return da.get_axis_num(time_coord_name)

    def _prepare_input_data(self, da):
        n = self.weights.shape[1]
        data = da.data.copy()
        time_axis = self._get_time_axis_dim_num(da)
        if data.ndim == 1:
            data = data.reshape((-1, 1))

        if data.shape[time_axis] != n:
            message = f"""The length ({data.shape[time_axis]}) of input time dimension does not
            match to that of the provided remapper ({n})"""
            raise ValueError(message)

        if time_axis != 0:
            data = np.moveaxis(data, time_axis, 0)

        trailing_shape = data.shape[1:]
        data = data.reshape((n, -1))

        return data, trailing_shape

    def _prepare_output_data(self, input_data, output_data, time_axis, trailing_shape):

        shape = (output_data.shape[0], *trailing_shape)
        data = np.moveaxis(output_data.reshape(shape), 0, time_axis)

        original_dims = input_data.dims
        coords = OrderedDict()
        dims = []
        for dim in original_dims:
            if dim != self._from_axis.metadata['time_coord_name']:
                if dim in input_data.coords:
                    coords[dim] = input_data.coords[dim]
                    dims.append(dim)
            else:

                times = self._from_axis._bindings[self.binding](
                    self.outgoing_time_bounds,
                    axis=self._from_axis.metadata['time_bounds_dim_axis_num'],
                )
                times = xr.DataArray(times)
                coords[dim] = xr.DataArray(
                    times,
                    coords={self._from_axis.metadata['time_coord_name']: times},
                    attrs=input_data.attrs,
                )
                dims.append(dim)

        return xr.DataArray(data, dims=dims, coords=coords)

    def average(self, da):
        time_axis = self._get_time_axis_dim_num(da)
        input_data, trailing_shape = self._prepare_input_data(da)
        nan_mask = np.isnan(input_data)
        non_nan_mask = np.ones(input_data.shape, dtype=np.int8)
        non_nan_mask[nan_mask] = 0
        input_data[nan_mask] = 0

        inverse_sum_effective_weights = np.reciprocal(self.weights * non_nan_mask)
        output_data = np.multiply(self.weights * input_data, inverse_sum_effective_weights)
        output_data = self._prepare_output_data(da, output_data, time_axis, trailing_shape)
        return output_data
