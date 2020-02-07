from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix
from xarray.core import resample_cftime

from .axis import Axis, _get_time_bounds_dims


class Remapper:
    def __init__(
        self, ds, freq, time_coord_name='time', closed=None, label=None, base=0, loffset=None
    ):
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
        closed : {None, 'left', 'right'} , optional
            Make the interval closed with respect to the given frequency to the
            'left', 'right', or both sided (None, the default)
        label : {None, 'left', 'right'}, optional
            Interval boundary to use for labeling.
        base : int, optional
             The "origin" of the adjusted Periods, by default 0
        loffset : str, DateOffset, timedelta object, optional
            The dateoffset to which the Periods will be adjusted, by default None

        """
        self._ds = ds
        self._from_axis = Axis(ds, time_coord_name)
        self.metadata = self._from_axis.metadata
        self.freq = freq
        self.decoded_time_bounds_in = self._from_axis.decoded_time_bounds
        self.decoded_time_bounds_out = self._generate_output_time_bounds(
            freq, closed=closed, label=label, base=base, loffset=loffset
        )
        self.weights = self._get_coverage_matrix(
            self.decoded_time_bounds_in, self.decoded_time_bounds_out
        )

    def _generate_output_time_bounds(self, freq, closed=None, label=None, base=0, loffset=None):
        tb_dim = self.metadata['time_bounds_dim']
        lower_time_bounds = self.decoded_time_bounds_in.isel({tb_dim: 0}).data
        upper_time_bounds = self.decoded_time_bounds_in.isel({tb_dim: 1}).data

        if self.metadata['decoded_time_object_type'] == 'cftime':
            grouper = resample_cftime.CFTimeGrouper(freq, closed, label, base, loffset)
            lower_time_bounds = xr.CFTimeIndex(lower_time_bounds)
            upper_time_bounds = xr.CFTimeIndex(upper_time_bounds)
        else:
            grouper = pd.Grouper(freq=freq, closed=closed, label=label, base=base, loffset=loffset)
            lower_time_bounds = pd.DatetimeIndex(lower_time_bounds)
            upper_time_bounds = pd.DatetimeIndex(upper_time_bounds)

        lower_time_bounds, _ = _get_index_and_items(lower_time_bounds, grouper)
        upper_time_bounds, _ = _get_index_and_items(upper_time_bounds, grouper)

        dims = _get_time_bounds_dims(self.metadata)
        data = np.vstack((lower_time_bounds, upper_time_bounds))
        if self.metadata['time_bounds_dim_axis_num'] == 1:
            data = data.T

        return xr.DataArray(dims=dims, data=data,)

    def _get_coverage_matrix(self, decoded_time_bounds_in, decoded_time_bounds_out):
        encoded_time_bounds_in, _, _ = xr.coding.times.encode_cf_datetime(decoded_time_bounds_in)
        encoded_time_bounds_out, _, _ = xr.coding.times.encode_cf_datetime(decoded_time_bounds_out)

        if self.metadata['time_bounds_dim_axis_num'] == 1:
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

        wgts = csr_matrix((weights, (row_idx, col_idx)), shape=(m, n)).tolil()
        mask = np.asarray(wgts.sum(axis=1)).flatten() == 0
        wgts[mask, 0] = np.nan
        return wgts.tocsr()

    def _get_time_axis_dim_num(self, da):
        """
        Return the dimension number of the time axis coordinate in a DataArray.
        """
        time_coord_name = self.metadata['time_coord_name']
        return da.get_axis_num(time_coord_name)

    def _prepare_input_data(self, da):
        n = self.weights.shape[1]
        data = da.data.copy()
        time_axis = self._get_time_axis_dim_num(da)
        if data.ndim == 1:
            data = data.reshape((-1, 1))

        if data.shape[time_axis] != n:
            message = f'The length ({data.shape[time_axis]}) of input time dimension does not match to that of the provided remapper ({n})'
            raise ValueError(message)

        if time_axis != 0:
            data = np.moveaxis(data, time_axis, 0)

        trailing_shape = data.shape[1:]
        data = data.reshape((n, -1))

        return data, trailing_shape

    def _prepare_output_data(self, input_data, output_data, time_axis, trailing_shape):

        shape = (output_data.shape[0], *trailing_shape)
        print(shape)
        data = np.moveaxis(output_data.reshape(shape), 0, time_axis)

        original_dims = input_data.dims
        coords = OrderedDict()
        dims = []
        for dim in original_dims:
            if dim != self.metadata['time_coord_name']:
                if dim in input_data.coords:
                    coords[dim] = input_data.coords[dim]
                    dims.append(dim)
            else:
                times = xr.DataArray(
                    self.decoded_time_bounds_out.mean(dim=self.metadata['time_bounds_dim'])
                )
                coords[dim] = xr.DataArray(
                    times, coords={self.metadata['time_coord_name']: times}, attrs=input_data.attrs
                )
                dims.append(dim)

        return xr.DataArray(data.squeeze(), dims=dims, coords=coords)

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


def _get_index_and_items(index, grouper):
    """
    Copied from xarray: https://bit.ly/3896G6Q
    """
    s = pd.Series(np.arange(index.size), index)
    if isinstance(grouper, resample_cftime.CFTimeGrouper):
        first_items = grouper.first_items(index)
    else:
        first_items = s.groupby(grouper).first()
        xr.core.groupby._apply_loffset(grouper, first_items)
    full_index = first_items.index
    if first_items.isnull().any():
        first_items = first_items.dropna()
    return full_index, first_items
