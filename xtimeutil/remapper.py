import numpy as np
import pandas as pd
import xarray as xr
from xarray.core import resample_cftime

from .axis import Axis


class Remapper:
    def __init__(
        self, ds, freq, closed=None, label=None, base=0, loffset=None, time_coord_name='time'
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
        closed : {None, 'left', 'right'} , optional
            Make the interval closed with respect to the given frequency to the
            'left', 'right', or both sided (None, the default)
        label : {None, 'left', 'right'}, optional
            Interval boundary to use for labeling.
        base : int, optional
             The "origin" of the adjusted Periods, by default 0
        loffset : str, DateOffset, timedelta object, optional
            The dateoffset to which the Periods will be adjusted, by default None
        time_coord_name : str, optional
            Name of the time coordinate, by default 'time'
        """
        self._ds = ds
        self._from_axis = Axis(ds, time_coord_name)
        self.metadata = self._from_axis.metadata
        self.freq = freq
        self.time_bounds_out = self._generate_output_time_bounds(
            freq, closed=closed, label=label, base=base, loffset=loffset
        )

    def _generate_output_time_bounds(self, freq, closed=None, label=None, base=0, loffset=None):

        lower_time_bounds = self._from_axis.decoded_time_bounds[:, 0].data
        upper_time_bounds = self._from_axis.decoded_time_bounds[:, 1].data

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

        return xr.DataArray(
            dims=[self.metadata['time_coord_name'], self.metadata['time_bounds_dim'] or 'd2'],
            data=np.vstack((lower_time_bounds, upper_time_bounds)).T,
        )


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
