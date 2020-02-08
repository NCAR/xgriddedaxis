import numpy as np
import xarray as xr


class Axis:
    """
    An object that represents a time coordinate axis.
    """

    _bindings = {'left': np.min, 'right': np.max, 'middle': np.mean}

    def __init__(self, ds, time_coord_name='time', binding='middle'):
        """
        Create a new Axis object from an input dataset

        Parameters
        ----------
        ds : xarray.Dataset
            Contains the relevant time coordinate information.
        time_coord_name : str, optional
            Name for time coordinate to use, by default 'time'
        binding : {'left', 'right', 'middle'}, optional
            Defines different ways a data tick could be bound to an interval.

            - `left`: means that the data tick is bound to the left/beginning of
              the interval or the lower time bound.

            - `right`: means that the data tick is bound to the right/end of the
               interval or the upper time bound.

            - `middle`: means that the data tick is bound half-way through between
               lower time bound and upper time bound.
        """
        _validate_time_coord(ds, time_coord_name)
        self._ds = ds
        self.metadata = {}
        if binding in Axis._bindings:
            self.metadata['binding'] = binding
        else:
            message = f'Could not find the Time Axis binding associated to `{binding}`. '
            message += f'Possible options are: {list(Axis._bindings.keys())}'
            raise KeyError(message)
        self.metadata['is_time_decoded'] = _is_time_decoded(ds[time_coord_name])
        self.metadata['time_coord_name'] = time_coord_name
        self.metadata.update(self._get_time_attrs())
        if self.metadata['is_time_decoded']:
            self.metadata['input_time_object_type'] = _datetime_object_type(ds[time_coord_name])
            self._time = xr.coding.times.encode_cf_datetime(
                self._ds[self.metadata['time_coord_name']],
                units=self.metadata['units'],
                calendar=self.metadata['calendar'],
            )[0]
        else:
            self.metadata['input_time_object_type'] = 'numeric'
            self._time = self._ds[self.metadata['time_coord_name']].copy()

        if _is_time_decoded(self._ds[self.metadata['time_bounds_varname']]):
            self._time_bounds = xr.coding.times.encode_cf_datetime(
                self._ds[self.metadata['time_bounds_varname']],
                units=self.metadata['units'],
                calendar=self.metadata['calendar'],
            )[0]

        else:
            self._time_bounds = self._ds[self.metadata['time_bounds_varname']].copy()

        self.decoded_times = xr.DataArray(
            dims=[self.metadata['time_coord_name']],
            data=xr.coding.times.decode_cf_datetime(
                Axis._bindings[self.metadata['binding']](
                    self._time_bounds, axis=self.metadata['time_bounds_dim_axis_num']
                ),
                self.metadata['units'],
                self.metadata['calendar'],
            ),
        )

        dims = _get_time_bounds_dims(self.metadata)

        self.decoded_time_bounds = xr.DataArray(
            dims=dims,
            data=xr.coding.times.decode_cf_datetime(
                self._time_bounds, self.metadata['units'], self.metadata['calendar']
            ),
        )

        self.metadata['decoded_time_object_type'] = _datetime_object_type(self.decoded_times)

    def _get_time_attrs(self):
        attrs = getattr(self._ds[self.metadata['time_coord_name']], 'attrs')
        encoding = getattr(self._ds[self.metadata['time_coord_name']], 'encoding')

        if 'units' in attrs:
            units = attrs['units']
        elif 'units' in encoding:
            units = encoding['units']

        if 'calendar' in attrs:
            calendar = attrs['calendar']
        elif 'calendar' in encoding:
            calendar = encoding['calendar']
        else:
            calendar = 'standard'

        if 'bounds' in attrs:
            time_bounds_varname = attrs['bounds']

        elif 'bounds' in encoding:
            time_bounds_varname = encoding['bounds']

        else:
            message = f'Unable to infer the time coordinate boundary variable'
            # TODO: Tell the user how to generate/provide time bounds.
            raise RuntimeError(message)

        dims = set(self._ds[time_bounds_varname].dims)
        time_coord_name = set([self.metadata['time_coord_name']])
        time_bounds_dim = (dims - time_coord_name).pop()
        time_bounds_dim_axis_num = self._ds[time_bounds_varname].get_axis_num(time_bounds_dim)
        return {
            'units': units,
            'calendar': calendar,
            'time_bounds_varname': time_bounds_varname,
            'time_bounds_dim': time_bounds_dim,
            'time_bounds_dim_axis_num': time_bounds_dim_axis_num,
        }


def _get_time_bounds_dims(metadata):

    if metadata['time_bounds_dim_axis_num'] == 1:
        dims = [metadata['time_coord_name'], metadata['time_bounds_dim']]
    else:
        dims = [metadata['time_bounds_dim'], metadata['time_coord_name']]

    return dims


def _validate_time_coord(ds, time_coord_name):
    if time_coord_name not in ds.variables:
        raise KeyError(f'Could not find time coordinate name = {time_coord_name} in the dataset.')


def _is_time_decoded(x):
    return xr.core.common._contains_datetime_like_objects(x)


def _datetime_object_type(x):
    if xr.core.common.contains_cftime_datetimes(x):
        return 'cftime'
    else:
        return 'np_datetime'
