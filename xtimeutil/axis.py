import numpy as np
import xarray as xr

BINDINGS = {'left': np.min, 'right': np.max, 'middle': np.mean}


def get_time_axis_info(ds, time_coord_name='time', binding=None):
    """
    Parameters
    ----------
    ds : xarray.Dataset
        Contains the relevant time coordinate information.
    time_coord_name : str, optional
        Name for time coordinate to use, by default 'time'
    binding : {'left', 'right', 'middle'}, optional
        Defines different ways time data tick could be bound to an interval.
        If None (default), attempt at inferring the time data tick binding from the
        input data set.

        - `left`: means that the data tick is bound to the left/beginning of
            the interval or the lower time bound.

        - `right`: means that the data tick is bound to the right/end of the
            interval or the upper time bound.

        - `middle`: means that the data tick is bound half-way through between
            lower time bound and upper time bound.
    """

    _validate_time_coord(ds, time_coord_name)
    info = {'time_coord_name': time_coord_name}
    info['is_time_decoded'] = _is_time_decoded(ds[time_coord_name])
    time_attrs = _get_time_attrs(ds, time_coord_name)

    info['use_cftime'] = _use_cftime(
        ds[time_coord_name], time_attrs['units'], time_attrs['calendar']
    )

    encoded_times, encoded_time_bounds = _get_encoded_time(
        ds,
        info['is_time_decoded'],
        time_coord_name,
        time_attrs['time_bounds_varname'],
        time_attrs['units'],
        time_attrs['calendar'],
    )

    if binding is None:
        binding = _infer_time_data_tick_binding(encoded_time_bounds[0], encoded_times[0])
        info['binding'] = binding
    elif binding in BINDINGS:
        info['binding'] = binding
    else:
        message = f'Could not find the Time Axis binding associated to `{binding}`. '
        message += f'Possible options are: {list(BINDINGS.keys())}'
        raise KeyError(message)

    decoded_times, decoded_time_bounds = _get_decoded_time(
        time_coord_name,
        info['binding'],
        encoded_time_bounds,
        time_attrs['time_bounds_dim_axis_num'],
        time_attrs['time_bounds_dim'],
        time_attrs['units'],
        time_attrs['calendar'],
        info['use_cftime'],
    )

    info.update(time_attrs)
    info['encoded_times'], info['encoded_time_bounds'] = encoded_times, encoded_time_bounds
    info['decoded_times'], info['decoded_time_bounds'] = decoded_times, decoded_time_bounds
    return info


def _get_encoded_time(ds, is_time_decoded, time_coord_name, time_bounds_varname, units, calendar):
    if is_time_decoded:
        encoded_times = xr.coding.times.encode_cf_datetime(
            ds[time_coord_name], units=units, calendar=calendar,
        )[0]

        encoded_time_bounds = xr.coding.times.encode_cf_datetime(
            ds[time_bounds_varname], units=units, calendar=calendar,
        )[0]
    else:
        encoded_times = ds[time_coord_name].copy()
        encoded_time_bounds = ds[time_bounds_varname].copy()

    return encoded_times, encoded_time_bounds


def _get_decoded_time(
    time_coord_name,
    binding,
    encoded_time_bounds,
    time_bounds_dim_axis_num,
    time_bounds_dim,
    units,
    calendar,
    use_cftime,
):

    decoded_times = xr.DataArray(
        dims=[time_coord_name],
        data=xr.coding.times.decode_cf_datetime(
            BINDINGS[binding](encoded_time_bounds, axis=time_bounds_dim_axis_num),
            units,
            calendar,
            use_cftime=use_cftime,
        ),
    )

    dims = _get_time_bounds_dims(time_bounds_dim_axis_num, time_bounds_dim, time_coord_name)

    decoded_time_bounds = xr.DataArray(
        dims=dims,
        data=xr.coding.times.decode_cf_datetime(
            encoded_time_bounds, units, calendar, use_cftime=use_cftime,
        ),
    )

    return decoded_times, decoded_time_bounds


def _get_time_attrs(ds, time_coord_name):
    attrs = getattr(ds[time_coord_name], 'attrs')
    encoding = getattr(ds[time_coord_name], 'encoding')

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
        message = f'Unable to infer the {time_coord_name} coordinate boundary variable'
        # TODO: Tell the user how to generate/provide time bounds.
        raise RuntimeError(message)

    dims = set(ds[time_bounds_varname].dims)
    time_coord_name = set([time_coord_name])
    time_bounds_dim = (dims - time_coord_name).pop()
    time_bounds_dim_axis_num = ds[time_bounds_varname].get_axis_num(time_bounds_dim)
    return {
        'units': units,
        'calendar': calendar,
        'time_bounds_varname': time_bounds_varname,
        'time_bounds_dim': time_bounds_dim,
        'time_bounds_dim_axis_num': time_bounds_dim_axis_num,
    }


def _get_time_bounds_dims(time_bounds_dim_axis_num, time_bounds_dim, time_coord_name):

    if time_bounds_dim_axis_num == 1:
        dims = [time_coord_name, time_bounds_dim]
    else:
        dims = [time_bounds_dim, time_coord_name]

    return dims


def _validate_time_coord(ds, time_coord_name):
    if time_coord_name not in ds.variables:
        raise KeyError(f'Could not find time coordinate name = {time_coord_name} in the dataset.')


def _is_time_decoded(x):
    return xr.core.common._contains_datetime_like_objects(x)


def _use_cftime(x, units, calendar):
    if xr.core.common.contains_cftime_datetimes(x):
        return True

    elif xr.core.common.is_np_datetime_like(x.dtype):
        return False

    else:

        dummy = xr.coding.times.decode_cf_datetime(x[:2], units=units, calendar=calendar)
        if xr.core.common.is_np_datetime_like(np.array(dummy).dtype):
            return False
        else:
            return True


def _infer_time_data_tick_binding(sample_time_bound, sample_time_data_tick):

    if np.isclose(sample_time_data_tick, sample_time_bound[1]):
        return 'right'

    elif np.isclose(sample_time_data_tick, sample_time_bound[0]):
        return 'left'

    elif np.isclose(
        sample_time_data_tick,
        sample_time_bound[0] + (sample_time_bound[1] - sample_time_bound[0]) / 2.0,
    ):
        return 'middle'

    else:
        message = f"""Unable to infer time data tick binding from the input data set.
                   Please specify the `binding` parameter. Accepted values: {list(BINDINGS.keys())}"""
        raise RuntimeError(message)
