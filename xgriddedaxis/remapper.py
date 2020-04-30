import dask
import numpy as np
import sparse
import xarray as xr
from scipy.sparse import csr_matrix

_TO_KEY = 'to'
_FROM_KEY = 'from'


class Remapper:
    """
    An object that facilitates conversion of data between two one dimensional axes.
    """

    def __init__(
        self, from_axis, to_axis, axis_name, boundary_variable,
    ):
        """
        Create a new Remapper object that facilitates conversion of data between two
        one-dimensional axes.

        Parameters
        ----------
        from_axis : xarray.Dataset
           Contains `from` axis information. This dataset should have
           a 2D bounds variable (containing lower and upper bounds) and the
           corresponding data ticks as a coordinate. The data tick defines where
           in the bounds interval you are associating the data point.
        to_axis : xarray.Dataset
           Contains `to` axis information. This dataset should have
           a 2D bounds variable (containing lower and upper bounds) and the
           corresponding data ticks as a coordinate. The data tick defines where
           in the bounds interval you are associating the data point.
        axis_name : str
           Name of the axis. For example, `time`, `lat`, etc..
        boundary_variable : str
           Name of the variable with bounds array.

        Notes
        -----
        The lower bounds values must be monotonically increasing.
        The upper bounds do NOT need to be monotonically increasing.
        However, they should not be smaller than their counter-part lower bounds.

        """
        self.axis_name = axis_name
        self.boundary_variable = boundary_variable
        self.from_axis = from_axis
        self.to_axis = to_axis
        self.coverage_info = None
        self.weights = self.generate_weights()

    def generate_weights(self):
        """
        Generate remapping weights.
        """
        self.coverage_info = get_coverage_info(
            self.from_axis[self.boundary_variable].data, self.to_axis[self.boundary_variable].data,
        )

        from_data_ticks = self.from_axis[self.axis_name].data
        to_data_ticks = self.to_axis[self.axis_name].data

        _data_ticks_sanity_check(from_data_ticks)
        _data_ticks_sanity_check(to_data_ticks)
        coords = {
            _FROM_KEY: from_data_ticks,
            _TO_KEY: to_data_ticks,
        }
        weights = construct_coverage_matrix(
            self.coverage_info['weights'],
            self.coverage_info['col_idx'],
            self.coverage_info['row_idx'],
            self.coverage_info['shape'],
            coords=coords,
        )
        return weights

    def __call__(self, data):
        """
        Apply remapping weights to data.

        Parameters
        ----------
        data : xarray.DataArray, xarray.Dataset
            Data to map from the "from" axis to the "to" axis.
        Returns
        -------
        outdata : xarray.DataArray, xarray.Dataset
            Remapped data. Data type is the same as input data type.
            All the dimensions are the same as the input data except the "from" axis.

        Raises
        ------
        TypeError
            if input data is not an xarray DataArray or xarray Dataset.
        """
        if isinstance(data, xr.DataArray):
            return self._remap_dataarray(data)
        elif isinstance(data, xr.Dataset):
            return self._remap_dataset(data)
        else:
            raise TypeError('input data must be xarray DataArray or xarray Dataset!')

    def _remap_dataarray(self, dr_in):
        weights = self.weights.copy()
        # Convert sparse matrix into a dense one to avoid TypeError when performing dot product
        # TypeError: no implementation found for 'numpy.einsum' on types that implement
        # __array_function__: [<class 'sparse._coo.core.COO'>, <class 'numpy.ndarray'>]
        weights.data = weights.data.todense()
        indata = _sanitize_input_data(dr_in, self.axis_name, self.weights)
        if isinstance(indata.data, dask.array.Array):
            from_axis_chunks = dict(zip(indata.dims, indata.chunks))[self.axis_name][0]

            weights = weights.chunk({_TO_KEY: from_axis_chunks})
            return _apply_weights(weights, indata, self.axis_name,)
        else:
            return _apply_weights(weights, indata, self.axis_name)

    def _remap_dataset(self, ds_in):
        raise NotImplementedError('Currently only works on xarray DataArrays')


def _sanitize_input_data(data, axis_name, weights):
    message = (
        f'The length ({data[axis_name].size}) of `from_axis` {axis_name} dimension does not match '
        f"with the provided remapper object's `from_axis` {axis_name} dimension ({weights[_FROM_KEY].size})"
    )
    assert data[axis_name].size == weights[_FROM_KEY].size, message
    indata = data.copy()
    indata[axis_name] = weights[_FROM_KEY].data
    return indata


def _apply_weights(weights, indata, axis_name):
    """
    Apply remapping weights to data. We apply weights normalization
    when we have missing values in the input data.

    Parameters
    ----------
    weights : xarray.DataArray
       Remapping weights
    indata : xarray.DataArray
       Input data to remap to a new axis.
    axis_name : str
        Name of the axis. For example, `time`, `lat`, etc..

    Returns
    -------
    outdata : xarray.DataArray
       Remapped data

    """
    indata = indata.rename({axis_name: _FROM_KEY})
    nan_mask = indata.isnull()
    non_nan_mask = xr.ones_like(indata, dtype=np.int8)
    non_nan_mask = non_nan_mask.where(~nan_mask, 0)
    indata = indata.where(~nan_mask, 0)
    inverse_sum_effective_weights = np.reciprocal(xr.dot(weights, non_nan_mask))
    outdata = xr.dot(weights, indata) * inverse_sum_effective_weights
    return outdata.rename({_TO_KEY: axis_name})


def get_coverage_info(from_bounds, to_bounds):
    """
    Compute the overlap/coverage between the "from" and "to" bounds

    Parameters
    ----------
    from_bounds : numpy.array
        "from" bounds
    to_bounds : numpy.array
        "to" bounds
    Returns
    -------
    dict
        Dictionary containing information used to generate remapping weights matrix
    """
    _bounds_sanity_check(from_bounds)
    _bounds_sanity_check(to_bounds)

    from_lower_bounds = from_bounds[:, 0]
    from_upper_bounds = from_bounds[:, 1]
    to_lower_bounds = to_bounds[:, 0]
    to_upper_bounds = to_bounds[:, 1]

    n = from_lower_bounds.size
    m = to_lower_bounds.size

    row_idx = []
    col_idx = []
    weights = []
    for r in range(m):
        toLB = to_lower_bounds[r]
        toUB = to_upper_bounds[r]
        toLength = toUB - toLB
        for c in range(n):
            fromLB = from_lower_bounds[c]
            fromUB = from_upper_bounds[c]
            fromLength = fromUB - fromLB

            if (fromUB <= toLB) or (fromLB >= toUB):  # No coverage
                continue
            elif (fromLB <= toLB) and (fromUB >= toLB) and (fromUB <= toUB):
                row_idx.append(r)
                col_idx.append(c)
                fraction_overlap = (fromUB - toLB) / fromLength
                weights.append(fraction_overlap * (fromLength / toLength))
            elif (fromLB >= toLB) and (fromLB < toUB) and (fromUB >= toUB):
                row_idx.append(r)
                col_idx.append(c)
                fraction_overlap = (toUB - fromLB) / fromLength
                weights.append(fraction_overlap * (fromLength / toLength))
            elif (fromLB >= toLB) and (fromUB <= toUB):
                row_idx.append(r)
                col_idx.append(c)
                fraction_overlap = 1.0
                weights.append(fraction_overlap * (fromLength / toLength))
            elif (fromLB <= toLB) and (fromUB >= toUB):
                row_idx.append(r)
                col_idx.append(c)
                fraction_overlap = (toUB - toLB) / fromLength
                weights.append(fraction_overlap * (fromLength / toLength))

    coverage = {
        'weights': weights,
        'col_idx': col_idx,
        'row_idx': row_idx,
        'shape': (m, n),
    }
    return coverage


def construct_coverage_matrix(weights, col_idx, row_idx, shape, coords):
    """
    Generate remapping weights sparse matrix.

    Parameters
    ----------
    weights : array_like
        Contain overlap/coverage between the "from" and "to" bounds
    col_idx : array_like
        column indices
    row_idx : array_like
        row indices
    shape : tuple
        Shape of the matrix
    coords : dict
        Dictionary-like container of coordinate arrays.

    Returns
    -------
    xarray.DataArray
        Contains the remapping weights (stored as a sparse matrix in COO format)
    """
    wgts = csr_matrix((weights, (row_idx, col_idx)), shape=shape).tolil()
    mask = np.asarray(wgts.sum(axis=1)).flatten() == 0
    wgts[mask, 0] = np.nan
    wgts = sparse.COO.from_scipy_sparse(wgts)
    weights = xr.DataArray(data=wgts, dims=['to', 'from'], coords=coords)
    return weights


def _bounds_sanity_check(bounds):
    # Make sure bounds is 2D
    assert bounds.ndim == 2, f'Bounds must be a 2D array. Found bounds dimensions = {bounds.ndim}'
    # Make sure bounds shape is of form (n, 2)
    assert (
        bounds.shape[1] == 2
    ), f'Bounds must be a 2D array with shape: (n, 2). Found bounds shape = {bounds.shape}'
    # Make sure lower_i <= upper_i
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError('all lower bounds must be smaller than their counter-part upper bounds')
    # Make sure lower_i < lower_{i+1}
    if np.any(bounds[0, :-1] >= bounds[0, 1:]):
        raise ValueError('lower bound values must be monotonically increasing.')


def _data_ticks_sanity_check(data_ticks):
    assert data_ticks.ndim == 1, f'data ticks must be a one dimensional array.'
    message = 'data ticks must be monotically increasing.'
    assert np.all(data_ticks[:-1] < data_ticks[1:]), message
