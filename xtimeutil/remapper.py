from .axis import Axis


class Remapper:
    def __init__(self, ds, freq, time_coord_name='time'):
        self._ds = ds
        self._from_axis = Axis(ds, time_coord_name)
        self.metadata = self._from_axis.metadata
        self.freq = freq
