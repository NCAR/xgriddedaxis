#!/usr/bin/env python
""" Top-level module for xtime. """
from pkg_resources import DistributionNotFound, get_distribution

from .axis import Axis  # noqa: F401

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
