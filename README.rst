
xtimeutil: Time Boundary-aware Operations with xarray
======================================================

|pypi| |conda forge| |Build Status| |codecov| |docs| |GitHub Workflow Status|


**xtimeutil** is a Python package for managing time boundary axis and different operations
related to time with xarray. xtimeutil consumes and produces xarray_ data structures,
which are coordinate and metadata-rich representations of multidimensional array data.


xtimeutil was motivated by the fact that xarray_ is not aware of the time boundary variable when
performing operations such as **resampling**. The main objective of xtimeutil is to provide a set of
utilities that enables fluid translation between data at different temporal intervals while being aware
of the time boundary variable.

The fundamental concept in xtimeutil is a `Remapper` object. `Remapper`'s role includes:

- **Generating** an outgoing time boundary axis when given information of the incoming time boundary axis.
- Using the incoming time boundary and outgoing time boundary axes to **generate remapping weights**.
- **Remapping** data on the incoming time boundary axis to the outgoing time boundary axis using generated weights.


For more information, read the full
`xtimeutil documentation`_.

Installation
------------

xtimeutil can be installed from PyPI with pip:

.. code-block:: bash

    python -m pip install xtimeutil


It is also available from `conda-forge` for conda installations:

.. code-block:: bash

    conda install -c conda-forge xtimeutil


.. _xarray: http://xarray.pydata.org
.. _xtimeutil documentation: https://xtimeutil.readthedocs.io

.. |GitHub Workflow Status| image:: https://img.shields.io/github/workflow/status/NCAR/xtimeutil/code-style?label=Code%20Style&style=for-the-badge
    :target: https://github.com/NCAR/xtimeutil/actions

.. |Build Status| image:: https://img.shields.io/circleci/project/github/NCAR/xtimeutil/master.svg?style=for-the-badge&logo=circleci
    :target: https://circleci.com/gh/NCAR/xtimeutil/tree/master

.. |codecov| image:: https://img.shields.io/codecov/c/github/NCAR/xtimeutil.svg?style=for-the-badge
    :target: https://codecov.io/gh/NCAR/xtimeutil

.. |docs| image:: https://img.shields.io/readthedocs/xtimeutil/latest.svg?style=for-the-badge
    :target: https://xtimeutil.readthedocs.io/en/latest/?badge=latest

.. |pypi| image:: https://img.shields.io/pypi/v/xtimeutil.svg?style=for-the-badge
    :target: https://pypi.org/project/xtimeutil

.. |conda forge| image:: https://img.shields.io/conda/vn/conda-forge/xtimeutil.svg?style=for-the-badge
    :target: https://anaconda.org/conda-forge/xtimeutil
