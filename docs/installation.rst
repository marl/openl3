.. _installation:

Installation instructions
=========================

Dependencies
-----------------------
Tensorflow
__________
Because Tensorflow comes in CPU-only and GPU variants, we leave it up to the user to install the version that best fits
their usecase.

On most platforms, either of the following commands should properly install Tensorflow:

>>> pip install tensorflow # CPU-only version
>>> pip install tensorflow-gpu # GPU version

For more detailed information, please consult the
`Tensorflow installation documentation <https://www.tensorflow.org/install/>`_.

libsndfile
__________
``openl3`` depends on the `pysoundfile` module to load audio files, which depends on the non-Python library
``libsndfile``. On Windows and macOS, these will be installed via ``pip`` and you can therefore skip this step.
However, on Linux this must be installed manually via your platform's package manager.
For Debian-based distributions (such as Ubuntu), this can be done by simply running

>>> apt-get install libsndfile1

For more detailed information, please consult the
`pysoundfile installation documentation <https://pysoundfile.readthedocs.io/en/0.9.0/#installation>`_.


Installing openl3
-----------------
The simplest way to install ``openl3`` is by using ``pip``, which will also install the additional required dependencies
if needed. To install ``openl3`` using ``pip``, simply run

>>> pip install openl3

To install the latest version of openl3 from source:

1. Clone or pull the lastest version:

>>> git clone git@github.com:marl/openl3.git

2. Install using pip to handle python dependencies:

>>> cd openl3
>>> pip install -e .
