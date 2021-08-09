.. _installation:

Installation instructions
=========================

Dependencies
-----------------------

libsndfile
__________
OpenL3 depends on the ``pysoundfile`` module to load audio files, which depends on the non-Python library
``libsndfile``. On Windows and macOS, these will be installed via ``pip`` and you can therefore skip this step.
However, on Linux this must be installed manually via your platform's package manager.
For Debian-based distributions (such as Ubuntu), this can be done by simply running

>>> apt-get install libsndfile1

Alternatively, if you are using ``conda``, you can install ``libsndfile`` simply by running

>>> conda install -c conda-forge libsndfile

For more detailed information, please consult the
`pysoundfile installation documentation <https://pysoundfile.readthedocs.io/en/0.9.0/#installation>`_.

Tensorflow
__________
Starting with ``openl3>=0.4.0``, Openl3 has been upgraded to use Tensorflow 2. Because Tensorflow 2 and higher now includes GPU support, ``tensorflow>=2.0.0`` is included as a dependency and no longer needs to be installed separately. 

If you are interested in using Tensorflow 1.x, please install using ``pip install 'openl3<=0.3.1'``.

Tensorflow 1x & OpenL3 <= v0.3.1
********************************
Because Tensorflow 1.x comes in CPU-only and GPU variants, we leave it up to the user to install the version that best fits
their usecase.

On most platforms, either of the following commands should properly install Tensorflow:

>>> pip install "tensorflow<1.14" # CPU-only version
>>> pip install "tensorflow-gpu<1.14" # GPU version

Installing OpenL3
-----------------
The simplest way to install OpenL3 is by using ``pip``, which will also install the additional required dependencies
if needed. To install OpenL3 using ``pip``, simply run

>>> pip install openl3

To install the latest version of OpenL3 from source:

1. Clone or pull the latest version, only retrieving the ``main`` branch to avoid downloading the branch where we store the model weight files (these will be properly downloaded during installation).

>>> git clone git@github.com:marl/openl3.git --branch main --single-branch

2. Install using pip to handle python dependencies. The installation also downloads model files, which requires a stable network connection.

>>> cd openl3
>>> pip install -e .
