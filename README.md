# OpenL3

OpenL3 is an open-source Python library for computing deep audio and image embeddings.

[![PyPI](https://img.shields.io/badge/python-2.7%2C%203.5%2C%203.6-blue.svg)](https://pypi.python.org/pypi/openl3)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://choosealicense.com/licenses/mit/)
[![Build Status](https://travis-ci.org/marl/openl3.svg?branch=master)](https://travis-ci.org/marl/openl3)
[![Coverage Status](https://coveralls.io/repos/github/marl/openl3/badge.svg?branch=master)](https://coveralls.io/github/marl/openl3?branch=master)
[![Documentation Status](https://readthedocs.org/projects/openl3/badge/?version=latest)](http://openl3.readthedocs.io/en/latest/?badge=latest)

Please refer to the [documentation](https://openl3.readthedocs.io/en/latest/) for detailed instructions and examples.

The audio and image embedding models provided here are published as part of [1], and are based on the Look, Listen and Learn approach [2]. For details about the embedding models and how they were trained, please see:

[Look, Listen and Learn More: Design Choices for Deep Audio Embeddings](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/cramer_looklistenlearnmore_icassp_2019.pdf)<br/>
Jason Cramer, Ho-Hsiang Wu, Justin Salamon, and Juan Pablo Bello.<br/>
IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP), pages 3852–3856, Brighton, UK, May 2019.

# Installing OpenL3

Dependencies
------------
#### Tensorflow
Because Tensorflow comes in CPU-only and GPU variants, we leave it up to the user to install the version that best fits
their usecase.

On most platforms, either of the following commands should properly install Tensorflow:

    pip install tensorflow # CPU-only version
    pip install tensorflow-gpu # GPU version

For more detailed information, please consult the
[Tensorflow installation documentation](https://www.tensorflow.org/install/).

#### libsndfile
OpenL3 depends on the `pysoundfile` module to load audio files, which depends on the non-Python library
``libsndfile``. On Windows and macOS, these will be installed via ``pip`` and you can therefore skip this step.
However, on Linux this must be installed manually via your platform's package manager.
For Debian-based distributions (such as Ubuntu), this can be done by simply running

    apt-get install libsndfile1

For more detailed information, please consult the
[`pysoundfile` installation documentation](https://pysoundfile.readthedocs.io/en/0.9.0/#installation>).


Installing OpenL3
-----------------
The simplest way to install OpenL3 is by using ``pip``, which will also install the additional required dependencies
if needed. To install OpenL3 using ``pip``, simply run

    pip install openl3

To install the latest version of OpenL3 from source:

1. Clone or pull the lastest version:

        git clone git@github.com:marl/openl3.git

2. Install using pip to handle python dependencies:

        cd openl3
        pip install -e .

# Using OpenL3

To help you get started with OpenL3 please see the
[tutorial](http://openl3.readthedocs.io/en/latest/tutorial.html).


# Acknowledging OpenL3

Please cite the following papers when using OpenL3 in your work:

[1] [Look, Listen and Learn More: Design Choices for Deep Audio Embeddings](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/cramer_looklistenlearnmore_icassp_2019.pdf)<br/>
Jason Cramer, Ho-Hsiang Wu, Justin Salamon, and Juan Pablo Bello.<br/>
IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP), pages 3852–3856, Brighton, UK, May 2019.

[2] Look, Listen and Learn<br/>
Relja Arandjelović and Andrew Zisserman<br/>
IEEE International Conference on Computer Vision (ICCV), Venice, Italy, Oct. 2017.
