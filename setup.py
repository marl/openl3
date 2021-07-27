import os
import sys
import gzip
import imp
from itertools import product
from setuptools import setup, find_packages

from urllib.request import urlretrieve

module_dir = 'openl3'
modalities = ['audio', 'image']
input_reprs = ['linear', 'mel128', 'mel256']
content_type = ['music', 'env']
model_version_str = 'v0_4_0'
weight_files = ['openl3_{}_{}_{}.h5'.format(*tup)
                for tup in product(modalities, input_reprs, content_type)]
base_url = 'https://github.com/marl/openl3/raw/models/'

if len(sys.argv) > 1 and sys.argv[1] == 'sdist':
    # exclude the weight files in sdist
    weight_files = []
else:
    # in all other cases, decompress the weights file if necessary
    for weight_file in weight_files:
        weight_path = os.path.join(module_dir, weight_file)
        if not os.path.isfile(weight_path):
            weight_fname = os.path.splitext(weight_file)[0]
            compressed_file = '{}-{}.h5.gz'.format(weight_fname, model_version_str)
            compressed_path = os.path.join(module_dir, compressed_file)
            if not os.path.isfile(compressed_file):
                print('Downloading weight file {} ...'.format(compressed_file))
                urlretrieve(base_url + compressed_file, compressed_path)
            print('Decompressing ...')
            with gzip.open(compressed_path, 'rb') as source:
                with open(weight_path, 'wb') as target:
                    target.write(source.read())
            print('Decompression complete')
            os.remove(compressed_path)
            print('Removing compressed file')

version = imp.load_source('openl3.version', os.path.join('openl3', 'version.py'))

with open('README.md') as file:
    long_description = file.read()

setup(
    name='openl3',
    version=version.version,
    description='Deep audio and image embeddings, based on Look, Listen, and Learn approach',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/marl/openl3',
    author='Jason Cramer, Ho-Hsiang Wu, Bea Steers, and Justin Salamon',
    author_email='jtcramer@nyu.edu',
    packages=find_packages(),
    entry_points={
        'console_scripts': ['openl3=openl3.cli:main'],
    },
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='deep audio embeddings machine listening learning tensorflow keras',
    project_urls={
        'Source': 'https://github.com/marl/openl3',
        'Tracker': 'https://github.com/marl/openl3/issues',
        'Documentation': 'https://readthedocs.org/projects/openl3/'
    },
    install_requires=[
        'tensorflow>=2.0.0',
        'numpy>=1.13.0',
        'scipy>=0.19.1',
        'kapre>=0.3.5',
        'soundfile>=0.9.0.post1',
        'resampy>=0.2.1,<0.3.0',
        'h5py>=2.7.0,<3.0.0',
        'moviepy>=1.0.0',
        'scikit-image>=0.14.3'
        'librosa>=0.7.2',  # version limit from kapre
    ],
    extras_require={
        'docs': [
                'sphinx==1.2.3',  # autodoc was broken in 1.3.1
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
        'tests': []
    },
    package_data={
        'openl3': weight_files
    },
)
