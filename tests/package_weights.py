'''Package the .h5 weights files so that they are ready to be uploaded to the git models folder.

Assuming you set it up like this:

Clone the openl3 models branch (this is where we'll put the weights files)

    $ git clone https://... models/
    $ cd models
    $ git checkout models
    $ cd ..

Clone the openl3 code branch and install (this downloads the weights files)

    $ git clone https://... openl3/
    $ cd openl3
    $ pip install -e .

If you're upgrading from the v0.2.0 weights to the v0.4.0 weights, you'll want to remove 
the Spectrogram layers from the weight files. You can do it like this:

    $ # just as a safety check - make sure it's just showing the spectrogram/melspectrogram weights
    $ python tests/migration/remove_layers.py
    $ # actually patch the weights:
    $ python tests/migration/remove_layers.py --doit

Now to compress the weights files and put them into the models branch.

    $ python tests/package_weights.py --out-dir ../models

Now you just have to push up the changes to git! (`cd ../models && git add *.gz && ...`)

'''
import os
import glob
import gzip
import itertools

module_dir = 'openl3'
modalities = ['audio', 'image']
input_reprs = ['linear', 'mel128', 'mel256']
content_type = ['music', 'env']
model_version_str = 'v0_4_0'
weight_files = ['openl3_{}_{}_{}.h5'.format(*tup)
                for tup in itertools.product(modalities, input_reprs, content_type)]
base_url = 'https://github.com/marl/openl3/raw/models/'


# def decompress(version=model_version_str):
#     try:
#         from urllib.request import urlretrieve
#     except ImportError:
#         from urllib import urlretrieve
#     version = version.replace('.', '_')

#     # in all other cases, decompress the weights file if necessary
#     for weight_file in weight_files:
#         weight_path = os.path.join(module_dir, weight_file)
#         if os.path.isfile(weight_path):
#             continue

#         compressed_file = '{}-{}.h5.gz'.format(os.path.splitext(weight_file)[0], version)
#         compressed_path = os.path.join(module_dir, compressed_file)

#         if not os.path.isfile(compressed_file):
#             print('Downloading weight file {} ...'.format(compressed_file))
#             urlretrieve(base_url + compressed_file, compressed_path)

#         print('Decompressing ...')
#         with gzip.open(compressed_path, 'rb') as source:
#             with open(weight_path, 'wb') as target:
#                 target.write(source.read())
#         print('Decompression complete.')
#         os.remove(compressed_path)
#         print('Removed compressed file.')


def compress(*weight_files, out_dir=None, version=model_version_str, overwrite=False):
    '''Compress the weights using the same compression technique used to download the weights in setup.py.'''
    version = version.replace('.', '_')
    weight_files = weight_files or glob.glob(os.path.abspath(os.path.join(__file__, '../../openl3/*.h5')))
    out_dir = os.path.abspath(out_dir or os.path.join(__file__, '../../../models'))  # assume adjacent repo
    os.makedirs(out_dir, exist_ok=True)

    for weight_file in weight_files:
        if not overwrite and not os.path.isfile(weight_file):
            print('Weight file', weight_file, 'does not exist. skipping...')
            continue

        compressed_file = '{}-{}.h5.gz'.format(os.path.splitext(os.path.basename(weight_file))[0], version)
        compressed_path = os.path.join(out_dir, compressed_file)
        print('Compressing', weight_file, 'to', compressed_path)

        with open(weight_file, 'rb') as source:
            with gzip.open(compressed_path, 'wb') as target:
                target.write(source.read())
        print('Done!')

if __name__ == '__main__':
    import fire
    fire.Fire(compress)