import os
import glob
import h5py


def run(*files, search='spectrogram', doit=False):
    '''Remove certain layers from a model's stored HDF5 weight files.

    Basically, I was getting the error:
    `you're trying to load a weights file with 17 layers into a model with 16 layers`

    Which was basically because there was an extra `spectrogram` layer that wasn't there
    in the new version of `kapre`.

    Since auto-incrementing weight naming is a bit different between versions, 
    we couldn't just do "by_name".
    
    Arguments:
        *files (str): weight files to check.
        search (str): the search term to use when looking thru layer names.
        doit (bool): If this is not set to True, then it will just print 
            out the layers that it finds. This is a precaution to prevent 
            you from accidentally deleting weights.
    '''
    # extra .. gets rid of fname
    pattern = os.path.abspath(os.path.join(
        __file__, '../../', 
        'openl3/*.h5'))

    deleted = {}
    for fname in files or glob.glob(pattern):
        print()
        print('Doing things to', fname, '...')
        with h5py.File(fname, 'a') as hdf:
            # finding any spectrogram layers
            groups = {False: [], True: []}
            for l in hdf.attrs['layer_names']:
                groups[search.encode('utf-8') in l].append(l)

            print('Matched layers', groups[True])
            if not doit:
                continue

            # so that keras doesn't remember
            hdf.attrs['layer_names'] = groups[False]
            # sayonara spectrogram weights
            for name in groups[True]:
                print('deleting weights for:', name)
                del hdf[name]

        deleted[fname] = groups[True]
    return deleted


if __name__ == '__main__':
    import fire
    fire.Fire(run)