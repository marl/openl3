import os
import glob
import pickle
import numpy as np
import soundfile

TEST_DIR = os.path.dirname(__file__)
TEST_AUDIO_DIR = os.path.abspath(os.path.join(TEST_DIR, '../data', 'audio'))
CHIRP_44K_PATH = os.path.join(TEST_AUDIO_DIR, 'chirp_44k.wav')

files = [CHIRP_44K_PATH]

adj_file = lambda *fs: os.path.join(os.path.dirname(__file__), *fs)

def compute(input_repr="linear", content_type="music", embedding_size=6144, hop_size=1):
    '''Compute the outputs for all layers in a model using the current openl3 version.
    
    To switch versions, run this in a different python environment with your desired version
    installed.

    Currently we're doing this for the chirp 44k regression test file since that's what it was 
    failing on.

    '''
    import openl3
    model = openl3.models.load_audio_embedding_model(input_repr, content_type, embedding_size)
    model.summary()

    out_file = adj_file('layerbylayer-{}-{}-{}-{}.pkl'.format(
        openl3.__version__, input_repr, content_type, embedding_size))

    # use each intermediate step as an output (for comparison)
    print('building')
    outs = []
    y = model.input
    layers = model.layers[1:]
    names = [l.name for l in layers]
    for layer in layers:
        y = layer(y)
        outs.append(y)
    stepmodel = type(model)(model.inputs, outs)  # trying to not import keras/tf

    results = {}
    for f in files:
        y, sr = soundfile.read(f)
        print('Processing', f, sr, y.shape, '...')
        batches = openl3.core._preprocess_audio_batch(y, sr, hop_size=hop_size)
        print('batches:', batches.shape, batches.min(), batches.max())
        Z = stepmodel.predict(batches)
        for rep, name in zip(Z, names):
            print(name)
            print(rep.shape, [rep.min(), rep.max()], rep.mean())
        results[f] = Z
        print('-'*20)

    with open(out_file, 'wb') as f:
        pickle.dump({'files': results, 'names': names}, f)
    print('Wrote to', out_file)

def compare(results1_file, results2_file):
    '''Compare the results between two result files. Just need to use the minimum unique substring 
    you need to identify them with. But if you specify a path to an existing file, it will use that.

    e.g., say you have: 
        layerbylayer-0.3.1-linear-env-6144.pkl 
        layerbylayer-0.4.0-linear-env-6144.pkl 
        layerbylayer-0.3.1-mel256-env-6144.pkl
        layerbylayer-0.4.0-mel256-env-6144.pkl

    you can do: `python check_layer_by_layer.py 3.1-l 4.0-l` to compare the two
    
    '''
    results1, names1 = _load_results(results1_file)
    results2, names2 = _load_results(results2_file)
    
    print('Comparing:')
    print(results1_file)
    print(results2_file)
    print()

    for f in set(results1) | set(results2):
        print(f)
        for rep1, rep2, name1, name2 in zip(results1[f], results2[f], names1, names2):
            _compare(rep1, rep2, name1, name2)
        print()

def _compare(rep1, rep2, name1, name2):
    close = np.allclose(rep1, rep2, rtol=1e-05, atol=1e-05, equal_nan=False)
    print(name1, name2, 'allclose', close)
    print(rep1.shape, [rep1.min(), rep1.max()], rep1.mean())
    print(rep2.shape, [rep2.min(), rep2.max()], rep2.mean())

def show(results1_file, results2_file, fileindex=0, mapindex=0, limit=3, offset=0, clip=None):
    '''Plot the differences between the outputs of two versions.
    
    Arguments:
        results1_file (str): the first results file. See compare for more info.
        results2_file (str): the second results file. See compare for more info.
        fileindex (int): the index of the file to look at.
        mapindex (int): the index of the feature map to look at. By default it uses the 
            first one.
        limit (int): the number of layers to show. By default, first 3, Use `--nolimit` to show all.
        offset (int): the layer index to start at.
        clip (float): how much to clip off the values in the diff plot. Useful if you want to clip the 
            extreme values.
    '''
    import librosa.display
    import matplotlib.pyplot as plt
    results1, names1 = _load_results(results1_file)
    results2, names2 = _load_results(results2_file)
    version_names = [f.split('/')[-1] for f in (results1_file, results2_file)]

    f = files[fileindex]
    for rep1, rep2, name1, name2 in list(zip(results1[f], results2[f], names1, names2))[offset:offset+limit if limit else None]:
        rep1, rep2 = rep1[:,5:-30], rep2[:,5:-30]
        _compare(rep1, rep2, name1, name2)
        if rep1.ndim < 4:
            print(name1, name2, rep1.shape, 'too small for imshow')
            continue

        plt.figure(figsize=(15, 8))
        diff = np.abs(rep2 - rep1)
        for i, xs in enumerate(zip(rep1, rep2, np.clip(diff, -clip, clip) if clip else diff)):
            for j, (x, name, vname) in enumerate(zip(xs, (name1, name2,''), version_names+['diff'])):
                plt.subplot(len(rep1), len(xs), 1 + i*len(xs)+j)
                librosa.display.specshow(x[...,mapindex], cmap='magma')
                if i == 0:
                    # plt.title('{}:{} time={} feature_idx={}'.format(vname, name, i, mapindex))
                    plt.title('{}:{}'.format(vname, name))
                else:
                    plt.title('time={} feature_idx={}'.format(i, mapindex))
                plt.colorbar()
        plt.tight_layout()
        plt.show()

def _load_results(file, sep='+'):
    file = str(file)
    if not os.path.isfile(file):
        # find files matching query
        pattern = adj_file('*.pkl'.format(file))
        parts = file.split(sep)
        files = [f for f in glob.glob(pattern) if all(p in f for p in parts)]
        if not files:
            raise OSError('No files matching {} in {}'.format(file, pattern))
        elif len(files) > 1:
            raise OSError('Your query {} matched multiple files: {}. Add another distinguishing '
                          'substring. e.g. {}{}something-else'.format(file, files, file, sep))
        file = files[0]
    with open(file, 'rb') as f:
        results = pickle.load(f)
    return results['files'], results['names']

def _debugallclose(x1, x2, rtol=1e-04, atol=1e-04, **kw):
    passed = _allclose(x1, x2, rtol=rtol, atol=atol, **kw)
    if not passed:
        x1, x2 = np.asarray(x1), np.asarray(x2)
        print('shapes:', x1.shape, x2.shape)
        print('nans:', np.mean(np.isnan(x1)), np.mean(np.isnan(x2)))
        diff = np.abs(x2 - x1)
        print('amount above rtol:', np.mean(diff > rtol))
        print('min:', diff.min(), 'max:', diff.max(), 'mean:', diff.mean())
        print()

    return passed

_allclose = np.allclose
np.allclose = _debugallclose

if __name__ == '__main__':
    import fire
    fire.Fire({'compute': compute, 'compare': compare, 'show': show})