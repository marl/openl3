import openl3
import pytest
import tempfile
import numpy as np
import os
import shutil
import soundfile as sf
from openl3.openl3_exceptions import OpenL3Error
from openl3.openl3_warnings import OpenL3Warning


TEST_DIR = os.path.dirname(__file__)
TEST_AUDIO_DIR = os.path.join(TEST_DIR, 'data', 'audio')

# Test audio file paths
CHIRP_MONO_PATH = os.path.join(TEST_AUDIO_DIR, 'chirp_mono.wav')
CHIRP_STEREO_PATH = os.path.join(TEST_AUDIO_DIR, 'chirp_stereo.wav')
CHIRP_44K_PATH = os.path.join(TEST_AUDIO_DIR, 'chirp_44k.wav')
CHIRP_1S_PATH = os.path.join(TEST_AUDIO_DIR, 'chirp_1s.wav')
EMPTY_PATH = os.path.join(TEST_AUDIO_DIR, 'empty.wav')
SHORT_PATH = os.path.join(TEST_AUDIO_DIR, 'short.wav')
SILENCE_PATH = os.path.join(TEST_AUDIO_DIR, 'silence.wav')


def test_get_embedding():
    hop_size = 0.1
    tol = 1e-5

    # Make sure all embedding types work fine
    audio, sr = sf.read(CHIRP_MONO_PATH)
    emb1, ts1 = openl3.get_embedding(audio, sr,
        input_repr="mel256", content_type="music", embedding_size=512,
        center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = openl3.get_embedding(audio, sr,
        input_repr="mel128", content_type="music", embedding_size=512,
        center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = openl3.get_embedding(audio, sr,
        input_repr="mel128", content_type="music", embedding_size=6144,
        center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 6144
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = openl3.get_embedding(audio, sr,
        input_repr="linear", content_type="music", embedding_size=512,
        center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    audio, sr = sf.read(CHIRP_MONO_PATH)
    emb1, ts1 = openl3.get_embedding(audio, sr,
        input_repr="linear", content_type="music", embedding_size=6144,
        center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 6144
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = openl3.get_embedding(audio, sr,
        input_repr="mel256", content_type="env", embedding_size=512,
        center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = openl3.get_embedding(audio, sr,
        input_repr="mel256", content_type="env", embedding_size=6144,
        center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 6144
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = openl3.get_embedding(audio, sr,
        input_repr="mel128", content_type="env", embedding_size=512,
        center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = openl3.get_embedding(audio, sr,
        input_repr="mel128", content_type="env", embedding_size=6144,
        center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 6144
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = openl3.get_embedding(audio, sr,
        input_repr="linear", content_type="env", embedding_size=512,
        center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = openl3.get_embedding(audio, sr,
        input_repr="linear", content_type="env", embedding_size=6144,
        center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 6144
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = openl3.get_embedding(audio, sr,
        input_repr="mel256", content_type="music", embedding_size=6144,
        center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 6144
    assert not np.any(np.isnan(emb1))

    # Make sure that the embeddings are approximately the same with mono and stereo
    audio, sr = sf.read(CHIRP_STEREO_PATH)
    emb2, ts2 = openl3.get_embedding(audio, sr,
        input_repr="mel256", content_type="music", embedding_size=6144,
        center=True, hop_size=0.1, verbose=1)

    # assert np.all(np.abs(emb1 - emb2) < tol)
    # assert np.all(np.abs(ts1 - ts2) < tol)
    assert not np.any(np.isnan(emb2))

    # Make sure that the embeddings are approximately the same if we resample the audio
    audio, sr = sf.read(CHIRP_44K_PATH)
    emb3, ts3 = openl3.get_embedding(audio, sr,
        input_repr="mel256", content_type="music", embedding_size=6144,
        center=True, hop_size=0.1, verbose=1)

    # assert np.all(np.abs(emb1 - emb3) < tol)
    # assert np.all(np.abs(ts1 - ts3) < tol)
    assert not np.any(np.isnan(emb3))

    # Make sure empty audio is handled
    audio, sr = sf.read(EMPTY_PATH)
    pytest.raises(OpenL3Error, openl3.get_embedding, audio, sr,
        input_repr="mel256", content_type="music", embedding_size=6144,
        center=True, hop_size=0.1, verbose=1)

    # Make sure short audio can be handled
    audio, sr = sf.read(SHORT_PATH)
    emb4, ts4 = openl3.get_embedding(audio, sr,
        input_repr="mel256", content_type="music", embedding_size=6144,
        center=False, hop_size=0.1, verbose=1)

    assert emb4.shape[0] == 1
    assert emb4.shape[1] == 6144
    assert len(ts4) == 1
    assert ts4[0] == 0
    assert not np.any(np.isnan(emb4))

    # Make sure user is warned when audio is too short
    pytest.warns(OpenL3Warning, openl3.get_embedding, audio, sr,
        input_repr="mel256", content_type="music", embedding_size=6144,
        center=True, hop_size=0.1, verbose=1)

    # Make sure silence is handled
    audio, sr = sf.read(SILENCE_PATH)
    emb5, ts5 = openl3.get_embedding(audio, sr,
        input_repr="mel256", content_type="music", embedding_size=6144,
        center=True, hop_size=0.1, verbose=1)
    assert emb5.shape[1] == 6144
    assert not np.any(np.isnan(emb5))

    pytest.warns(OpenL3Warning, openl3.get_embedding, audio, sr,
        input_repr="mel256", content_type="music", embedding_size=6144,
        center=True, hop_size=0.1, verbose=1)

    # Check for centering
    audio, sr = sf.read(CHIRP_1S_PATH)
    emb6, ts6 = openl3.get_embedding(audio, sr,
        input_repr="mel256", content_type="music", embedding_size=6144,
        center=True, hop_size=hop_size, verbose=1)
    n_frames = 1 + int((audio.shape[0] + 0.5*sr - sr) / int(hop_size*sr))
    assert emb6.shape[0] == n_frames

    emb7, ts7 = openl3.get_embedding(audio, sr,
        input_repr="mel256", content_type="music", embedding_size=6144,
        center=False, hop_size=hop_size, verbose=1)
    n_frames = 1 + int((audio.shape[0] - sr) / int(hop_size*sr))
    assert emb7.shape[0] == n_frames

    # Check for hop size

    hop_size = 0.2
    emb8, ts8 = openl3.get_embedding(audio, sr,
        input_repr="mel256", content_type="music", embedding_size=6144,
        center=True, hop_size=hop_size, verbose=1)
    n_frames = 1 + int((audio.shape[0] + 0.5*sr - sr) / int(hop_size*sr))
    assert emb8.shape[0] == n_frames

    # Make sure changing verbosity doesn't break
    openl3.get_embedding(audio, sr,
        input_repr="mel256", content_type="music", embedding_size=6144,
        center=True, hop_size=hop_size, verbose=0)

    # Make sure invalid arguments don't work
    pytest.raises(OpenL3Error, openl3.get_embedding, audio, sr,
        input_repr="invalid", content_type="music", embedding_size=6144,
        center=True, hop_size=0.1, verbose=1)
    pytest.raises(OpenL3Error, openl3.get_embedding, audio, sr,
        input_repr="mel256", content_type="invalid", embedding_size=6144,
        center=True, hop_size=0.1, verbose=1)
    pytest.raises(OpenL3Error, openl3.get_embedding, audio, sr,
        input_repr="mel256", content_type="invalid", embedding_size=42,
        center=True, hop_size=0.1, verbose=1)
    pytest.raises(OpenL3Error, openl3.get_embedding, audio, sr,
        input_repr="mel256", content_type="music", embedding_size="invalid",
        center=True, hop_size=0.1, verbose=1)
    pytest.raises(OpenL3Error, openl3.get_embedding, audio, sr,
        input_repr="mel256", content_type="music", embedding_size=6144,
        center=True, hop_size=0, verbose=1)
    pytest.raises(OpenL3Error, openl3.get_embedding, audio, sr,
        input_repr="mel256", content_type="music", embedding_size=6144,
        center=True, hop_size=-1, verbose=1)
    pytest.raises(OpenL3Error, openl3.get_embedding, audio, sr,
        input_repr="mel256", content_type="music", embedding_size=6144,
        center=True, hop_size=0.1, verbose=-1)


def test_get_output_path():
    test_filepath = '/path/to/the/test/file/audio.wav'
    suffix = 'embedding.npy'
    test_output_dir = '/tmp/test/output/dir'
    exp_output_path = '/tmp/test/output/dir/audio_embedding.npy'
    output_path = openl3.get_output_path(test_filepath, suffix, test_output_dir)
    assert output_path == exp_output_path

    # No output directory
    exp_output_path = '/path/to/the/test/file/audio_embedding.npy'
    output_path = openl3.get_output_path(test_filepath, suffix)
    assert output_path == exp_output_path

    # No suffix
    exp_output_path = '/path/to/the/test/file/audio.npy'
    output_path = openl3.get_output_path(test_filepath, '.npy')
    assert output_path == exp_output_path


def test_process_file():
    test_output_dir = tempfile.mkdtemp()
    test_subdir = os.path.join(test_output_dir, "subdir")
    os.makedirs(test_subdir)

    # Make a copy of the file so we can test the case where we save to the same directory
    input_path_alt = os.path.join(test_subdir, "chirp_mono.wav")
    shutil.copy(CHIRP_MONO_PATH, test_subdir)

    exp_output_path1 = os.path.join(test_output_dir, "chirp_mono.npy")
    exp_output_path2 = os.path.join(test_output_dir, "chirp_mono_suffix.npy")
    exp_output_path3 = os.path.join(test_subdir, "chirp_mono.npy")
    try:
        openl3.process_file(CHIRP_MONO_PATH, output_dir=test_output_dir)
        openl3.process_file(CHIRP_MONO_PATH, output_dir=test_output_dir, suffix='suffix')
        openl3.process_file(input_path_alt)

        # Make sure paths all exist
        assert os.path.exists(exp_output_path1)
        assert os.path.exists(exp_output_path2)
        assert os.path.exists(exp_output_path3)

        data = np.load(exp_output_path1)
        assert 'embedding' in data
        assert 'timestamps' in data

        embedding = data['embedding']
        timestamps = data['timestamps']

        # Quick sanity check on data
        assert embedding.ndim == 2
        assert timestamps.ndim == 1

        # Make sure that suffices work
    finally:
        shutil.rmtree(test_output_dir)

    # Make sure we fail when file cannot be opened
    pytest.raises(OpenL3Error, openl3.process_file, '/fake/directory/asdf.wav')
