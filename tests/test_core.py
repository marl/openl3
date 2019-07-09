import openl3
import pytest
import tempfile
import numpy as np
import os
import shutil
import soundfile as sf
from skimage.io import imread
from openl3.openl3_exceptions import OpenL3Error
from openl3.openl3_warnings import OpenL3Warning


TEST_DIR = os.path.dirname(__file__)
TEST_AUDIO_DIR = os.path.join(TEST_DIR, 'data', 'audio')
TEST_IMAGE_DIR = os.path.join(TEST_DIR, 'data', 'image')

# Test audio file paths
CHIRP_MONO_PATH = os.path.join(TEST_AUDIO_DIR, 'chirp_mono.wav')
CHIRP_STEREO_PATH = os.path.join(TEST_AUDIO_DIR, 'chirp_stereo.wav')
CHIRP_44K_PATH = os.path.join(TEST_AUDIO_DIR, 'chirp_44k.wav')
CHIRP_1S_PATH = os.path.join(TEST_AUDIO_DIR, 'chirp_1s.wav')
EMPTY_PATH = os.path.join(TEST_AUDIO_DIR, 'empty.wav')
SHORT_PATH = os.path.join(TEST_AUDIO_DIR, 'short.wav')
SILENCE_PATH = os.path.join(TEST_AUDIO_DIR, 'silence.wav')

# Test image file paths
DAISY_PATH = os.path.join(TEST_IMAGE_DIR, 'daisy.jpg')
BLANK_PATH = os.path.join(TEST_IMAGE_DIR, 'blank.png')
SMALL_PATH = os.path.join(TEST_IMAGE_DIR, 'smol.jpg')
BENTO_PATH = os.path.join(TEST_IMAGE_DIR, 'bento.mp4')


def test_get_audio_embedding():
    hop_size = 0.1
    tol = 1e-5

    # Make sure all embedding types work fine
    audio, sr = sf.read(CHIRP_MONO_PATH)
    emb1, ts1 = openl3.get_audio_embedding(audio, sr,
                                           input_repr="mel256", content_type="music", embedding_size=512,
                                           center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = openl3.get_audio_embedding(audio, sr,
                                           input_repr="mel256", content_type="music", embedding_size=6144,
                                           center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 6144
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = openl3.get_audio_embedding(audio, sr,
                                           input_repr="mel128", content_type="music", embedding_size=512,
                                           center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = openl3.get_audio_embedding(audio, sr,
                                           input_repr="mel128", content_type="music", embedding_size=6144,
                                           center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 6144
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = openl3.get_audio_embedding(audio, sr,
                                           input_repr="linear", content_type="music", embedding_size=512,
                                           center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    audio, sr = sf.read(CHIRP_MONO_PATH)
    emb1, ts1 = openl3.get_audio_embedding(audio, sr,
                                           input_repr="linear", content_type="music", embedding_size=6144,
                                           center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 6144
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = openl3.get_audio_embedding(audio, sr,
                                           input_repr="mel256", content_type="env", embedding_size=512,
                                           center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = openl3.get_audio_embedding(audio, sr,
                                           input_repr="mel256", content_type="env", embedding_size=6144,
                                           center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 6144
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = openl3.get_audio_embedding(audio, sr,
                                           input_repr="mel128", content_type="env", embedding_size=512,
                                           center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = openl3.get_audio_embedding(audio, sr,
                                           input_repr="mel128", content_type="env", embedding_size=6144,
                                           center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 6144
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = openl3.get_audio_embedding(audio, sr,
                                           input_repr="linear", content_type="env", embedding_size=512,
                                           center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    emb1, ts1 = openl3.get_audio_embedding(audio, sr,
                                           input_repr="linear", content_type="env", embedding_size=6144,
                                           center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(np.diff(ts1) - hop_size) < tol)
    assert emb1.shape[1] == 6144
    assert not np.any(np.isnan(emb1))

    # Make sure we can load a model and pass it in
    model = openl3.models.load_audio_embedding_model("linear", "env", 6144)
    emb1load, ts1load = openl3.get_audio_embedding(audio, sr,
                                                   model=model, center=True, hop_size=hop_size, verbose=1)
    assert np.all(np.abs(emb1load - emb1) < tol)
    assert np.all(np.abs(ts1load - ts1) < tol)

    # Make sure that the embeddings are approximately the same with mono and stereo
    audio, sr = sf.read(CHIRP_STEREO_PATH)
    emb2, ts2 = openl3.get_audio_embedding(audio, sr,
                                           input_repr="mel256", content_type="music", embedding_size=6144,
                                           center=True, hop_size=0.1, verbose=1)

    # assert np.all(np.abs(emb1 - emb2) < tol)
    # assert np.all(np.abs(ts1 - ts2) < tol)
    assert not np.any(np.isnan(emb2))

    # Make sure that the embeddings are approximately the same if we resample the audio
    audio, sr = sf.read(CHIRP_44K_PATH)
    emb3, ts3 = openl3.get_audio_embedding(audio, sr,
                                           input_repr="mel256", content_type="music", embedding_size=6144,
                                           center=True, hop_size=0.1, verbose=1)

    # assert np.all(np.abs(emb1 - emb3) < tol)
    # assert np.all(np.abs(ts1 - ts3) < tol)
    assert not np.any(np.isnan(emb3))

    # Make sure empty audio is handled
    audio, sr = sf.read(EMPTY_PATH)
    pytest.raises(OpenL3Error, openl3.get_audio_embedding, audio, sr,
                  input_repr="mel256", content_type="music", embedding_size=6144,
                  center=True, hop_size=0.1, verbose=1)

    # Make sure user is warned when audio is too short
    audio, sr = sf.read(SHORT_PATH)
    pytest.warns(OpenL3Warning, openl3.get_audio_embedding, audio, sr,
                 input_repr="mel256", content_type="music", embedding_size=6144,
                 center=False, hop_size=0.1, verbose=1)

    # Make sure short audio can be handled
    emb4, ts4 = openl3.get_audio_embedding(audio, sr,
                                           input_repr="mel256", content_type="music", embedding_size=6144,
                                           center=False, hop_size=0.1, verbose=1)

    assert emb4.shape[0] == 1
    assert emb4.shape[1] == 6144
    assert len(ts4) == 1
    assert ts4[0] == 0
    assert not np.any(np.isnan(emb4))

    # Make sure silence is handled
    audio, sr = sf.read(SILENCE_PATH)
    pytest.warns(OpenL3Warning, openl3.get_audio_embedding, audio, sr,
                 input_repr="mel256", content_type="music", embedding_size=6144,
                 center=True, hop_size=0.1, verbose=1)

    emb5, ts5 = openl3.get_audio_embedding(audio, sr,
                                           input_repr="mel256", content_type="music", embedding_size=6144,
                                           center=True, hop_size=0.1, verbose=1)
    assert emb5.shape[1] == 6144
    assert not np.any(np.isnan(emb5))

    # Check for centering
    audio, sr = sf.read(CHIRP_1S_PATH)
    emb6, ts6 = openl3.get_audio_embedding(audio, sr,
                                           input_repr="mel256", content_type="music", embedding_size=6144,
                                           center=True, hop_size=hop_size, verbose=1)
    n_frames = 1 + int((audio.shape[0] + sr//2 - sr) / float(int(hop_size*sr)))
    assert emb6.shape[0] == n_frames

    emb7, ts7 = openl3.get_audio_embedding(audio, sr,
                                           input_repr="mel256", content_type="music", embedding_size=6144,
                                           center=False, hop_size=hop_size, verbose=1)
    n_frames = 1 + int((audio.shape[0] - sr) / float(int(hop_size*sr)))
    assert emb7.shape[0] == n_frames

    # Check for hop size
    hop_size = 0.2
    emb8, ts8 = openl3.get_audio_embedding(audio, sr,
                                           input_repr="mel256", content_type="music", embedding_size=6144,
                                           center=False, hop_size=hop_size, verbose=1)
    n_frames = 1 + int((audio.shape[0] - sr) / float(int(hop_size*sr)))
    assert emb8.shape[0] == n_frames

    # Make sure changing verbosity doesn't break
    openl3.get_audio_embedding(audio, sr,
                               input_repr="mel256", content_type="music", embedding_size=6144,
                               center=True, hop_size=hop_size, verbose=0)

    # Make sure invalid arguments don't work
    pytest.raises(OpenL3Error, openl3.get_audio_embedding, audio, sr,
                  model="invalid", center=True, hop_size=0.1, verbose=1)
    pytest.raises(OpenL3Error, openl3.get_audio_embedding, audio, "invalid",
                  input_repr="mel256", content_type="music", embedding_size=6144,
                  center=True, hop_size=0.1, verbose=1)
    pytest.raises(OpenL3Error, openl3.get_audio_embedding, audio, sr,
                  input_repr="invalid", content_type="music", embedding_size=6144,
                  center=True, hop_size=0.1, verbose=1)
    pytest.raises(OpenL3Error, openl3.get_audio_embedding, audio, sr,
                  input_repr="mel256", content_type="invalid", embedding_size=6144,
                  center=True, hop_size=0.1, verbose=1)
    pytest.raises(OpenL3Error, openl3.get_audio_embedding, audio, sr,
                  input_repr="mel256", content_type="invalid", embedding_size=42,
                  center=True, hop_size=0.1, verbose=1)
    pytest.raises(OpenL3Error, openl3.get_audio_embedding, audio, sr,
                  input_repr="mel256", content_type="music", embedding_size="invalid",
                  center=True, hop_size=0.1, verbose=1)
    pytest.raises(OpenL3Error, openl3.get_audio_embedding, audio, sr,
                  input_repr="mel256", content_type="music", embedding_size=6144,
                  center=True, hop_size=0, verbose=1)
    pytest.raises(OpenL3Error, openl3.get_audio_embedding, audio, sr,
                  input_repr="mel256", content_type="music", embedding_size=6144,
                  center=True, hop_size=-1, verbose=1)
    pytest.raises(OpenL3Error, openl3.get_audio_embedding, audio, sr,
                  input_repr="mel256", content_type="music", embedding_size=6144,
                  center=True, hop_size=0.1, verbose=-1)
    pytest.raises(OpenL3Error, openl3.get_audio_embedding, audio, sr,
                  input_repr="mel256", content_type="music", embedding_size=6144,
                  center='invalid', hop_size=0.1, verbose=1)
    pytest.raises(OpenL3Error, openl3.get_audio_embedding, np.ones((10, 10, 10)), sr,
                  input_repr="mel256", content_type="music", embedding_size=6144,
                  center=True, hop_size=0.1, verbose=1)


def test_get_image_embedding():
    frame_rate = 24
    tol = 1e-5

    image = imread(DAISY_PATH)
    image_tile = np.tile(image, (10, 1, 1, 1))

    # Make sure we get correct timestamps if we pass in a sequence and a frame
    # rate
    emb1, ts1 = openl3.get_image_embedding(image_tile, frame_rate=frame_rate,
                                           input_repr="mel256", content_type="music",
                                           embedding_size=512, verbose=1)
    assert emb1.shape[1] == 512
    assert np.all(np.abs(np.diff(ts1) - 1.0/frame_rate) < tol)
    assert not np.any(np.isnan(emb1))

    # Make sure all embedding types work fine
    emb1 = openl3.get_image_embedding(image,
                                      input_repr="mel256", content_type="music",
                                      embedding_size=512, verbose=1)
    assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    emb1 = openl3.get_image_embedding(image,
                                      input_repr="mel256", content_type="music",
                                      embedding_size=8192, verbose=1)
    assert emb1.shape[1] == 8192
    assert not np.any(np.isnan(emb1))

    emb1 = openl3.get_image_embedding(image,
                                      input_repr="mel128", content_type="music",
                                      embedding_size=512, verbose=1)
    assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    emb1 = openl3.get_image_embedding(image,
                                      input_repr="mel128", content_type="music",
                                      embedding_size=8192, verbose=1)
    assert emb1.shape[1] == 8192
    assert not np.any(np.isnan(emb1))

    emb1 = openl3.get_image_embedding(image,
                                      input_repr="linear", content_type="music",
                                      embedding_size=512, verbose=1)
    assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    emb1 = openl3.get_image_embedding(image,
                                      input_repr="linear", content_type="music",
                                      embedding_size=8192, verbose=1)
    assert emb1.shape[1] == 8192
    assert not np.any(np.isnan(emb1))

    emb1 = openl3.get_image_embedding(image,
                                      input_repr="mel256", content_type="env",
                                      embedding_size=512, verbose=1)
    assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    emb1 = openl3.get_image_embedding(image,
                                      input_repr="mel256", content_type="env",
                                      embedding_size=8192, verbose=1)
    assert emb1.shape[1] == 8192
    assert not np.any(np.isnan(emb1))

    emb1 = openl3.get_image_embedding(image,
                                      input_repr="mel128", content_type="env",
                                      embedding_size=512, verbose=1)
    assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    emb1 = openl3.get_image_embedding(image,
                                      input_repr="mel128", content_type="env",
                                      embedding_size=8192, verbose=1)
    assert emb1.shape[1] == 8192
    assert not np.any(np.isnan(emb1))

    emb1 = openl3.get_image_embedding(image,
                                      input_repr="linear", content_type="env",
                                      embedding_size=512, verbose=1)
    assert emb1.shape[1] == 512
    assert not np.any(np.isnan(emb1))

    emb1 = openl3.get_image_embedding(image,
                                      input_repr="linear", content_type="env",
                                      embedding_size=8192, verbose=1)
    assert emb1.shape[1] == 8192
    assert not np.any(np.isnan(emb1))

    # Make sure we can load a model and pass it in
    model = openl3.models.load_image_embedding_model("linear", "env", 8192)
    emb1load = openl3.get_image_embedding(image, model=model, verbose=1)
    assert np.all(np.abs(emb1load - emb1) < tol)

    # Make sure blank image is handled
    image = imread(BLANK_PATH)
    pytest.warns(OpenL3Warning, openl3.get_image_embedding, image,
                 input_repr="mel256", content_type="music",
                 embedding_size=8192, verbose=1)

    # Make sure user is warned when image is too small
    image = imread(SMALL_PATH)
    pytest.warns(OpenL3Warning, openl3.get_image_embedding, image,
                 input_repr="mel256", content_type="music",
                 embedding_size=8192, verbose=1)

    # Make sure changing verbosity doesn't break
    openl3.get_image_embedding(image,
                               input_repr="mel256", content_type="music",
                               embedding_size=8192, verbose=0)

    # Make sure invalid arguments don't work
    pytest.raises(OpenL3Error, openl3.get_image_embedding, image,
                  model="invalid", verbose=1)
    pytest.raises(OpenL3Error, openl3.get_image_embedding, image,
                  frame_rate="invalid", input_repr="mel256",
                  content_type="music", embedding_size=8192, verbose=1)
    pytest.raises(OpenL3Error, openl3.get_image_embedding, image,
                  input_repr="invalid", content_type="music",
                  embedding_size=8192, verbose=1)
    pytest.raises(OpenL3Error, openl3.get_image_embedding, image,
                  input_repr="mel256", content_type="invalid",
                  embedding_size=8192, verbose=1)
    pytest.raises(OpenL3Error, openl3.get_image_embedding, image,
                  input_repr="mel256", content_type="invalid",
                  embedding_size=42, verbose=1)
    pytest.raises(OpenL3Error, openl3.get_image_embedding, image,
                  input_repr="mel256", content_type="music",
                  embedding_size="invalid", verbose=1)
    pytest.raises(OpenL3Error, openl3.get_image_embedding, image,
                  input_repr="mel256", content_type="music",
                  embedding_size=8192, verbose=1)
    pytest.raises(OpenL3Error, openl3.get_image_embedding, image,
                  input_repr="mel256", content_type="music",
                  embedding_size=8192, verbose=-1)
    pytest.raises(OpenL3Error, openl3.get_image_embedding,
                  np.ones((10, 10)), input_repr="mel256",
                  content_type="music", embedding_size=8192, verbose=1)
    pytest.raises(OpenL3Error, openl3.get_image_embedding,
                  np.ones((10, 10, 10)), input_repr="mel256",
                  content_type="music", embedding_size=8192, verbose=1)
    pytest.raises(OpenL3Error, openl3.get_image_embedding,
                  np.ones((10, 10, 10, 10)), input_repr="mel256",
                  content_type="music", embedding_size=8192, verbose=1)
    pytest.raises(OpenL3Error, openl3.get_image_embedding,
                  np.ones((10, 10, 10, 10, 10)), input_repr="mel256",
                  content_type="music", embedding_size=8192, verbose=1)


def test_get_output_path():
    test_filepath = '/path/to/the/test/file/audio.wav'
    suffix = 'embedding.npz'
    test_output_dir = '/tmp/test/output/dir'
    exp_output_path = '/tmp/test/output/dir/audio_embedding.npz'
    output_path = openl3.get_output_path(test_filepath, suffix, test_output_dir)
    assert output_path == exp_output_path

    # No output directory
    exp_output_path = '/path/to/the/test/file/audio_embedding.npz'
    output_path = openl3.get_output_path(test_filepath, suffix)
    assert output_path == exp_output_path

    # No suffix
    exp_output_path = '/path/to/the/test/file/audio.npz'
    output_path = openl3.get_output_path(test_filepath, '.npz')
    assert output_path == exp_output_path


def test_process_audio_file():
    test_output_dir = tempfile.mkdtemp()
    test_subdir = os.path.join(test_output_dir, "subdir")
    os.makedirs(test_subdir)

    # Make a copy of the file so we can test the case where we save to the same directory
    input_path_alt = os.path.join(test_subdir, "chirp_mono.wav")
    shutil.copy(CHIRP_MONO_PATH, test_subdir)

    invalid_file_path = os.path.join(test_subdir, "invalid.wav")
    with open(invalid_file_path, 'w') as f:
        f.write('This is not an audio file.')

    exp_output_path1 = os.path.join(test_output_dir, "chirp_mono.npz")
    exp_output_path2 = os.path.join(test_output_dir, "chirp_mono_suffix.npz")
    exp_output_path3 = os.path.join(test_subdir, "chirp_mono.npz")
    try:
        openl3.process_audio_file(CHIRP_MONO_PATH, output_dir=test_output_dir)
        openl3.process_audio_file(CHIRP_MONO_PATH, output_dir=test_output_dir, suffix='suffix')
        openl3.process_audio_file(input_path_alt)

        # Make sure we fail when invalid files are provided
        pytest.raises(OpenL3Error, openl3.process_audio_file, invalid_file_path)

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
    pytest.raises(OpenL3Error, openl3.process_audio_file, '/fake/directory/asdf.wav')


def test_process_image_file():
    test_output_dir = tempfile.mkdtemp()
    test_subdir = os.path.join(test_output_dir, "subdir")
    os.makedirs(test_subdir)

    # Make a copy of the file so we can test the case where we save to the same directory
    input_path_alt = os.path.join(test_subdir, "daisy.jpg")
    shutil.copy(DAISY_PATH, test_subdir)

    invalid_file_path = os.path.join(test_subdir, "invalid.jpg")
    with open(invalid_file_path, 'w') as f:
        f.write('This is not an image file.')

    exp_output_path1 = os.path.join(test_output_dir, "daisy.npz")
    exp_output_path2 = os.path.join(test_output_dir, "daisy_suffix.npz")
    exp_output_path3 = os.path.join(test_subdir, "daisy.npz")
    try:
        openl3.process_image_file(DAISY_PATH, output_dir=test_output_dir)
        openl3.process_image_file(DAISY_PATH, output_dir=test_output_dir, suffix='suffix')
        openl3.process_image_file(input_path_alt)

        # Make sure we fail when invalid files are provided
        pytest.raises(OpenL3Error, openl3.process_image_file, invalid_file_path)

        # Make sure paths all exist
        assert os.path.exists(exp_output_path1)
        assert os.path.exists(exp_output_path2)
        assert os.path.exists(exp_output_path3)

        data = np.load(exp_output_path1)
        assert 'embedding' in data

        embedding = data['embedding']

        # Quick sanity check on data
        assert embedding.ndim == 2

        # Make sure that suffices work
    finally:
        shutil.rmtree(test_output_dir)

    # Make sure we fail when file cannot be opened
    pytest.raises(OpenL3Error, openl3.process_image_file, '/fake/directory/asdf.jpg')


def test_process_video_file():
    test_output_dir = tempfile.mkdtemp()
    test_subdir = os.path.join(test_output_dir, "subdir")
    os.makedirs(test_subdir)

    # Make a copy of the file so we can test the case where we save to the same directory
    input_path_alt = os.path.join(test_subdir, "bento.mp4")
    shutil.copy(BENTO_PATH, test_subdir)

    invalid_file_path = os.path.join(test_subdir, "invalid.mp4")
    with open(invalid_file_path, 'w') as f:
        f.write('This is not an video file.')

    exp_audio_output_path1 = os.path.join(test_output_dir, "bento_audio.npz")
    exp_audio_output_path2 = os.path.join(test_output_dir, "bento_audio_suffix.npz")
    exp_audio_output_path3 = os.path.join(test_subdir, "bento_audio.npz")
    exp_image_output_path1 = os.path.join(test_output_dir, "bento_image.npz")
    exp_image_output_path2 = os.path.join(test_output_dir, "bento_image_suffix.npz")
    exp_image_output_path3 = os.path.join(test_subdir, "bento_image.npz")
    try:
        openl3.process_video_file(BENTO_PATH, output_dir=test_output_dir)
        openl3.process_video_file(BENTO_PATH, output_dir=test_output_dir, suffix='suffix')
        openl3.process_video_file(input_path_alt)

        # Make sure we fail when invalid files are provided
        pytest.raises(OpenL3Error, openl3.process_video_file, invalid_file_path)

        # Make sure paths all exist
        assert os.path.exists(exp_audio_output_path1)
        assert os.path.exists(exp_audio_output_path2)
        assert os.path.exists(exp_audio_output_path3)
        assert os.path.exists(exp_image_output_path1)
        assert os.path.exists(exp_image_output_path2)
        assert os.path.exists(exp_image_output_path3)

        audio_data = np.load(exp_audio_output_path1)
        assert 'embedding' in audio_data
        assert 'timestamps' in audio_data

        audio_embedding = audio_data['embedding']

        # Quick sanity check on data
        assert audio_embedding.ndim == 2

        image_data = np.load(exp_image_output_path1)
        assert 'embedding' in image_data
        assert 'timestamps' in image_data

        image_embedding = image_data['embedding']

        # Quick sanity check on data
        assert image_embedding.ndim == 2

    finally:
        shutil.rmtree(test_output_dir)

    # Make sure we fail when file cannot be opened
    pytest.raises(OpenL3Error, openl3.process_video_file, '/fake/directory/asdf.mp4')


def test_center_audio():
    audio_len = 100
    audio = np.ones((audio_len,))

    # Test even window size
    frame_len = 50
    centered = openl3.core._center_audio(audio, frame_len)
    assert centered.size == 125
    assert np.all(centered[:25] == 0)
    assert np.array_equal(audio, centered[25:])

    # Test odd window size
    frame_len = 49
    centered = openl3.core._center_audio(audio, frame_len)
    assert centered.size == 124
    assert np.all(centered[:24] == 0)
    assert np.array_equal(audio, centered[24:])


def test_pad_audio():
    frame_len = 50
    hop_len = 25

    # Test short case
    audio_len = 10
    audio = np.ones((audio_len,))
    padded = openl3.core._pad_audio(audio, frame_len, hop_len)
    assert padded.size == 50
    assert np.array_equal(padded[:10], audio)
    assert np.all(padded[10:] == 0)

    # Test case when audio needs to be padded so all samples are processed
    audio_len = 90
    audio = np.ones((audio_len,))
    padded = openl3.core._pad_audio(audio, frame_len, hop_len)
    assert padded.size == 100
    assert np.array_equal(padded[:90], audio)
    assert np.all(padded[90:] == 0)

    # Test case when audio does not need padding
    audio_len = 100
    audio = np.ones((audio_len,))
    padded = openl3.core._pad_audio(audio, frame_len, hop_len)
    assert padded.size == 100
    assert np.array_equal(padded, audio)


def test_preprocess_image_batch():
    # Test a single large image
    single_big_img_arr = np.random.random((512, 1024, 3)) * 2 - 1
    batch = openl3.core._preprocess_image_batch(single_big_img_arr)
    assert batch.ndim == 4
    assert batch.shape[1:] == (224, 224, 3)
    assert batch.shape[0] == 1
    assert batch.dtype == np.float32
    assert (batch <= 1).all() and (batch >= -1).all()

    # Test a batch of large images
    batch_big_img_arr = np.random.random((10, 512, 1024, 3)) * 2 - 1
    batch = openl3.core._preprocess_image_batch(batch_big_img_arr)
    assert batch.ndim == 4
    assert batch.shape[1:] == (224, 224, 3)
    assert batch.shape[0] == batch_big_img_arr.shape[0]
    assert batch.dtype == np.float32
    assert (batch <= 1).all() and (batch >= -1).all()

    # Test a single image that is smaller than 256x256 but greater than 224x224
    single_img_arr = np.random.random((240, 240, 3)) * 2 - 1
    batch = openl3.core._preprocess_image_batch(single_img_arr)
    assert batch.ndim == 4
    assert batch.shape[1:] == (224, 224, 3)
    assert batch.shape[0] == 1
    assert batch.dtype == np.float32
    assert (batch <= 1).all() and (batch >= -1).all()
    assert np.allclose(batch[0], single_img_arr[8:-8, 8:-8, :])


    # Test a single image that is the correct size
    single_img_arr = np.random.random((224, 224, 3)) * 2 - 1
    batch = openl3.core._preprocess_image_batch(single_img_arr)
    assert batch.ndim == 4
    assert batch.shape[1:] == (224, 224, 3)
    assert batch.shape[0] == 1
    assert batch.dtype == np.float32
    assert (batch <= 1).all() and (batch >= -1).all()
    assert np.allclose(batch[0], single_img_arr)

    # Test a single image that is the correct size and is encoded as uint8
    single_img_int_arr = np.random.randint(256, size=(224, 224, 3), dtype='uint8')
    batch = openl3.core._preprocess_image_batch(single_img_int_arr)
    assert batch.ndim == 4
    assert batch.shape[1:] == (224, 224, 3)
    assert batch.shape[0] == 1
    assert (batch <= 1).all() and (batch >= -1).all()
    assert batch.dtype == np.float32
