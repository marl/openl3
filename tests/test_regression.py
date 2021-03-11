import os
from openl3.cli import run
import tempfile
import numpy as np
import shutil


TEST_DIR = os.path.dirname(__file__)
TEST_AUDIO_DIR = os.path.join(TEST_DIR, 'data', 'audio')
TEST_IMAGE_DIR = os.path.join(TEST_DIR, 'data', 'image')
TEST_VIDEO_DIR = os.path.join(TEST_DIR, 'data', 'video')

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
SMALL_PATH = os.path.join(TEST_IMAGE_DIR, 'smol.png')

# Test video file paths
BENTO_PATH = os.path.join(TEST_VIDEO_DIR, 'bento.mp4')

# Regression file paths
TEST_REG_DIR = os.path.join(TEST_DIR, 'data', 'regression')
REG_CHIRP_44K_PATH = os.path.join(TEST_REG_DIR, 'chirp_44k.npz')
REG_CHIRP_44K_LINEAR_PATH = os.path.join(TEST_REG_DIR, 'chirp_44k_linear.npz')
REG_DAISY_PATH = os.path.join(TEST_REG_DIR, 'daisy.npz')
REG_DAISY_LINEAR_PATH = os.path.join(TEST_REG_DIR, 'daisy_linear.npz')
REG_BENTO_AUDIO_PATH = os.path.join(TEST_REG_DIR, 'bento_audio.npz')
REG_BENTO_AUDIO_LINEAR_PATH = os.path.join(TEST_REG_DIR, 'bento_audio_linear.npz')
REG_BENTO_IMAGE_PATH = os.path.join(TEST_REG_DIR, 'bento_image.npz')
REG_BENTO_IMAGE_LINEAR_PATH = os.path.join(TEST_REG_DIR, 'bento_image_linear.npz')


def test_audio_regression(capsys):
    # test correct execution on test audio file (regression)
    tempdir = tempfile.mkdtemp()
    run('audio', CHIRP_44K_PATH, output_dir=tempdir, verbose=True)

    # check output file created
    audio_outfile = os.path.join(tempdir, 'chirp_44k.npz')
    assert os.path.isfile(audio_outfile)

    # regression test
    audio_data_reg = np.load(REG_CHIRP_44K_PATH)
    audio_data_out = np.load(audio_outfile)
    assert sorted(audio_data_out.files) == sorted(audio_data_reg.files) == sorted(
        ['embedding', 'timestamps'])
    assert np.allclose(audio_data_out['timestamps'], audio_data_reg['timestamps'],
                       rtol=1e-05, atol=1e-05, equal_nan=False)
    assert np.allclose(audio_data_out['embedding'], audio_data_reg['embedding'],
                       rtol=1e-05, atol=1e-05, equal_nan=False)

    # SECOND regression test
    run('audio', CHIRP_44K_PATH, output_dir=tempdir, suffix='linear', input_repr='linear',
        content_type='env', audio_embedding_size=512, audio_center=False, audio_hop_size=0.5,
        verbose=False)
    # check output file created
    audio_outfile = os.path.join(tempdir, 'chirp_44k_linear.npz')
    assert os.path.isfile(audio_outfile)

    # regression test
    audio_data_reg = np.load(REG_CHIRP_44K_LINEAR_PATH)
    audio_data_out = np.load(audio_outfile)
    assert sorted(audio_data_out.files) == sorted(audio_data_reg.files) == sorted(
        ['embedding', 'timestamps'])
    assert np.allclose(audio_data_out['timestamps'], audio_data_reg['timestamps'],
                       rtol=1e-05, atol=1e-05, equal_nan=False)
    assert np.allclose(audio_data_out['embedding'], audio_data_reg['embedding'],
                       rtol=1e-05, atol=1e-05, equal_nan=False)

    # delete output file and temp folder
    shutil.rmtree(tempdir)


def test_image_regression(capsys):
    # test correct execution on test image file (regression)
    tempdir = tempfile.mkdtemp()
    run('image', DAISY_PATH, output_dir=tempdir, verbose=True)

    # check output file created
    image_outfile = os.path.join(tempdir, 'daisy.npz')
    assert os.path.isfile(image_outfile)

    # regression test
    image_data_reg = np.load(REG_DAISY_PATH)
    image_data_out = np.load(image_outfile)
    assert sorted(image_data_out.files) == sorted(image_data_reg.files) == ['embedding']
    assert np.allclose(image_data_out['embedding'], image_data_reg['embedding'],
                       rtol=1e-05, atol=1e-05, equal_nan=False)

    # SECOND regression test
    run('image', DAISY_PATH, output_dir=tempdir, suffix='linear', input_repr='linear',
        content_type='env', image_embedding_size=512, verbose=False)
    # check output file created
    image_outfile = os.path.join(tempdir, 'daisy_linear.npz')
    assert os.path.isfile(image_outfile)

    # regression test
    image_data_reg = np.load(REG_DAISY_LINEAR_PATH)
    image_data_out = np.load(image_outfile)
    assert sorted(image_data_out.files) == sorted(image_data_reg.files) == ['embedding']
    assert np.allclose(image_data_out['embedding'], image_data_reg['embedding'],
                       rtol=1e-05, atol=1e-05, equal_nan=False)

    # delete output file and temp folder
    shutil.rmtree(tempdir)


def test_video_regression(capsys):
    tempdir = tempfile.mkdtemp()

    ## Video processing regression tests
    run('video', BENTO_PATH, output_dir=tempdir, verbose=True)

    # check output files created
    audio_outfile = os.path.join(tempdir, 'bento_audio.npz')
    assert os.path.isfile(audio_outfile)
    image_outfile = os.path.join(tempdir, 'bento_image.npz')
    assert os.path.isfile(image_outfile)

    # regression test
    audio_data_reg = np.load(REG_BENTO_AUDIO_PATH)
    audio_data_out = np.load(audio_outfile)
    image_data_reg = np.load(REG_BENTO_IMAGE_PATH)
    image_data_out = np.load(image_outfile)

    assert sorted(audio_data_out.files) == sorted(audio_data_reg.files) == sorted(
        ['embedding', 'timestamps'])
    assert np.allclose(audio_data_out['timestamps'], audio_data_reg['timestamps'],
                       rtol=1e-05, atol=1e-05, equal_nan=False)
    assert np.allclose(audio_data_out['embedding'], audio_data_reg['embedding'],
                       rtol=1e-05, atol=1e-05, equal_nan=False)

    assert sorted(image_data_out.files) == sorted(image_data_reg.files) == sorted(
        ['embedding', 'timestamps'])
    assert np.allclose(image_data_out['timestamps'], image_data_reg['timestamps'],
                       rtol=1e-05, atol=1e-05, equal_nan=False)
    assert np.allclose(image_data_out['embedding'], image_data_reg['embedding'],
                       rtol=1e-05, atol=1e-05, equal_nan=False)

    # SECOND regression test
    run('video', BENTO_PATH, output_dir=tempdir, suffix='linear', input_repr='linear',
        content_type='env', audio_embedding_size=512, image_embedding_size=512,
        audio_center=False, audio_hop_size=0.5, verbose=False)

    # check output files created
    audio_outfile = os.path.join(tempdir, 'bento_audio_linear.npz')
    assert os.path.isfile(audio_outfile)
    image_outfile = os.path.join(tempdir, 'bento_image_linear.npz')
    assert os.path.isfile(image_outfile)

    # regression test
    audio_data_reg = np.load(REG_BENTO_AUDIO_LINEAR_PATH)
    audio_data_out = np.load(audio_outfile)
    image_data_reg = np.load(REG_BENTO_IMAGE_LINEAR_PATH)
    image_data_out = np.load(image_outfile)

    assert sorted(audio_data_out.files) == sorted(audio_data_reg.files) == sorted(
        ['embedding', 'timestamps'])
    assert np.allclose(audio_data_out['timestamps'], audio_data_reg['timestamps'],
                       rtol=1e-05, atol=1e-05, equal_nan=False)
    assert np.allclose(audio_data_out['embedding'], audio_data_reg['embedding'],
                       rtol=1e-05, atol=1e-05, equal_nan=False)

    assert sorted(image_data_out.files) == sorted(image_data_reg.files) == sorted(
        ['embedding', 'timestamps'])
    assert np.allclose(image_data_out['timestamps'], image_data_reg['timestamps'],
                       rtol=1e-05, atol=1e-05, equal_nan=False)
    assert np.allclose(image_data_out['embedding'], image_data_reg['embedding'],
                       rtol=1e-05, atol=1e-05, equal_nan=False)

