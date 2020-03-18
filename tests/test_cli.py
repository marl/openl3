import pytest
import os
from openl3.cli import positive_int, positive_float, get_file_list, parse_args,\
                       run, main
from argparse import ArgumentTypeError
from openl3.openl3_exceptions import OpenL3Error
import tempfile
import numpy as np
try:
    # python 3.4+ should use builtin unittest.mock not mock package
    from unittest.mock import patch
except ImportError:
    from mock import patch


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

# Test video file paths
BENTO_PATH = os.path.join(TEST_VIDEO_DIR, 'bento.mp4')

# Regression file paths
TEST_REG_DIR = os.path.join(TEST_DIR, 'data', 'regression')
REG_CHIRP_44K_PATH = os.path.join(TEST_REG_DIR, 'chirp_44k.npz')
REG_CHIRP_44K_LINEAR_PATH = os.path.join(TEST_REG_DIR, 'chirp_44k_linear.npz')
REG_BENTO_AUDIO_PATH = os.path.join(TEST_REG_DIR, 'bento_audio.npz')
REG_BENTO_AUDIO_LINEAR_PATH = os.path.join(TEST_REG_DIR, 'bento_audio_linear.npz')
REG_BENTO_IMAGE_PATH = os.path.join(TEST_REG_DIR, 'bento_image.npz')
REG_BENTO_IMAGE_LINEAR_PATH = os.path.join(TEST_REG_DIR, 'bento_image_linear.npz')


def test_positive_float():

    # test that returned value is float
    f = positive_float(5)
    assert f == 5.0
    assert type(f) is float

    # test it works for valid strings
    f = positive_float('1.3')
    assert f == 1.3
    assert type(f) is float

    # make sure error raised for all invalid values:
    invalid = [-5, -1.0, None, 'hello']
    for i in invalid:
        pytest.raises(ArgumentTypeError, positive_float, i)


def test_positive_int():
    # test that returned value is int
    i = positive_int(5)
    assert i == 5
    assert type(i) is int

    i = positive_int(5.0)
    assert i == 5
    assert type(i) is int

    # test it works for valid strings
    i = positive_int('1')
    assert i == 1
    assert type(i) is int

    # make sure error raised for all invalid values:
    invalid = [-5, -1.0, None, 'hello']
    for i in invalid:
        pytest.raises(ArgumentTypeError, positive_int, i)


def test_get_file_list():

    # test for invalid input (must be iterable, e.g. list)
    pytest.raises(ArgumentTypeError, get_file_list, CHIRP_44K_PATH)

    # test for valid list of file paths
    flist = get_file_list([CHIRP_44K_PATH, CHIRP_1S_PATH])
    assert len(flist) == 2
    assert flist[0] == CHIRP_44K_PATH and flist[1] == CHIRP_1S_PATH

    # test for valid folder
    flist = get_file_list([TEST_AUDIO_DIR])
    assert len(flist) == 7

    flist = sorted(flist)
    assert flist[0] == CHIRP_1S_PATH
    assert flist[1] == CHIRP_44K_PATH
    assert flist[2] == CHIRP_MONO_PATH
    assert flist[3] == CHIRP_STEREO_PATH
    assert flist[4] == EMPTY_PATH
    assert flist[5] == SHORT_PATH
    assert flist[6] == SILENCE_PATH

    # combine list of files and folders
    flist = get_file_list([TEST_AUDIO_DIR, CHIRP_44K_PATH])
    assert len(flist) == 8

    # nonexistent path
    pytest.raises(OpenL3Error, get_file_list, ['/fake/path/to/file'])


def test_parse_args():

    # test for all the defaults
    args = ['audio', CHIRP_44K_PATH]
    args = parse_args(args)
    assert args.modality == 'audio'
    assert args.inputs == [CHIRP_44K_PATH]
    assert args.output_dir is None
    assert args.suffix is None
    assert args.input_repr == 'mel256'
    assert args.content_type == 'music'
    assert args.audio_embedding_size == 6144
    assert args.no_audio_centering is False
    assert args.audio_hop_size == 0.1
    assert args.image_embedding_size == 8192
    assert args.quiet is False

    # test when setting all values
    args = ['video', BENTO_PATH, '-o', '/output/dir', '--suffix', 'suffix',
            '--input-repr', 'linear', '--content-type', 'env',
            '--audio-embedding-size', '512', '--no-audio-centering',
            '--audio-hop-size', '0.5', '--image-embedding-size', '512',
            '--quiet']
    args = parse_args(args)
    assert args.inputs == [BENTO_PATH]
    assert args.output_dir == '/output/dir'
    assert args.suffix == 'suffix'
    assert args.input_repr == 'linear'
    assert args.content_type == 'env'
    assert args.audio_embedding_size == 512
    assert args.no_audio_centering is True
    assert args.audio_hop_size == 0.5
    assert args.image_embedding_size == 512
    assert args.quiet is True


def test_run(capsys):

    # test invalid input
    invalid = [None, 5, 1.0]
    for i in invalid:
        pytest.raises(OpenL3Error, run, i, i)

    # test empty input folder
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        tempdir = tempfile.mkdtemp()
        run('audio', [tempdir])

    # make sure it exited
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == -1

    # make sure it printed a message
    captured = capsys.readouterr()
    expected_message = 'openl3: No files found in {}. Aborting.\n'.format(str([tempdir]))
    assert captured.out == expected_message

    # delete tempdir
    if os.path.exists(tempdir):
        os.rmdir(tempdir)

    # test invalid modality
    with pytest.raises(OpenL3Error) as pytest_wrapped_e:
        tempdir = tempfile.mkdtemp()
        run('invalid', CHIRP_44K_PATH, output_dir=tempdir)

    # delete tempdir
    if os.path.exists(tempdir):
        os.rmdir(tempdir)


def test_main():

    # Duplicate audio regression test from test_run just to hit coverage
    tempdir = tempfile.mkdtemp()
    with patch('sys.argv', ['openl3', 'audio', CHIRP_44K_PATH, '--output-dir', tempdir]):
        main()

    # check output file created
    outfile = os.path.join(tempdir, 'chirp_44k.npz')
    assert os.path.isfile(outfile)

    # regression test
    data_reg = np.load(REG_CHIRP_44K_PATH)
    data_out = np.load(outfile)

    assert sorted(data_out.files) == sorted(data_reg.files) == sorted(
        ['embedding', 'timestamps'])
    assert np.allclose(data_out['timestamps'], data_reg['timestamps'],
                       rtol=1e-05, atol=1e-05, equal_nan=False)
    assert np.allclose(data_out['embedding'], data_reg['embedding'],
                       rtol=1e-03, atol=1e-03, equal_nan=False)


def test_script_main():

    # Duplicate audio regression test from test_run just to hit coverage
    tempdir = tempfile.mkdtemp()
    with patch('sys.argv', ['openl3', 'audio', CHIRP_44K_PATH, '--output-dir', tempdir]):
        import openl3.__main__

    # check output file created
    outfile = os.path.join(tempdir, 'chirp_44k.npz')
    assert os.path.isfile(outfile)

    # regression test
    data_reg = np.load(REG_CHIRP_44K_PATH)
    data_out = np.load(outfile)

    assert sorted(data_out.files) == sorted(data_reg.files) == sorted(
        ['embedding', 'timestamps'])
    assert np.allclose(data_out['timestamps'], data_reg['timestamps'],
                       rtol=1e-05, atol=1e-05, equal_nan=False)
    assert np.allclose(data_out['embedding'], data_reg['embedding'],
                       rtol=1e-03, atol=1e-03, equal_nan=False)
