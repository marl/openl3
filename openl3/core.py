import os
import sklearn.decomposition
import keras
import resampy
import traceback
import soundfile as sf
import numpy as np
from numbers import Real
from math import ceil
import warnings
from scipy.misc import imresize, imread
from .models import load_audio_embedding_model, load_image_embedding_model
from .openl3_exceptions import OpenL3Error
from .openl3_warnings import OpenL3Warning
from moviepy.video.io import VideoFileClip
import skimage


TARGET_SR = 48000


def _center_audio(audio, frame_len):
    """Center audio so that first sample will occur in the middle of the first frame"""
    return np.pad(audio, (int(frame_len / 2.0), 0), mode='constant', constant_values=0)


def _pad_audio(audio, frame_len, hop_len):
    """Pad audio if necessary so that all samples are processed"""
    audio_len = audio.size
    if audio_len < frame_len:
        pad_length = frame_len - audio_len
    else:
        pad_length = int(np.ceil((audio_len - frame_len)/float(hop_len))) * hop_len \
                     - (audio_len - frame_len)

    if pad_length > 0:
        audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)

    return audio


def get_audio_embedding(audio, sr, model=None, input_repr="mel256",
                        content_type="music", embedding_size=6144,
                        center=True, hop_size=0.1, verbose=1):
    """
    Computes and returns L3 embedding for given audio data

    Parameters
    ----------
    audio : np.ndarray [shape=(N,) or (N,C)]
        1D numpy array of audio data.
    sr : int
        Sampling rate, if not 48kHz will audio will be resampled.
    model : keras.models.Model or None
        Loaded model object. If a model is provided, then `input_repr`,
        `content_type`, and `embedding_size` will be ignored.
        If None is provided, the model will be loaded using
        the provided values of `input_repr`, `content_type` and
        `embedding_size`.
    input_repr : "linear", "mel128", or "mel256"
        Spectrogram representation used for model. Ignored if `model` is
        a valid Keras model.
    content_type : "music" or "env"
        Type of content used to train embedding. Ignored if `model` is
        a valid Keras model.
    embedding_size : 6144 or 512
        Embedding dimensionality. Ignored if `model` is a valid
        Keras model.
    center : boolean
        If True, pads beginning of signal so timestamps correspond
        to center of window.
    hop_size : float
        Hop size in seconds.
    verbose : 0 or 1
        Keras verbosity.

    Returns
    -------
        embedding : np.ndarray [shape=(T, D)]
            Array of embeddings for each window.
        timestamps : np.ndarray [shape=(T,)]
            Array of timestamps corresponding to each embedding in the output.

    """
    if audio.size == 0:
        raise OpenL3Error('Got empty audio')

    # Warn user if audio is all zero
    if np.all(audio == 0):
        warnings.warn('Provided audio is all zeros', OpenL3Warning)

    if model is not None and not isinstance(model, keras.models.Model):
        raise OpenL3Error('Invalid model provided. Must be of type keras.model.Models'
                          ' but got {}'.format(str(type(model))))

    if str(input_repr) not in ("linear", "mel128", "mel256"):
        raise OpenL3Error('Invalid input representation "{}"'.format(input_repr))

    if str(content_type) not in ("music", "env"):
        raise OpenL3Error('Invalid content type "{}"'.format(content_type))

    if embedding_size not in (6144, 512):
        raise OpenL3Error('Invalid content type "{}"'.format(embedding_size))

    if (sr is not None) and (not isinstance(sr, Real) or sr <= 0):
        raise OpenL3Error('Invalid sample rate {}'.format(sr))

    if not isinstance(hop_size, Real) or hop_size <= 0:
        raise OpenL3Error('Invalid hop size {}'.format(hop_size))

    if verbose not in (0, 1):
        raise OpenL3Error('Invalid verbosity level {}'.format(verbose))

    if center not in (True, False):
        raise OpenL3Error('Invalid center value {}'.format(center))

    # Check audio array dimension
    if audio.ndim > 2:
        raise OpenL3Error('Audio array can only be be 1D or 2D')
    elif audio.ndim == 2:
        # Downmix if multichannel
        audio = np.mean(audio, axis=1)

    # Resample if necessary
    if sr != TARGET_SR:
        audio = resampy.resample(audio, sr_orig=sr, sr_new=TARGET_SR, filter='kaiser_best')

    # Get embedding model
    if model is None:
        model = load_audio_embedding_model(input_repr, content_type, embedding_size)

    audio_len = audio.size
    frame_len = TARGET_SR
    hop_len = int(hop_size * TARGET_SR)

    if audio_len < frame_len:
        warnings.warn('Duration of provided audio is shorter than window size (1 second). Audio will be padded.',
                      OpenL3Warning)

    if center:
        # Center audio
        audio = _center_audio(audio, frame_len)

    # Pad if necessary to ensure that we process all samples
    audio = _pad_audio(audio, frame_len, hop_len)

    # Split audio into frames, copied from librosa.util.frame
    n_frames = 1 + int((len(audio) - frame_len) / float(hop_len))
    x = np.lib.stride_tricks.as_strided(audio, shape=(frame_len, n_frames),
        strides=(audio.itemsize, hop_len * audio.itemsize)).T

    # Add a channel dimension
    x = x.reshape((x.shape[0], 1, x.shape[-1]))

    # Get embedding and timestamps
    embedding = model.predict(x, verbose=verbose)

    ts = np.arange(embedding.shape[0]) * hop_size

    return embedding, ts


def process_audio_file(filepath, output_dir=None, suffix=None, model=None,
                       input_repr="mel256", content_type="music",
                       embedding_size=6144, center=True, hop_size=0.1, verbose=True):
    """
    Computes and saves L3 embedding for given audio file

    Parameters
    ----------
    filepath : str
        Path to WAV file to be processed.
    output_dir : str or None
        Path to directory for saving output files. If None, output files will
        be saved to the directory containing the input file.
    suffix : str or None
        String to be appended to the output filename, i.e. <base filename>_<suffix>.npz.
        If None, then no suffix will be added, i.e. <base filename>.npz.
    model : keras.models.Model or None
        Loaded model object. If a model is provided, then `input_repr`,
        `content_type`, and `embedding_size` will be ignored.
        If None is provided, the model will be loaded using
        the provided values of `input_repr`, `content_type` and
        `embedding_size`.
    input_repr : "linear", "mel128", or "mel256"
        Spectrogram representation used for model. Ignored if `model` is
        a valid Keras model.
    content_type : "music" or "env"
        Type of content used to train embedding. Ignored if `model` is
        a valid Keras model.
    embedding_size : 6144 or 512
        Embedding dimensionality. Ignored if `model` is a valid
        Keras model.
    center : boolean
        If True, pads beginning of signal so timestamps correspond
        to center of window.
    hop_size : float
        Hop size in seconds.
    verbose : 0 or 1
        Keras verbosity.

    Returns
    -------

    """
    if not os.path.exists(filepath):
        raise OpenL3Error('File "{}" could not be found.'.format(filepath))

    try:
        audio, sr = sf.read(filepath)
    except Exception:
        raise OpenL3Error('Could not open file "{}":\n{}'.format(filepath, traceback.format_exc()))

    if not suffix:
        suffix = ""

    output_path = get_output_path(filepath, suffix + ".npz", output_dir=output_dir)

    embedding, ts = get_audio_embedding(audio, sr, model=model, input_repr=input_repr,
                                        content_type=content_type,
                                        embedding_size=embedding_size, center=center,
                                        hop_size=hop_size, verbose=1 if verbose else 0)

    np.savez(output_path, embedding=embedding, timestamps=ts)
    assert os.path.exists(output_path)


def _preprocess_image_batch(image):
    """
    Preprocesses an image array so that they are rescaled and cropped to the
    appropriate dimensions required by the embedding model.

    Parameters
    ----------
    image : np.ndarray [shape=(H, W, C) or (N, H, W, C)]
        3D or 4D numpy array of image data. If the images are not 224x224,
        the images are resized so that the smallest size is 256 and then
        the center 224x224 patch is extracted from the images.

    Returns
    -------
    batch : np.ndarray [shape=(N, H, W, C)]
        4d numpy array of image data.
    """
    if image.ndim == 3:
        # Add a batch dimension dimension
        image = image[np.newaxis, ...]

    if min(image.shape[1], image.shape[2]) < 224:
        err_msg = 'Image(s) must be at at least as large as 224x224 px. ' \
                  'Got image(s) of size {}x{} px'
        raise OpenL3Error(err_msg.format(image.shape[1], image.shape[2]))

    if image.shape[1] != 224 or image.shape[2] != 224:
        # If image is not 224x224, rescale to 256x256, and take center
        # 224x224 image patch, corresponding to what was done in L3
        scaling = 256.0 / min(image.shape[1], image.shape[2])
        batch = np.zeros((image.shape[0], 224, 224, 3))
        for idx, frame in enumerate(image):
            # Only reshape if image is larger than 256x256
            if min(frame.shape[0], frame.shape[1]) > 256:
                frame = imresize(frame, scaling, interp='bilinear')
            x1, x2 = frame.shape[:-1]
            startx1 = x1//2-(224//2)
            startx2 = x2//2-(224//2)
            batch[idx] = frame[startx1:startx1+224,startx2:startx2+224]
    else:
        batch = image

    # Make sure image is in [-1, 1]
    batch = 2 * skimage.img_as_float32(skimage.img_as_ubyte(batch)) - 1

    return batch


def get_image_embedding(image, frame_rate=None, model=None,
                         input_repr="mel256", content_type="music",
                         embedding_size=8192, verbose=1):
    """
    Computes and returns L3 embedding for given video frame (image) data

    Parameters
    ----------
    image : np.ndarray [shape=(H, W, C) or (N, H, W, C)]
        3D or 4D numpy array of image data. If the images are not 224x224,
        the images are resized so that the smallest size is 256 and then
        the center 224x224 patch is extracted from the images.
    frame_rate : int or None
        Video frame rate (if applicable), which if provided results in
        a timestamp array being returned. If None, no timestamp array is
        returned.
    model : keras.models.Model or None
        Loaded model object. If a model is provided, then `input_repr`,
        `content_type`, and `embedding_size` will be ignored.
        If None is provided, the model will be loaded using
        the provided values of `input_repr`, `content_type` and
        `embedding_size`.
    input_repr : "linear", "mel128", or "mel256"
        Spectrogram representation used for to train audio part of audio-visual
        correspondence model. Ignored if `model` is a valid Keras model.
    content_type : "music" or "env"
        Type of content used to train embedding. Ignored if `model` is
        a valid Keras model.
    embedding_size : 8192 or 512
        Embedding dimensionality. Ignored if `model` is a valid
        Keras model.
    verbose : 0 or 1
        Keras verbosity.

    Returns
    -------
        embedding : np.ndarray [shape=(N, D)]
            Array of embeddings for each frame.
        timestamps : np.ndarray [shape=(N,)]
            Array of timestamps for each frame. If `frame_rate` is None,
            this is not returned.
    """
    if image.size == 0:
        raise OpenL3Error('Got empty image')

    # Warn user if image is all zero
    if np.all(image == 0):
        warnings.warn('Provided image is all zeros', OpenL3Warning)

    if model is not None and not isinstance(model, keras.models.Model):
        raise OpenL3Error('Invalid model provided. Must be of type keras.model.Models'
                          ' but got {}'.format(str(type(model))))

    if str(input_repr) not in ("linear", "mel128", "mel256"):
        raise OpenL3Error('Invalid input representation "{}"'.format(input_repr))

    if str(content_type) not in ("music", "env"):
        raise OpenL3Error('Invalid content type "{}"'.format(content_type))

    if embedding_size not in (8192, 512):
        raise OpenL3Error('Invalid content type "{}"'.format(embedding_size))

    if verbose not in (0, 1):
        raise OpenL3Error('Invalid verbosity level {}'.format(verbose))

    if (frame_rate is not None) and (not isinstance(frame_rate, Real) or sr <= 0):
        raise OpenL3Error('Invalid frame rate {}'.format(frame_rate))

    # Check image array dimension
    if image.ndim not in (3, 4):
        raise OpenL3Error('RGB image array can only be 3D or 4D (sequence of videos)')

    # Get embedding model
    if model is None:
        model = load_image_embedding_model(input_repr, content_type, embedding_size)

    if image.shape[-1] != 3:
        raise OpenL3Error('Need 3 channel images corresponding to RGB.')

    # Preprocess image to scale appropriate scale
    x = _preprocess_image_batch(image)

    # Get embedding and timestamps
    embedding = model.predict(x, verbose=verbose)

    if frame_rate is not None:
        ts = np.arange(embedding.shape[0]) / float(frame_rate)
        return embedding, ts
    else:
        return embedding


def process_image_file(filepath, output_dir=None, suffix=None, model=None,
                       input_repr="mel256", content_type="music",
                       embedding_size=8192, verbose=True):
    """
    Computes and saves L3 embedding for given image file

    Parameters
    ----------
    filepath : str
        Path to image file to be processed.
    output_dir : str or None
        Path to directory for saving output files. If None, output files will
        be saved to the directory containing the input file.
    suffix : str or None
        String to be appended to the output filename, i.e. <base filename>_<suffix>.npz.
        If None, then no suffix will be added, i.e. <base filename>.npz.
    model : keras.models.Model or None
        Loaded model object. If a model is provided, then `input_repr`,
        `content_type`, and `embedding_size` will be ignored.
        If None is provided, the model will be loaded using
        the provided values of `input_repr`, `content_type` and
        `embedding_size`.
    input_repr : "linear", "mel128", or "mel256"
        Spectrogram representation used for model. Ignored if `model` is
        a valid Keras model.
    content_type : "music" or "env"
        Type of content used to train embedding. Ignored if `model` is
        a valid Keras model.
    embedding_size : 8192 or 512
        Embedding dimensionality. Ignored if `model` is a valid
        Keras model.
    verbose : 0 or 1
        Keras verbosity.

    Returns
    -------

    """
    if not os.path.exists(filepath):
        raise OpenL3Error('File "{}" could not be found.'.format(filepath))

    try:
        image = imread(filepath, mode='RGB')
    except Exception:
        raise OpenL3Error('Could not open file "{}":\n{}'.format(filepath, traceback.format_exc()))

    if not suffix:
        suffix = ""

    output_path = get_output_path(filepath, suffix + ".npz", output_dir=output_dir)

    embedding, ts = get_image_embedding(image, model=model,
                                        input_repr=input_repr,
                                        content_type=content_type,
                                        embedding_size=embedding_size,
                                        verbose=1 if verbose else 0)

    np.savez(output_path, embedding=embedding, timestamps=ts)
    assert os.path.exists(output_path)


def process_video_file(filepath, output_dir=None, suffix=None,
                       audio_model=None, image_model=None,
                       input_repr="mel256", content_type="music",
                       audio_embedding_size=6144, audio_center=True,
                       audio_hop_size=0.1, image_embedding_size=8192,
                       verbose=True):
    """
    Computes and saves L3 audio and video frame embeddings for given video file

    Parameters
    ----------
    filepath : str
        Path to video file to be processed.
    output_dir : str or None
        Path to directory for saving output files. If None, output files will
        be saved to the directory containing the input file.
    suffix : str or None
        String to be appended to the output filename,
        i.e. <base filename>_<modality>_<suffix>.npz.
        If None, then no suffix will be added,
        i.e. <base filename>_<modality>.npz.
    audio_model : keras.models.Model or None
        Loaded audio model object. If a model is provided, then `input_repr`,
        `content_type`, and `embedding_size` will be ignored.
        If None is provided, the model will be loaded using
        the provided values of `input_repr`, `content_type` and
        `embedding_size`.
    image_model : keras.models.Model or None
        Loaded audio model object. If a model is provided, then `input_repr`,
        `content_type`, and `embedding_size` will be ignored.
        If None is provided, the model will be loaded using
        the provided values of `input_repr`, `content_type` and
        `embedding_size`.
    input_repr : "linear", "mel128", or "mel256"
        Spectrogram representation used for audio model. Ignored if `model` is
        a valid Keras model.
    content_type : "music" or "env"
        Type of content used to train embedding. Ignored if `model` is
        a valid Keras model.
    audio_embedding_size : 6144 or 512
        Audio embedding dimensionality. Ignored if `model` is a valid Keras model.
    audio_center : boolean
        If True, pads beginning of audio signal so timestamps correspond
        to center of window.
    audio_hop_size : float
        Hop size in seconds.
    image_embedding_size : 8192 or 512
        Video frame embedding dimensionality. Ignored if `model` is a valid Keras model.
    verbose : 0 or 1
        Keras verbosity.

    Returns
    -------

    """
    if not os.path.exists(filepath):
        raise OpenL3Error('File "{}" could not be found.'.format(filepath))

    try:
        clip = VideoFileClip(filepath, target_resolution=(256, 256),
                             audio_fps=TARGET_SR)
        audio = clip.audio.to_soundarray(fps=TARGET_SR)
        images = np.array([frame for frame in clip.iter_frames()])
    except Exception:
        raise OpenL3Error('Could not open file "{}":\n{}'.format(filepath, traceback.format_exc()))

    if not suffix:
        suffix = ""
    audio_output_path = get_output_path(filepath, '_'.join('audio', suffix) + ".npz", output_dir=output_dir)
    image_output_path = get_output_path(filepath, '_'.join('image', suffix) + ".npz", output_dir=output_dir)

    audio_embedding, audio_ts = get_audio_embedding(
        audio, TARGET_SR, model=audio_model, input_repr=input_repr,
        content_type=content_type, embedding_size=audio_embedding_size,
        center=audio_center, hop_size=audio_hop_size, verbose=1)

    image_embedding, image_ts = get_image_embedding(
        images, int(clip.fps), model=image_model, input_repr=input_repr,
        content_type=content_type,
        embedding_size=image_embedding_size,
        verbose=1 if verbose else 0)

    np.savez(audio_output_path, embedding=audio_embedding, timestamps=audio_ts)
    assert os.path.exists(audio_output_path)
    np.savez(image_output_path, embedding=image_embedding, timestamps=image_ts)
    assert os.path.exists(image_output_path)


def get_output_path(filepath, suffix, output_dir=None):
    """

    Parameters
    ----------
    filepath : str
        Path to audio file to be processed
    suffix : str
        String to append to filename (including extension)
    output_dir : str or None
        Path to directory where file will be saved. If None, will use directory of given filepath.

    Returns
    -------
    output_path : str
        Path to output file

    """
    base_filename = os.path.splitext(os.path.basename(filepath))[0]
    if not output_dir:
        output_dir = os.path.dirname(filepath)

    if suffix[0] != '.':
        output_filename = "{}_{}".format(base_filename, suffix)
    else:
        output_filename = base_filename + suffix

    return os.path.join(output_dir, output_filename)
