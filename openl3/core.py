import os
import tensorflow as tf
import resampy
import traceback
import soundfile as sf
import numpy as np
from numbers import Real
from math import ceil
import warnings
from .models import load_audio_embedding_model, load_image_embedding_model, _validate_audio_frontend
from .openl3_exceptions import OpenL3Error
from .openl3_warnings import OpenL3Warning

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
        pad_length = (
            int(np.ceil((audio_len - frame_len)/float(hop_len))) * hop_len 
            - (audio_len - frame_len))

    if pad_length > 0:
        audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)

    return audio


def _get_num_windows(audio_len, frame_len, hop_len, center):
    if center:
        audio_len += int(frame_len / 2.0)

    if audio_len <= frame_len:
        return 1
    else:
        return 1 + int(np.ceil((audio_len - frame_len)/float(hop_len)))


def _preprocess_audio_batch(audio, sr, center=True, hop_size=0.1):
    """Process audio into batch format suitable for input to embedding model """
    if audio.size == 0:
        raise OpenL3Error('Got empty audio')

    # Warn user if audio is all zero
    if np.all(audio == 0):
        warnings.warn('Provided audio is all zeros', OpenL3Warning)

    # Check audio array dimension
    if audio.ndim > 2:
        raise OpenL3Error('Audio array can only be be 1D or 2D')
    elif audio.ndim == 2:
        # Downmix if multichannel
        audio = np.mean(audio, axis=1)

    if not isinstance(sr, Real) or sr <= 0:
        raise OpenL3Error('Invalid sample rate {}'.format(sr))

    if not isinstance(hop_size, Real) or hop_size <= 0:
        raise OpenL3Error('Invalid hop size {}'.format(hop_size))

    if center not in (True, False):
        raise OpenL3Error('Invalid center value {}'.format(center))

    # Resample if necessary
    if sr != TARGET_SR:
        audio = resampy.resample(audio, sr_orig=sr, sr_new=TARGET_SR, filter='kaiser_best')

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
    return x


def _librosa_linear_frontend(audio, n_fft=512, hop_length=242, db_amin=1e-10, 
                             db_ref=1.0, dynamic_range=80.0):
    '''Librosa linear frontend designed to match original Kapre (0.1.4).'''
    import librosa
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, center=False))
    S = librosa.power_to_db(S, ref=db_ref, amin=db_amin, top_db=dynamic_range)
    S -= S.max()
    return S


def _librosa_mel_frontend(audio, sr, n_mels=128, n_fft=2048, hop_length=242,
                          db_amin=1e-10, db_ref=1.0, dynamic_range=80.0):
    '''Librosa mel frontend designed to match original Kapre (0.1.4).'''
    import librosa
    S = librosa.feature.melspectrogram(
        audio, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, 
        center=True, power=1.0)
    S = librosa.power_to_db(S, ref=db_ref, amin=db_amin, top_db=dynamic_range)
    S -= S.max()
    return S


def preprocess_audio(audio, sr, hop_size=0.1, input_repr=None, center=True, **kw):
    """
    Preprocess the audio into a format compatible with the model.

    Parameters
    ----------
    audio : np.ndarray [shape=(N,) or (N,C)] or list[np.ndarray]
        1D numpy array of audio data or list of audio arrays for multiple
        inputs.
    sr : int or list[int]
        Sampling rate, or list of sampling rates. If not 48kHz audio will
        be resampled.
    hop_size : float
        Hop size in seconds.
    input_repr : str or None
        Spectrogram representation used for model.
        If input_repr, is None, then no spectrogram is computed and
        it is assumed that the model contains the details about 
        the input representation.
    center : boolean
        If True, pads beginning of signal so timestamps correspond
        to center of window.

    Returns
    -------
    input_data (np.ndarray): The preprocessed audio. Depending on 
        the value of input_repr, it will be np.ndarray[batch, time, frequency, 1]
        if a valid input representation is provided,
        or np.ndarray[batch, time, 1] if no input_repr is provided.
    """
    x = _preprocess_audio_batch(audio, sr, hop_size=hop_size, center=center)  # this resamples to 48k
    if input_repr:
        if input_repr == 'linear':
            x = np.stack([_librosa_linear_frontend(xi[0], **kw) for xi in x])[..., None]
        elif input_repr == 'mel128':
            x = np.stack([_librosa_mel_frontend(xi[0], TARGET_SR, n_mels=128, **kw) for xi in x])[..., None]
        elif input_repr == 'mel256':
            x = np.stack([_librosa_mel_frontend(xi[0], TARGET_SR, n_mels=256, **kw) for xi in x])[..., None]
        else:
            raise OpenL3Error('Invalid input representation "{}"'.format(input_repr))
    return x



def get_audio_embedding(audio, sr, model=None, input_repr=None,
                        content_type="music", embedding_size=6144,
                        center=True, hop_size=0.1, batch_size=32,
                        frontend="kapre", verbose=True):
    """
    Computes and returns L3 embedding for given audio data.

    Embeddings are computed for 1-second windows of audio.

    Parameters
    ----------
    audio : np.ndarray [shape=(N,) or (N,C)] or list[np.ndarray]
        1D numpy array of audio data or list of audio arrays for multiple
        inputs.
    sr : int or list[int]
        Sampling rate, or list of sampling rates. If not 48kHz audio will
        be resampled.
    model : tf.keras.Model or None
        Loaded model object. If a model is provided, then `input_repr`,
        `content_type`, and `embedding_size` will be ignored.
        If None is provided, the model will be loaded using
        the provided values of `input_repr`, `content_type` and
        `embedding_size`.
    input_repr : "linear", "mel128", or "mel256"
        Spectrogram representation used for model. Ignored if `model` is
        a valid Keras model.
    content_type : "music" or "env"
        Type of content used to train the embedding model. Ignored if `model` is
        a valid Keras model.
    embedding_size : 6144 or 512
        Embedding dimensionality. Ignored if `model` is a valid
        Keras model.
    center : boolean
        If True, pads beginning of signal so timestamps correspond
        to center of window.
    hop_size : float
        Hop size in seconds.
    batch_size : int
        Batch size used for input to embedding model
    frontend : "kapre" or "librosa"
        The audio frontend to use. By default, it will use "kapre".
    verbose : bool
        If True, prints verbose messages.

    Returns
    -------
    embedding : np.ndarray [shape=(T, D)] or list[np.ndarray]
        Array of embeddings for each window or list of such arrays for
        multiple audio clips.
    timestamps : np.ndarray [shape=(T,)] or list[np.ndarray]
        Array of timestamps corresponding to each embedding in the output or
        list of such arrays for multiple audio cplips.

    """
    if model is not None and not isinstance(model, tf.keras.Model):
        raise OpenL3Error('Invalid model provided. Must be of type tf.keras.Model'
                          ' but got {}'.format(str(type(model))))

    frontend, input_repr = _validate_audio_frontend(frontend, input_repr, model)

    if str(content_type) not in ("music", "env"):
        raise OpenL3Error('Invalid content type "{}"'.format(content_type))

    if embedding_size not in (6144, 512):
        raise OpenL3Error('Invalid content type "{}"'.format(embedding_size))

    if verbose not in (0, 1):
        raise OpenL3Error('Invalid verbosity level {}'.format(verbose))

    if isinstance(audio, np.ndarray):
        audio_list = [audio]
        list_input = False
    elif isinstance(audio, list):
        audio_list = audio
        list_input = True
    else:
        err_msg = 'audio must be type list[np.ndarray] or np.ndarray. Got {}'
        raise OpenL3Error(err_msg.format(type(audio)))

    if isinstance(sr, Real):
        sr_list = [sr] * len(audio_list)
    elif isinstance(sr, list):
        sr_list = sr
    else:
        err_msg = 'sr must be type list[numbers.Real] or numbers.Real. Got {}'
        raise OpenL3Error(err_msg.format(type(sr)))

    if len(audio_list) != len(sr_list):
        err_msg = ('Mismatch between number of audio inputs ({}) and number of'
                   ' sample rates ({})')
        raise OpenL3Error(err_msg.format(len(audio_list), len(sr_list)))

    # Get embedding model
    if model is None:
        model = load_audio_embedding_model(
            input_repr, content_type, embedding_size, 
            frontend=frontend)

    # Collect all audio arrays in a single array
    batch = []
    for x, sr in zip(audio_list, sr_list):
        x = preprocess_audio(
            x, sr, hop_size=hop_size, center=center, 
            input_repr=input_repr if frontend == 'librosa' else None)
        batch.append(x)

    file_batch_size_list = [x.shape[0] for x in batch]
    batch = np.vstack(batch)
    # Compute embeddings
    batch_embedding = model.predict(batch, verbose=1 if verbose else 0,
                                    batch_size=batch_size)

    embedding_list = []
    start_idx = 0
    for file_batch_size in file_batch_size_list:
        end_idx = start_idx + file_batch_size
        embedding_list.append(batch_embedding[start_idx:end_idx, ...])
        start_idx = end_idx

    ts_list = [np.arange(z.shape[0]) * hop_size for z in embedding_list]

    if not list_input:
        return embedding_list[0], ts_list[0]
    return embedding_list, ts_list


def process_audio_file(filepath, output_dir=None, suffix=None, model=None,
                       input_repr=None, content_type="music",
                       embedding_size=6144, center=True, hop_size=0.1,
                       batch_size=32, overwrite=False, frontend="kapre", verbose=True):
    """
    Computes and saves L3 embedding for a given audio file

    Parameters
    ----------
    filepath : str or list[str]
        Path or list of paths to WAV file(s) to be processed.
    output_dir : str or None
        Path to directory for saving output files. If None, output files will
        be saved to the directory containing the input file.
    suffix : str or None
        String to be appended to the output filename, i.e. <base filename>_<suffix>.npz.
        If None, then no suffix will be added, i.e. <base filename>.npz.
    model : tf.keras.Model or None
        Loaded model object. If a model is provided, then `input_repr`,
        `content_type`, and `embedding_size` will be ignored.
        If None is provided, the model will be loaded using
        the provided values of `input_repr`, `content_type` and
        `embedding_size`.
    input_repr : "linear", "mel128", or "mel256"
        Spectrogram representation used as model input. Ignored if `model` is
        a valid Keras model with a Kapre frontend. This is required with a 
        Librosa frontend.
    content_type : "music" or "env"
        Type of content used to train the embedding model. Ignored if `model` is
        a valid Keras model.
    embedding_size : 6144 or 512
        Embedding dimensionality. Ignored if `model` is a valid
        Keras model.
    center : boolean
        If True, pads beginning of signal so timestamps correspond
        to center of window.
    hop_size : float
        Hop size in seconds.
    batch_size : int
        Batch size used for input to embedding model
    overwrite : bool
        If True, overwrites existing output files
    frontend : "kapre" or "librosa"
        The audio frontend to use. By default, it will use "kapre".
    verbose : bool
        If True, prints verbose messages.

    Returns
    -------

    """
    if isinstance(filepath, str):
        filepath_list = [filepath]
    elif isinstance(filepath, list):
        filepath_list = filepath
    else:
        err_msg = 'filepath should be type str or list[str], but got {}.'
        raise OpenL3Error(err_msg.format(filepath))

    if not suffix:
        suffix = ""

    # Load model
    frontend, input_repr = _validate_audio_frontend(frontend, input_repr, model)
    if not model:
        model = load_audio_embedding_model(input_repr, content_type,
                                           embedding_size, frontend=frontend)

    audio_list = []
    sr_list = []
    batch_filepath_list = []

    total_batch_size = 0

    num_files = len(filepath_list)
    for file_idx, filepath in enumerate(filepath_list):
        if not os.path.exists(filepath):
            raise OpenL3Error('File "{}" could not be found.'.format(filepath))

        if verbose:
            print("openl3: Processing {} ({}/{})".format(filepath,
                                                         file_idx+1,
                                                         num_files))

        # Skip if overwriting isn't enabled and output file exists
        output_path = get_output_path(filepath, suffix + ".npz",
                                      output_dir=output_dir)
        if os.path.exists(output_path) and not overwrite:
            err_msg = "openl3: {} exists and overwriting not enabled, skipping."
            print(err_msg.format(output_path))
            continue

        try:
            audio, sr = sf.read(filepath)
        except Exception:
            err_msg = 'Could not open file "{}":\n{}'
            raise OpenL3Error(err_msg.format(filepath, traceback.format_exc()))

        audio_list.append(audio)
        sr_list.append(sr)
        batch_filepath_list.append(filepath)

        audio_length = ceil(audio.shape[0] / float(TARGET_SR / sr))
        frame_length = TARGET_SR
        hop_length = int(hop_size * TARGET_SR)
        num_windows = _get_num_windows(audio_length, frame_length,
                                       hop_length, center)
        total_batch_size += num_windows

        if total_batch_size >= batch_size or file_idx == (num_files - 1):
            embedding_list, ts_list = get_audio_embedding(
                audio_list, sr_list, model=model,
                input_repr=input_repr,
                content_type=content_type,
                embedding_size=embedding_size,
                center=center,
                hop_size=hop_size,
                batch_size=batch_size,
                frontend=frontend,
                verbose=verbose)
            for fpath, embedding, ts in zip(batch_filepath_list,
                                            embedding_list,
                                            ts_list):
                output_path = get_output_path(fpath, suffix + ".npz",
                                              output_dir=output_dir)

                np.savez(output_path, embedding=embedding, timestamps=ts)
                assert os.path.exists(output_path)

                if verbose:
                    print("openl3: Saved {}".format(output_path))

            audio_list = []
            sr_list = []
            batch_filepath_list = []
            total_batch_size = 0


def _preprocess_image_batch(image):
    """
    Preprocesses an image array so that they are rescaled and cropped to the
    appropriate dimensions required by the embedding model.

    Parameters
    ----------
    image : np.ndarray [shape=(H, W, C) or (N, H, W, C)]
        3D or 4D numpy array of image data. If the images are not 224x224,
        the images are resized so that the smallest size is 256 and then
        the center 224x224 patch is extracted from the images. Any type
        is accepted, and will be converted to np.float32 in the range [-1,1].
        Signed data-types are assumed to take on negative values.

    Returns
    -------
    batch : np.ndarray [shape=(N, H, W, C)]
        4d numpy array of image data.
    """
    import skimage
    if image.size == 0:
        raise OpenL3Error('Got empty image')

    # Warn user if image is all zero
    if np.all(image == 0):
        warnings.warn('Provided image is all zeros', OpenL3Warning)

    # Check image array dimension
    if image.ndim not in (3, 4):
        raise OpenL3Error('RGB image array can only be 3D or 4D (sequence of videos)')

    if image.shape[-1] != 3:
        raise OpenL3Error('Need 3 channel images corresponding to RGB.')

    if image.ndim == 3:
        # Add a batch dimension dimension
        image = image[np.newaxis, ...]

    if min(image.shape[1], image.shape[2]) < 224:
        err_msg = ('Image(s) must be at at least as large as 224x224 px. '
                   'Got image(s) of size {}x{} px')
        raise OpenL3Error(err_msg.format(image.shape[1], image.shape[2]))

    if image.shape[1] != 224 or image.shape[2] != 224:
        # If image is not 224x224, rescale to 256x256, and take center
        # 224x224 image patch, corresponding to what was done in L3
        scaling = 256.0 / min(image.shape[1], image.shape[2])
        batch = np.zeros((image.shape[0], 224, 224, 3))
        for idx, frame in enumerate(image):
            # Only reshape if image is larger than 256x256
            if min(frame.shape[0], frame.shape[1]) > 256:
                frame = skimage.transform.rescale(frame, scaling)
            x1, x2 = frame.shape[:-1]
            startx1 = x1//2-(224//2)
            startx2 = x2//2-(224//2)
            batch[idx] = frame[startx1:startx1+224,startx2:startx2+224]
    else:
        batch = image

    # Make sure image correct type
    if batch.dtype in (np.float16, np.float32, np.float64, np.int8,
                       np.int16, np.int32, np.int64):
        batch = skimage.img_as_float32(batch)
    elif batch.dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        # If unsigned int, convert to range [-1, 1]
        batch = 2 * skimage.img_as_float32(batch) - 1

    # Make sure maximum magnitude is in the range [-1, 1]
    if np.max(np.abs(batch)) > 1:
        batch /= np.max(np.abs(batch))

    return batch


def get_image_embedding(image, frame_rate=None, model=None,
                        input_repr="mel256", content_type="music",
                        embedding_size=8192, batch_size=32, verbose=True):
    """
    Computes and returns L3 embedding for given video frame (image) data.

    Embeddings are computed for every image in the input.

    Parameters
    ----------
    image : np.ndarray [shape=(H, W, C) or (N, H, W, C)] or list[np.ndarray]
        3D or 4D numpy array of image data. If the images are not 224x224,
        the images are resized so that the smallest size is 256 and then
        the center 224x224 patch is extracted from the images. Any type
        is accepted, and will be converted to np.float32 in the range [-1,1].
        Signed data-types are assumed to take on negative values. A list of
        image arrays can also be provided.
    frame_rate : int or list[int] or None
        Video frame rate (if applicable), which if provided results in
        a timestamp array being returned. A list of frame rates can also be
        provided. If None, no timestamp array is returned.
    model : tf.keras.Model or None
        Loaded model object. If a model is provided, then `input_repr`,
        `content_type`, and `embedding_size` will be ignored.
        If None is provided, the model will be loaded using
        the provided values of `input_repr`, `content_type` and
        `embedding_size`.
    input_repr : "linear", "mel128", or "mel256"
        Spectrogram representation used for to train audio part of embedding
        model. Ignored if `model` is a valid Keras model.
    content_type : "music" or "env"
        Type of content used to train the embedding model. Ignored if `model` is
        a valid Keras model.
    embedding_size : 8192 or 512
        Embedding dimensionality. Ignored if `model` is a valid
        Keras model.
    batch_size : int
        Batch size used for input to embedding model
    verbose : bool
        If True, prints verbose messages.

    Returns
    -------
        embedding : np.ndarray [shape=(N, D)]
            Array of embeddings for each frame.
        timestamps : np.ndarray [shape=(N,)]
            Array of timestamps for each frame. If `frame_rate` is None,
            this is not returned.
    """
    if model is not None and not isinstance(model, tf.keras.Model):
        raise OpenL3Error('Invalid model provided. Must be of type tf.keras.Model'
                          ' but got {}'.format(str(type(model))))

    if str(input_repr) not in ("linear", "mel128", "mel256"):
        raise OpenL3Error('Invalid input representation "{}"'.format(input_repr))

    if str(content_type) not in ("music", "env"):
        raise OpenL3Error('Invalid content type "{}"'.format(content_type))

    if embedding_size not in (8192, 512):
        raise OpenL3Error('Invalid content type "{}"'.format(embedding_size))

    if verbose not in (0, 1):
        raise OpenL3Error('Invalid verbosity level {}'.format(verbose))

    # Get embedding model
    if model is None:
        model = load_image_embedding_model(input_repr, content_type, embedding_size)

    if isinstance(image, np.ndarray):
        image_list = [image]
        list_input = False
    elif isinstance(image, list):
        image_list = image
        list_input = True
    else:
        err_msg = 'image must be type list[np.ndarray] or np.ndarray. Got {}'
        raise OpenL3Error(err_msg.format(type(image)))

    if frame_rate is None or isinstance(frame_rate, Real):
        frame_rate_list = [frame_rate] * len(image_list)
    elif isinstance(frame_rate, list):
        frame_rate_list = frame_rate
    else:
        err_msg = 'frame rate must be type list[numbers.Real] or numbers.Real. Got {}'
        raise OpenL3Error(err_msg.format(type(frame_rate)))

    if len(image_list) != len(frame_rate_list):
        err_msg = ('Mismatch between number of image inputs ({}) and number of'
                   ' frame rates ({})')
        raise OpenL3Error(err_msg.format(len(image_list), len(frame_rate_list)))

    batch = []
    file_batch_size_list = []
    for image, frame_rate in zip(image_list, frame_rate_list):
        if (frame_rate is not None) and (not isinstance(frame_rate, Real) or frame_rate <= 0):
            raise OpenL3Error('Invalid frame rate {}'.format(frame_rate))

        # Preprocess image to scale appropriate scale
        x = _preprocess_image_batch(image)
        batch.append(x)
        file_batch_size_list.append(x.shape[0])

    batch = np.vstack(batch)
    # Compute embeddings
    batch_embedding = model.predict(batch, verbose=1 if verbose else 0,
                                    batch_size=batch_size)

    embedding_list = []
    ts_list = []
    start_idx = 0
    for file_batch_size in file_batch_size_list:
        end_idx = start_idx + file_batch_size
        embedding = batch_embedding[start_idx:end_idx, ...]
        embedding_list.append(embedding)
        if frame_rate is not None:
            ts = np.arange(embedding.shape[0]) / float(frame_rate)
            ts_list.append(ts)

        start_idx = end_idx

    if frame_rate is not None:
        if not list_input:
            return embedding_list[0], ts_list[0]
        else:
            return embedding_list, ts_list
    else:
        if not list_input:
            return embedding_list[0]
        else:
            return embedding_list


def process_image_file(filepath, output_dir=None, suffix=None, model=None,
                       input_repr="mel256", content_type="music",
                       embedding_size=8192, batch_size=32,
                       overwrite=False, verbose=True):
    """
    Computes and saves L3 embedding for a given image file

    Parameters
    ----------
    filepath : str or list[str]
        Path or list of paths to image file(s) to be processed.
    output_dir : str or None
        Path to directory for saving output files. If None, output files will
        be saved to the directory containing the input file.
    suffix : str or None
        String to be appended to the output filename, i.e. <base filename>_<suffix>.npz.
        If None, then no suffix will be added, i.e. <base filename>.npz.
    model : tf.keras.Model or None
        Loaded model object. If a model is provided, then `input_repr`,
        `content_type`, and `embedding_size` will be ignored.
        If None is provided, the model will be loaded using
        the provided values of `input_repr`, `content_type` and
        `embedding_size`.
    input_repr : "linear", "mel128", or "mel256"
        Spectrogram representation used for model. Ignored if `model` is
        a valid Keras model.
    content_type : "music" or "env"
        Type of content used to train the embedding model. Ignored if `model` is
        a valid Keras model.
    embedding_size : 8192 or 512
        Embedding dimensionality. Ignored if `model` is a valid
        Keras model.
    batch_size : int
        Batch size used for input to embedding model
    overwrite : bool
        If True, overwrites existing output files
    verbose : bool
        If True, prints verbose messages.

    Returns
    -------

    """
    import skimage
    if isinstance(filepath, str):
        filepath_list = [filepath]
    elif isinstance(filepath, list):
        filepath_list = filepath
    else:
        err_msg = 'filepath should be type str or list[str], but got {}.'
        raise OpenL3Error(err_msg.format(filepath))

    # Load model
    if not model:
        model = load_image_embedding_model(input_repr, content_type,
                                           embedding_size)

    if not suffix:
        suffix = ""

    image_list = []
    batch_filepath_list = []

    num_files = len(filepath_list)
    for file_idx, filepath in enumerate(filepath_list):
        if not os.path.exists(filepath):
            raise OpenL3Error('File "{}" could not be found.'.format(filepath))

        if verbose:
            print("openl3: Processing {} ({}/{})".format(filepath,
                                                         file_idx+1,
                                                         num_files))

        # Skip if overwriting isn't enabled and output file exists
        output_path = get_output_path(filepath, suffix + ".npz",
                                      output_dir=output_dir)
        if os.path.exists(output_path) and not overwrite:
            print("openl3: {} exists, skipping.".format(output_path))
            continue

        try:
            image = skimage.io.imread(filepath)
            # Get rid of alpha dimension
            if image.shape[-1] == 4:
                image = image[..., :3]
        except Exception:
            raise OpenL3Error('Could not open file "{}":\n{}'.format(filepath, traceback.format_exc()))

        image_list.append(image[np.newaxis, ...])
        batch_filepath_list.append(filepath)

        if len(image_list) >= batch_size or file_idx == (num_files - 1):
            embedding_list = get_image_embedding(
                image_list, model=model,
                input_repr=input_repr,
                content_type=content_type,
                embedding_size=embedding_size,
                verbose=verbose)
            for fpath, embedding in zip(batch_filepath_list, embedding_list):
                output_path = get_output_path(fpath, suffix + ".npz",
                                              output_dir=output_dir)

                np.savez(output_path, embedding=embedding)
                assert os.path.exists(output_path)

                if verbose:
                    print("openl3: Saved {}".format(output_path))

            image_list = []
            batch_filepath_list = []


def process_video_file(filepath, output_dir=None, suffix=None,
                       audio_model=None, image_model=None,
                       input_repr=None, content_type="music",
                       audio_embedding_size=6144, audio_center=True,
                       audio_hop_size=0.1, image_embedding_size=8192,
                       audio_batch_size=32, image_batch_size=32,
                       audio_frontend="kapre",
                       overwrite=False, verbose=True):
    """
    Computes and saves L3 audio and video frame embeddings for a given video file

    Note that image embeddings are computed for every frame of the video. Also
    note that embeddings for the audio and images are not temporally aligned.
    Please refer to the timestamps in the output files for the corresponding
    timestamps for each set of embeddings.

    Parameters
    ----------
    filepath : str or list[str]
        Path or list of paths to video file(s) to be processed.
    output_dir : str or None
        Path to directory for saving output files. If None, output files will
        be saved to the directory containing the input file.
    suffix : str or None
        String to be appended to the output filename,
        i.e. <base filename>_<modality>_<suffix>.npz.
        If None, then no suffix will be added,
        i.e. <base filename>_<modality>.npz.
    audio_model : tf.keras.Model or None
        Loaded audio model object. If a model is provided, then `input_repr`,
        `content_type`, and `embedding_size` will be ignored.
        If None is provided, the model will be loaded using
        the provided values of `input_repr`, `content_type` and
        `embedding_size`.
    image_model : tf.keras.Model or None
        Loaded audio model object. If a model is provided, then `input_repr`,
        `content_type`, and `embedding_size` will be ignored.
        If None is provided, the model will be loaded using
        the provided values of `input_repr`, `content_type` and
        `embedding_size`.
    input_repr : "linear", "mel128", or "mel256"
        Spectrogram representation used for audio model. Ignored if `model` is
        a valid Keras model with a Kapre frontend. This is required with a 
        Librosa frontend.
    content_type : "music" or "env"
        Type of content used to train the embedding model. Ignored if `model` is
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
    audio_batch_size : int
        Batch size used for input to audio embedding model
    image_batch_size : int
        Batch size used for input to image embedding model
    audio_frontend : "kapre" or "librosa"
        The audio frontend to use. By default, it will use "kapre".
    overwrite : bool
        If True, overwrites existing output files
    verbose : bool
        If True, prints verbose messages.

    Returns
    -------

    """
    from moviepy.video.io.VideoFileClip import VideoFileClip
    if isinstance(filepath, str):
        filepath_list = [filepath]
    elif isinstance(filepath, list):
        filepath_list = filepath
    else:
        err_msg = 'filepath should be type str or list[str], but got {}.'
        raise OpenL3Error(err_msg.format(filepath))

    audio_frontend, input_repr = _validate_audio_frontend(audio_frontend, input_repr, audio_model)

    # Load models
    if not audio_model:
        audio_model = load_audio_embedding_model(input_repr, content_type,
                                                 audio_embedding_size, 
                                                 frontend=audio_frontend)
    if not image_model:
        image_model = load_image_embedding_model(input_repr, content_type,
                                                 image_embedding_size)

    audio_suffix, image_suffix = "audio", "image"
    if suffix:
        audio_suffix += "_" + suffix
        image_suffix += "_" + suffix

    audio_list = []
    sr_list = []
    audio_batch_filepath_list = []
    total_audio_batch_size = 0

    image_list = []
    frame_rate_list = []
    image_batch_filepath_list = []

    num_files = len(filepath_list)
    for file_idx, filepath in enumerate(filepath_list):

        if not os.path.exists(filepath):
            raise OpenL3Error('File "{}" could not be found.'.format(filepath))

        if verbose:
            print("openl3: Processing {} ({}/{})".format(filepath,
                                                         file_idx+1,
                                                         num_files))

        # Skip if overwriting isn't enabled and output file exists
        audio_output_path = get_output_path(filepath, audio_suffix + ".npz",
                                            output_dir=output_dir)
        image_output_path = get_output_path(filepath, image_suffix + ".npz",
                                            output_dir=output_dir)
        skip_audio = os.path.exists(audio_output_path) and not overwrite
        skip_image = os.path.exists(image_output_path) and not overwrite

        if skip_audio and skip_image:
            err_msg = "openl3: {} and {} exist, skipping."
            print(err_msg.format(audio_output_path, image_output_path))
            continue

        try:
            clip = VideoFileClip(filepath, target_resolution=(256, 256),
                                 audio_fps=TARGET_SR)
            audio = clip.audio.to_soundarray(fps=TARGET_SR)
            images = np.array([frame for frame in clip.iter_frames()])
        except Exception:
            err_msg = 'Could not open file "{}":\n{}'
            raise OpenL3Error(err_msg.format(filepath, traceback.format_exc()))

        if not skip_audio:
            audio_list.append(audio)
            sr_list.append(TARGET_SR)
            audio_batch_filepath_list.append(filepath)
            audio_len = audio.shape[0]
            audio_hop_length = int(audio_hop_size * TARGET_SR)
            num_windows = 1 + max(ceil((audio_len - TARGET_SR)/float(audio_hop_length)), 0)
            total_audio_batch_size += num_windows
        else:
            err_msg = "openl3: {} exists, skipping audio embedding extraction."
            print(err_msg.format(audio_output_path))

        if not skip_image:
            image_list.append(images)
            frame_rate_list.append(int(clip.fps))
            image_batch_filepath_list.append(filepath)
        else:
            err_msg = "openl3: {} exists, skipping image embedding extraction."
            print(err_msg.format(image_output_path))

        if (total_audio_batch_size >= audio_batch_size or file_idx == (num_files - 1)) and len(audio_list) > 0:
            embedding_list, ts_list = get_audio_embedding(
                audio_list, sr_list, model=audio_model,
                input_repr=input_repr,
                content_type=content_type,
                embedding_size=audio_embedding_size,
                center=audio_center,
                hop_size=audio_hop_size,
                batch_size=audio_batch_size,
                frontend=audio_frontend,
                verbose=verbose)
            for fpath, embedding, ts in zip(audio_batch_filepath_list,
                                            embedding_list,
                                            ts_list):
                output_path = get_output_path(fpath, audio_suffix + ".npz",
                                              output_dir=output_dir)

                np.savez(output_path, embedding=embedding, timestamps=ts)
                assert os.path.exists(output_path)

                if verbose:
                    print("openl3: Saved {}".format(output_path))

            audio_list = []
            sr_list = []
            audio_batch_filepath_list = []
            total_audio_batch_size = 0

        if (len(image_list) >= image_batch_size or file_idx == (num_files - 1)) and len(image_list) > 0:
            embedding_list, ts_list = get_image_embedding(
                image_list, frame_rate_list,
                model=image_model, input_repr=input_repr,
                content_type=content_type,
                embedding_size=image_embedding_size,
                batch_size=image_batch_size,
                verbose=verbose)
            for fpath, embedding, ts in zip(image_batch_filepath_list,
                                            embedding_list,
                                            ts_list):
                output_path = get_output_path(fpath, image_suffix + ".npz",
                                              output_dir=output_dir)

                np.savez(output_path, embedding=embedding, timestamps=ts)
                assert os.path.exists(output_path)

                if verbose:
                    print("openl3: Saved {}".format(output_path))

            image_list = []
            frame_rate_list = []
            image_batch_filepath_list = []


def get_output_path(filepath, suffix, output_dir=None):
    """
    Returns path to output file corresponding to the given input file.

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
