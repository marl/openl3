import os
import warnings
import sklearn.decomposition
import numpy as np
from .openl3_exceptions import OpenL3Error

with warnings.catch_warnings():
    # Suppress TF and Keras warnings when importing
    warnings.simplefilter("ignore")
    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow.keras import Model
    from tensorflow.keras.layers import (
        Input, Conv2D, Permute, BatchNormalization, MaxPooling2D,
        Flatten, Activation, Lambda)
    import tensorflow.keras.regularizers as regularizers


VALID_FRONTENDS = ("librosa", "kapre")
VALID_INPUT_REPRS = ("linear", "mel128", "mel256")
VALID_CONTENT_TYPES = ("music", "env")
VALID_EMBEDDING_SIZES = (6144, 512)


def _log10(x):
    '''log10 tensorflow function.'''
    return tf.math.log(x) / tf.math.log(tf.constant(10, dtype=x.dtype))


def kapre_v0_1_4_magnitude_to_decibel(x, ref_value=1.0, amin=1e-10, dynamic_range=80.0):
    '''log10 tensorflow function.'''
    amin = tf.cast(amin or 1e-10, dtype=x.dtype)
    max_axis = tuple(range(K.ndim(x))[1:]) or None
    log_spec = 10. * _log10(K.maximum(x, amin))
    return K.maximum(
        log_spec - K.max(log_spec, axis=max_axis, keepdims=True), 
        -dynamic_range)


def __fix_kapre_spec(func):
    '''Wraps the kapre composite layer interface to revert .'''
    def get_spectrogram(*a, return_decibel=False, **kw):
        seq = func(*a, return_decibel=False, **kw)
        if return_decibel:
            seq.add(Lambda(kapre_v0_1_4_magnitude_to_decibel))
        seq.add(Permute((2, 1, 3)))  # the output is (None, t, f, ch) instead of (None, f, t, ch), so gotta fix that
        return seq
    return get_spectrogram


def _validate_audio_frontend(frontend='kapre', input_repr=None, model=None):
    '''Make sure that the audio frontend matches the model and input_repr.'''
    ndims = len(model.input_shape) if model is not None else None

    # if frontend == 'infer':  # detect which frontend to use
    #     if model is None:  # default
    #         frontend = 'kapre'
    #     elif ndims == 3:  # shape: [batch, channel, samples]
    #         frontend = 'kapre'
    #     elif ndims == 4:  # shape: [batch, frequency, time, channel]
    #         frontend = 'librosa'
    #     else:
    #         raise OpenL3Error(
    #             'Invalid model input shape: {}. Expected a model '
    #             'with either a 3 or 4 dimensional input,  got {}.'.format(model.input_shape, ndims))

    if frontend not in VALID_FRONTENDS:
        raise OpenL3Error('Invalid frontend "{}". Must be one of {}'.format(frontend, VALID_FRONTENDS))

    # validate that our model shape matches our frontend.
    if ndims is not None:
        if frontend == 'kapre' and ndims != 3:
            raise OpenL3Error('Invalid model input shape: {}. Expected 3 dims got {}.'.format(model.input_shape, ndims))
        if frontend == 'librosa' and ndims != 4:
            raise OpenL3Error('Invalid model input shape: {}. Expected 4 dims got {}.'.format(model.input_shape, ndims))

    if input_repr is None:
        if frontend == 'librosa':
            raise OpenL3Error('You must specify input_repr for a librosa frontend.')
        else:
            input_repr = 'mel256'
    
    if str(input_repr) not in VALID_INPUT_REPRS:
        raise OpenL3Error('Invalid input representation "{}". Must be one of {}'.format(input_repr, VALID_INPUT_REPRS))

    return frontend, input_repr


AUDIO_POOLING_SIZES = {
    'linear': {
        6144: (8, 8),
        512: (32, 24),
    },
    'mel128': {
        6144: (4, 8),
        512: (16, 24),
    },
    'mel256': {
        6144: (8, 8),
        512: (32, 24),
    }
}

IMAGE_POOLING_SIZES = {
    8192: (7, 7),
    512: (28, 28),
}


def load_audio_embedding_model(input_repr, content_type, embedding_size, frontend='kapre'):
    """
    Returns a model with the given characteristics. Loads the model
    if the model has not been loaded yet.

    Parameters
    ----------
    input_repr : "linear", "mel128", or "mel256"
        Spectrogram representation used for audio model.
    content_type : "music" or "env"
        Type of content used to train embedding.
    embedding_size : 6144 or 512
        Embedding dimensionality.
    frontend : "kapre" or "librosa"
        The audio frontend to use. If frontend == 'kapre', then the kapre frontend will
        be included. Otherwise no frontend will be added inside the keras model.

    Returns
    -------
    model : tf.keras.Model
        Model object.
    """
    frontend, input_repr = _validate_audio_frontend(frontend, input_repr)

    # Construct embedding model and load model weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = AUDIO_MODELS[input_repr](include_frontend=frontend == 'kapre')

    m.load_weights(get_audio_embedding_model_path(input_repr, content_type))

    # Pooling for final output embedding size
    pool_size = AUDIO_POOLING_SIZES[input_repr][embedding_size]
    y_a = MaxPooling2D(pool_size=pool_size, padding='same')(m.output)
    y_a = Flatten()(y_a)
    m = Model(inputs=m.input, outputs=y_a)
    m.frontend = frontend
    return m


def get_audio_embedding_model_path(input_repr, content_type):
    """
    Returns the local path to the model weights file for the model
    with the given characteristics

    Parameters
    ----------
    input_repr : "linear", "mel128", or "mel256"
        Spectrogram representation used for model.
    content_type : "music" or "env"
        Type of content used to train embedding.

    Returns
    -------
    output_path : str
        Path to given model object
    """
    return os.path.join(os.path.dirname(__file__),
                        'openl3_audio_{}_{}.h5'.format(input_repr, content_type))


def load_image_embedding_model(input_repr, content_type, embedding_size):
    """
    Returns a model with the given characteristics. Loads the model
    if the model has not been loaded yet.

    Parameters
    ----------
    input_repr : "linear", "mel128", or "mel256"
        Spectrogram representation used for audio model.
    content_type : "music" or "env"
        Type of content used to train embedding.
    embedding_size : 6144 or 512
        Embedding dimensionality.

    Returns
    -------
    model : tf.keras.Model
        Model object.
    """

    # Construct embedding model and load model weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = _construct_image_network()

    m.load_weights(get_image_embedding_model_path(input_repr, content_type))

    # Pooling for final output embedding size
    pool_size = IMAGE_POOLING_SIZES[embedding_size]
    y_i = MaxPooling2D(pool_size=pool_size, padding='same')(m.output)
    y_i = Flatten()(y_i)
    m = Model(inputs=m.input, outputs=y_i)
    return m


def get_image_embedding_model_path(input_repr, content_type):
    """
    Returns the local path to the model weights file for the model
    with the given characteristics

    Parameters
    ----------
    input_repr : "linear", "mel128", or "mel256"
        Spectrogram representation used for model.
    content_type : "music" or "env"
        Type of content used to train embedding.

    Returns
    -------
    output_path : str
        Path to given model object
    """
    return os.path.join(os.path.dirname(__file__),
                        'openl3_image_{}_{}.h5'.format(input_repr, content_type))


def _construct_linear_audio_network(include_frontend=True):
    """
    Returns an uninitialized model object for an audio network with a linear
    spectrogram input (With 257 frequency bins)

    Returns
    -------
    model : tf.keras.Model
        Model object.
    """

    weight_decay = 1e-5
    n_dft = 512
    n_hop = 242
    asr = 48000
    audio_window_dur = 1

    if include_frontend:
        # INPUT
        input_shape = (1, asr * audio_window_dur)
        x_a = Input(shape=input_shape, dtype='float32')

        # SPECTROGRAM PREPROCESSING
        # 257 x 197 x 1
        from kapre.composed import get_stft_magnitude_layer
        spec = __fix_kapre_spec(get_stft_magnitude_layer)(
            input_shape=input_shape, 
            n_fft=n_dft, hop_length=n_hop, return_decibel=True,
            input_data_format='channels_first', 
            output_data_format='channels_last')
        y_a = spec(x_a)
    else:  # NOTE: asr - n_dft because we're not padding (I think?)
        input_shape = (n_dft // 2 + 1, int(np.ceil((asr - n_dft) * audio_window_dur / n_hop)), 1)
        x_a = y_a = Input(shape=input_shape, dtype='float32')

    y_a = BatchNormalization()(y_a)

    # CONV BLOCK 1
    n_filter_a_1 = 64
    filt_size_a_1 = (3, 3)
    pool_size_a_1 = (2, 2)
    y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_1, strides=2)(y_a)

    # CONV BLOCK 2
    n_filter_a_2 = 128
    filt_size_a_2 = (3, 3)
    pool_size_a_2 = (2, 2)
    y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_2, strides=2)(y_a)

    # CONV BLOCK 3
    n_filter_a_3 = 256
    filt_size_a_3 = (3, 3)
    pool_size_a_3 = (2, 2)
    y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_3, strides=2)(y_a)

    # CONV BLOCK 4
    n_filter_a_4 = 512
    filt_size_a_4 = (3, 3)
    y_a = Conv2D(n_filter_a_4, filt_size_a_4, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_4, filt_size_a_4,
                 kernel_initializer='he_normal',
                 name='audio_embedding_layer', padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)

    m = Model(inputs=x_a, outputs=y_a)
    return m


def _construct_mel128_audio_network(include_frontend=True):
    """
    Returns an uninitialized model object for an audio network with a Mel
    spectrogram input (with 128 frequency bins).

    Returns
    -------
    model : tf.keras.Model
        Model object.
    """

    weight_decay = 1e-5
    n_dft = 2048
    n_mels = 128
    n_hop = 242
    asr = 48000
    audio_window_dur = 1

    

    if include_frontend:
        # INPUT
        input_shape = (1, asr * audio_window_dur)
        x_a = Input(shape=input_shape, dtype='float32')

        # MELSPECTROGRAM PREPROCESSING
        # 128 x 199 x 1
        from kapre.composed import get_melspectrogram_layer
        spec = __fix_kapre_spec(get_melspectrogram_layer)(
            input_shape=input_shape, 
            n_fft=n_dft, hop_length=n_hop, n_mels=n_mels, 
            sample_rate=asr, return_decibel=True, pad_end=True,
            input_data_format='channels_first', 
            output_data_format='channels_last')
        y_a = spec(x_a)
    else:
        input_shape = (n_mels, int(np.ceil(asr * audio_window_dur / n_hop)), 1)
        x_a = y_a = Input(shape=input_shape, dtype='float32')

    y_a = BatchNormalization()(y_a)

    # CONV BLOCK 1
    n_filter_a_1 = 64
    filt_size_a_1 = (3, 3)
    pool_size_a_1 = (2, 2)
    y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_1, strides=2)(y_a)

    # CONV BLOCK 2
    n_filter_a_2 = 128
    filt_size_a_2 = (3, 3)
    pool_size_a_2 = (2, 2)
    y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_2, strides=2)(y_a)

    # CONV BLOCK 3
    n_filter_a_3 = 256
    filt_size_a_3 = (3, 3)
    pool_size_a_3 = (2, 2)
    y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_3, strides=2)(y_a)

    # CONV BLOCK 4
    n_filter_a_4 = 512
    filt_size_a_4 = (3, 3)
    pool_size_a_4 = (16, 24)
    y_a = Conv2D(n_filter_a_4, filt_size_a_4, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_4, filt_size_a_4,
                 kernel_initializer='he_normal',
                 name='audio_embedding_layer', padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)

    m = Model(inputs=x_a, outputs=y_a)
    return m


def _construct_mel256_audio_network(include_frontend=True):
    """
    Returns an uninitialized model object for an audio network with a Mel
    spectrogram input (with 256 frequency bins).

    Returns
    -------
    model : tf.keras.Model
        Model object.
    """

    weight_decay = 1e-5
    n_dft = 2048
    n_mels = 256
    n_hop = 242
    asr = 48000
    audio_window_dur = 1

    if include_frontend:
        # INPUT
        input_shape = (1, asr * audio_window_dur)
        x_a = Input(shape=input_shape, dtype='float32')

        # MELSPECTROGRAM PREPROCESSING
        # 256 x 199 x 1
        from kapre.composed import get_melspectrogram_layer
        spec = __fix_kapre_spec(get_melspectrogram_layer)(
            input_shape=input_shape, 
            n_fft=n_dft, hop_length=n_hop, n_mels=n_mels,
            sample_rate=asr, return_decibel=True, pad_end=True,
            input_data_format='channels_first', 
            output_data_format='channels_last')
        y_a = spec(x_a)
    else:
        input_shape = (n_mels, int(np.ceil(asr * audio_window_dur / n_hop)), 1)
        x_a = y_a = Input(shape=input_shape, dtype='float32')


    y_a = BatchNormalization()(y_a)

    # CONV BLOCK 1
    n_filter_a_1 = 64
    filt_size_a_1 = (3, 3)
    pool_size_a_1 = (2, 2)
    y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_1, filt_size_a_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_1, strides=2)(y_a)

    # CONV BLOCK 2
    n_filter_a_2 = 128
    filt_size_a_2 = (3, 3)
    pool_size_a_2 = (2, 2)
    y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_2, filt_size_a_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_2, strides=2)(y_a)

    # CONV BLOCK 3
    n_filter_a_3 = 256
    filt_size_a_3 = (3, 3)
    pool_size_a_3 = (2, 2)
    y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_3, filt_size_a_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = MaxPooling2D(pool_size=pool_size_a_3, strides=2)(y_a)

    # CONV BLOCK 4
    n_filter_a_4 = 512
    filt_size_a_4 = (3, 3)
    y_a = Conv2D(n_filter_a_4, filt_size_a_4, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)
    y_a = Conv2D(n_filter_a_4, filt_size_a_4,
                 kernel_initializer='he_normal',
                 name='audio_embedding_layer', padding='same',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)

    m = Model(inputs=x_a, outputs=y_a)
    return m


def _construct_image_network():
    """
    Returns an uninitialized model object for a image network.

    Returns
    -------
    model : tf.keras.Model
        Model object.
    """

    weight_decay = 1e-5
    im_height = 224
    im_width = 224
    num_channels = 3

    x_i = Input(shape=(im_height, im_width, num_channels), dtype='float32')
    y_i = BatchNormalization()(x_i)

    # CONV BLOCK 1
    n_filter_i_1 = 64
    filt_size_i_1 = (3, 3)
    pool_size_i_1 = (2, 2)
    y_i = Conv2D(n_filter_i_1, filt_size_i_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = Conv2D(n_filter_i_1, filt_size_i_1, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_i)
    y_i = Activation('relu')(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = MaxPooling2D(pool_size=pool_size_i_1, strides=2, padding='same')(y_i)

    # CONV BLOCK 2
    n_filter_i_2 = 128
    filt_size_i_2 = (3, 3)
    pool_size_i_2 = (2, 2)
    y_i = Conv2D(n_filter_i_2, filt_size_i_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = Conv2D(n_filter_i_2, filt_size_i_2, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = MaxPooling2D(pool_size=pool_size_i_2, strides=2, padding='same')(y_i)

    # CONV BLOCK 3
    n_filter_i_3 = 256
    filt_size_i_3 = (3, 3)
    pool_size_i_3 = (2, 2)
    y_i = Conv2D(n_filter_i_3, filt_size_i_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = Conv2D(n_filter_i_3, filt_size_i_3, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = MaxPooling2D(pool_size=pool_size_i_3, strides=2, padding='same')(y_i)

    # CONV BLOCK 4
    n_filter_i_4 = 512
    filt_size_i_4 = (3, 3)
    pool_size_i_4 = (28, 28)
    y_i = Conv2D(n_filter_i_4, filt_size_i_4, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_i)
    y_i = BatchNormalization()(y_i)
    y_i = Activation('relu')(y_i)
    y_i = Conv2D(n_filter_i_4, filt_size_i_4,
                 name='vision_embedding_layer', padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_i)

    m = Model(inputs=x_i, outputs=y_i)

    return m


AUDIO_MODELS = {
    'linear': _construct_linear_audio_network,
    'mel128': _construct_mel128_audio_network,
    'mel256': _construct_mel256_audio_network
}
