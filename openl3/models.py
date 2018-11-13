import os
import warnings

with warnings.catch_warnings():
    # Suppress TF and Keras warnings when importing
    warnings.simplefilter("ignore")
    from kapre.time_frequency import Spectrogram, Melspectrogram
    from keras.layers import (
        Input, Conv2D, BatchNormalization, MaxPooling2D,
        Flatten, Activation, Lambda
    )
    from keras.models import Model
    import keras.regularizers as regularizers


POOLINGS = {
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


def get_embedding_model(input_repr, content_type, embedding_size):
    """
    Returns a model with the given characteristics. Loads the model
    if the model has not been loaded yet.

    Parameters
    ----------
    input_repr : "linear", "mel128", or "mel256"
        Spectrogram representation used for model.
    content_type : "music" or "env"
        Type of content used to train embedding.
    embedding_size : 6144 or 512
        Embedding dimensionality.

    Returns
    -------
    model : keras.models.Model
        Model object.
    """

    # Construct embedding model and load model weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MODELS[input_repr]()

    m.load_weights(get_embedding_model_path(input_repr, content_type))

    # Pooling for final output embedding size
    pool_size = POOLINGS[input_repr][embedding_size]
    y_a = MaxPooling2D(pool_size=pool_size, padding='same')(m.output)
    y_a = Flatten()(y_a)
    m = Model(inputs=m.input, outputs=y_a)
    return m


def get_embedding_model_path(input_repr, content_type):
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
                        'models',
                        'openl3_audio_{}_{}.h5'.format(input_repr, content_type))


def _construct_linear_audio_network():
    """
    Returns an uninitialized model object for a network with a linear
    spectrogram input (With 257 frequency bins)

    Returns
    -------
    model : keras.models.Model
        Model object.
    """

    weight_decay = 1e-5
    n_dft = 512
    n_hop = 242
    asr = 48000
    audio_window_dur = 1

    # INPUT
    x_a = Input(shape=(1, asr * audio_window_dur), dtype='float32')

    # SPECTROGRAM PREPROCESSING
    # 257 x 199 x 1
    y_a = Spectrogram(n_dft=n_dft, n_hop=n_hop, power_spectrogram=1.0,
                      return_decibel_spectrogram=True, padding='valid')(x_a)
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
    pool_size_a_4 = (32, 24)
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


def _construct_mel128_audio_network():
    """
    Returns an uninitialized model object for a network with a Mel
    spectrogram input (with 128 frequency bins).

    Returns
    -------
    model : keras.models.Model
        Model object.
    """

    weight_decay = 1e-5
    n_dft = 2048
    n_mels = 128
    n_hop = 242
    asr = 48000
    audio_window_dur = 1

    # INPUT
    x_a = Input(shape=(1, asr * audio_window_dur), dtype='float32')

    # MELSPECTROGRAM PREPROCESSING
    # 128 x 199 x 1
    y_a = Melspectrogram(n_dft=n_dft, n_hop=n_hop, n_mels=n_mels,
                      sr=asr, power_melgram=1.0, htk=True,
                      return_decibel_melgram=True, padding='same')(x_a)
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


def _construct_mel256_audio_network():
    """
    Returns an uninitialized model object for a network with a Mel
    spectrogram input (with 256 frequency bins).

    Returns
    -------
    model : keras.models.Model
        Model object.
    """

    weight_decay = 1e-5
    n_dft = 2048
    n_mels = 256
    n_hop = 242
    asr = 48000
    audio_window_dur = 1

    # INPUT
    x_a = Input(shape=(1, asr * audio_window_dur), dtype='float32')

    # MELSPECTROGRAM PREPROCESSING
    # 128 x 199 x 1
    y_a = Melspectrogram(n_dft=n_dft, n_hop=n_hop, n_mels=n_mels,
                      sr=asr, power_melgram=1.0, htk=True, # n_win=n_win,
                      return_decibel_melgram=True, padding='same')(x_a)
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
    pool_size_a_4 = (32, 24)
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


MODELS = {
    'linear': _construct_linear_audio_network,
    'mel128': _construct_mel128_audio_network,
    'mel256': _construct_mel256_audio_network
}
