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
    raise NotImplementedError()


def get_embedding_model_path(input_repr, content_type, embedding_size):
    """
    Returns the local path to the model weights file for the model
    with the given characteristics

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
    output_path : str
        Path to given model object

    """
    raise NotImplementedError()


def _construct_linear_audio_network():
    """
    Returns an uninitialized model object for a network with a linear
    spectrogram input (With 257 frequency bins)

    Returns
    -------
    model : keras.models.Model
        Model object.

    """
    raise NotImplementedError()


def _construct_mel128_audio_network():
    """
    Returns an uninitialized model object for a network with a Mel
    spectrogram input (with 128 frequency bins).

    Returns
    -------
    model : keras.models.Model
        Model object.

    """
    raise NotImplementedError()


def _construct_mel256_audio_network():
    """
    Returns an uninitialized model object for a network with a Mel
    spectrogram input (with 256 frequency bins).

    Returns
    -------
    model : keras.models.Model
        Model object.

    """
    raise NotImplementedError()

