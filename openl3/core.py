def get_embedding(audio, sr, input_repr="mel256", content_type="music", embedding_size=6144,
                  center=True, hop_size=0.1, verbose=1):
    """
    Computes and returns L3 embedding for given audio data

    Parameters
    ----------
    audio : np.ndarray [shape=(N,) or (N,C)]
        1D numpy array of audio data.
    sr : int
        Sampling rate, if not 48kHz will audio will be resampled.
    input_repr : "linear", "mel128", or "mel256"
        Spectrogram representation used for model.
    content_type : "music" or "env"
        Type of content used to train embedding.
    embedding_size : 6144 or 512
        Embedding dimensionality.
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
    raise NotImplementedError()


def process_file(filepath, output_filepath=None, input_repr="mel256", content_type="music",
                 embedding_size=6144, center=True, hop_size=0.1, verbose=True):
    """
    Computes and saves L3 embedding for given audio file

    Parameters
    ----------
    filepath : str
        Path to WAV file to be processed.
    output_filepath : str or None
        Path to directory for saving output files. If None, output files will
        be saved to the directory containing the input file.
    input_repr : "linear", "mel128", or "mel256"
        Spectrogram representation used for model.
    content_type : "music" or "env"
        Type of content used to train embedding.
    embedding_size : 6144 or 512
        Embedding dimensionality.
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
    raise NotImplementedError()


def get_output_path(filepath, suffix, output_dir):
    """

    Parameters
    ----------
    filepath : str
        Path to audio file to be processed
    suffix : str
        String to append to filename (including extension)
    output_dir : str
        Path to directory where file will be saved

    Returns
    -------
    output_path : str
        Path to output file

    """
    raise NotImplementedError()
