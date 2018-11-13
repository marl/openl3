.. _tutorial:

openl3 tutorial
===============

Introduction
------------
Welcome to the ``openl3`` tutorial! In this tutorial, we'll explain how ``openl3`` works
and show how to use it to compute audio embeddings for your audio files. Note that only audio
formats supported by `pysoundfile` are supported (e.g. WAV, OGG, FLAC).

.. _using_library:

Using the Library
-----------------


You can easily compute audio embeddings out of the box, like so:

.. code-block:: python

    import openl3
    import soundfile as sf

    audio, sr = sf.read('/path/to/file.wav')
    emb, ts = openl3.get_embedding(audio, sr)

``get_embedding`` returns two objects. The first object ``emb`` is a T-by-D numpy array,
where T is the number of analysis frames used to compute embeddings, and D is the dimensionality
of the embedding (which can be either 6144 or 512). The second object ``ts`` is a length-T
numpy array containing timestamps corresponding to each embedding (to the center of the analysis
window, by default). To use different embedding models, you can use code like the following:

.. code-block:: python

    import openl3
    import soundfile as sf

    audio, sr = sf.read('/path/to/file.wav')
    emb, ts = openl3.get_embedding(audio, sr,
        input_repr="linear", content_type="env")


where we have used an embedding model that uses 128-bin Mel-spectrogram inputs and has been
trained on AudioSet data containing mostly videos of musical performances.
You can also change the embedding dimensionality between 6144 and 512 (default 6144):

.. code-block:: python

    import openl3
    import soundfile as sf
    emb, ts = openl3.get_embedding(audio, sr, embedding_size=512)

By default, ``openl3`` will pad the signal by half of the window size (one second) so that the
the center of the first window corresponds to the beginning of the signal, and the corresponding
timestamps correspond to the center of the window. If you wish to disable this centering, you can
use code like the following:

.. code-block:: python

    import openl3
    import soundfile as sf
    emb, ts = openl3.get_embedding(audio, sr, center=True)

To change the hop size use to compute embeddings (which is 0.1s by default), you can run:

.. code-block:: python

    import openl3
    import soundfile as sf
    emb, ts = openl3.get_embedding(audio, sr, hop_size=0.5)

where we changed the hop size to 0.5 seconds in this example. Finally, you can set the Keras
model verbosity to either 0 or 1:

.. code-block:: python

    import openl3
    import soundfile as sf
    emb, ts = openl3.get_embedding(audio, sr, verbose=0)

To compute embeddings for an audio file and save them locally, you can use code like the following:

.. code-block:: python

    import openl3
    import numpy as np

    audio_filepath = '/path/to/file.wav'
    # Saves the file to '/path/to/file.npz'
    openl3.process_file(audio_filepath)
    # Saves the file to `/different/dir/file.npz`
    openl3.process_file(audio_filepath, output_dir='/different/dir', suffix='suffix')
    # Saves the file to '/path/to/file_suffix.npz'
    openl3.process_file(audio_filepath, suffix='suffix')

    data = np.load('/path/to/file.npz')
    emb, ts = data['embedding'], data['timestamps']

Using the CLI
-------------

To compute embeddings using a single file, the quickest way is to run:

.. code-block:: shell

    $ openl3 /path/to/file.wav

which will create an output file ``/path/to/file.npz`` containing the embedding and
corresponding timestamps. You can change the output directory where the output files are saved, you
can run something like:

.. code-block:: shell

    $ openl3 /path/to/file.wav --output /different/dir

which will create an output file ``/different/dir/file.npz``.
You can also specify multiple input files:

.. code-block:: shell

    $ openl3 /path/to/file1.wav /path/to/file2.wav /path/to/file3.wav


which will create the output files ``/different/dir/file1.npz``, ``/different/dir/file2.npz``,
and ``different/dir/file3.npz``. You can also provide one (or more) directory of files to process:

.. code-block:: shell

    $ openl3 /path/to/audio/dir

This will process all audio files in this directory, though it will not recursively traverse the
directory. You can append a suffix to output files by running something like the following command:

.. code-block:: shell

    $ openl3 /path/to/file.wav --suffix descriptive-suffix

which will create the output file ``/path/to/file_descriptive-suffix.npz``. Optional arguments can
be provided to change the type of embedding model (explained in the API reference).
You can run something like the following:

.. code-block:: shell

    $ openl3 /path/to/file.wav --input-repr mel128 --content-type env

By default, ``--input-repr`` is ``mel128`` and ``--content-type`` is ``music``. Corresponding to
an embedding model that uses 128-bin Mel-spectrogram inputs and has been trained on
AudioSet data containing mostly videos of musical performances. You can also change the embedding
dimensionality between 6144 and 512 (default 6144):

.. code-block:: shell

    $ openl3 /path/to/file.wav --embedding-size 512

By default, ``openl3`` will pad the signal by half of the window size (one second) so that the
the center of the first window corresponds to the beginning of the signal, and the corresponding
timestamps correspond to the center of the window. If you wish to disable this centering, you can
run:

.. code-block:: shell

    $ openl3 /path/to/file.wav --no-centering


To change the hop size use to compute embeddings (which is 0.1s by default), you can run:

.. code-block:: shell

    $ openl3 /path/to/file.wav --hop-size 0.5

where we changed the hop size to 0.5 seconds in this example. Finally, you can suppress non-error
output by running:

.. code-block:: shell

    $ openl3 /path/to/file.wav --quiet
