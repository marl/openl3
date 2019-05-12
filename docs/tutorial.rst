.. _tutorial:

OpenL3 tutorial
===============

Introduction
------------
Welcome to the OpenL3 tutorial! In this tutorial, we'll show how to use OpenL3
to compute audio embeddings for your audio files. The supported audio formats
are those supported by the `pysoundfile` library, which is used for loading the audio (e.g. WAV, OGG, FLAC).

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
where T is the number of embedding frames and D is the dimensionality
of the embedding (which can be either 6144 or 512, details below). The second object ``ts`` is a length-T
numpy array containing timestamps corresponding to each embedding frame (each timestamp corresponds
to the center of each analysis window by default).

By default, OpenL3 extracts embeddings with a model that:

* Is trained on AudioSet videos containing mostly musical performances.
* Uses a mel-spectrogram time-frequency representation with 128 bands
* Returns an embedding of dimensionality 6144

These options defaults can be changed via the following optional parameters:

* content_type: "env", "music" (default)
* input_repr: "linear", "mel128" (default), "mel256"
* embedding_size: 512, 6144 (default)

For example, the following code computes an embedding using a model trained on environmental
videos using a spectrogram with a linear frequency axis and an embedding dimensionality of 512:

.. code-block:: python

    emb, ts = openl3.get_embedding(audio, sr, content_type="env",
                                   input_repr="linear", embedding_size=512)

By default OpenL3 will pad the beginning of the input audio signal by 0.5 seconds (half of the window size) so that the
the center of the first window corresponds to the beginning of the signal ("zero centered"), and the returned timestamps
correspond to the center of each window. You can disable this centering like this:

.. code-block:: python

    emb, ts = openl3.get_embedding(audio, sr, center=False)

The hop size used to extract the embedding is 0.1 seconds by default (i.e. an embedding frame rate of 10 Hz).
In the following example we change the hop size from 0.1 (10 frames per second) to 0.5 (2 frames per second):

.. code-block:: python

    emb, ts = openl3.get_embedding(audio, sr, hop_size=0.5)

Finally, you can silence the Keras printout during inference (verbosity) by changing it from 1 (default) to 0:

.. code-block:: python

    emb, ts = openl3.get_embedding(audio, sr, verbose=0)

By default, the model file is loaded from disk every time ``get_embedding`` is called. To avoid unnecessary I/O when
processing multiple files with the same model, you can load it manually and pass it to the function via the
``model`` parameter:

.. code-block:: python

    model = openl3.models.load_embedding_model(input_repr="mel256", content_type="music",
                                               embedding_size=512)
    emb, ts = openl3.get_embedding(audio, sr, model=model)

Note that when a model is provided via the ``model`` parameter any values passed to the ``input_repr``, ``content_type`` and
``embedding_size`` parameters of ``get_embedding`` will be ignored.

To compute embeddings for an audio file and directly save them to disk you can use ``process_file``:

.. code-block:: python

    import openl3
    import numpy as np

    audio_filepath = '/path/to/file.wav'

    # Save the embedding to '/path/to/file.npz'
    openl3.process_file(audio_filepath)

    # Save the embedding to `/path/to/file_suffix.npz`
    openl3.process_file(audio_filepath, suffix='suffix')

    # Save the embedding to '/different/dir/file_suffix.npz'
    openl3.process_file(audio_filepath, suffix='suffix', output_dir='/different/dir')

The embddings can be loaded from disk using numpy:

.. code-block:: python

    import numpy as np

    data = np.load('/path/to/file.npz')
    emb, ts = data['embedding'], data['timestamps']

As with ``get_embedding`, you can load the model manually and pass it to ``process_file`` to avoid loading the model multiple times:

.. code-block:: python

    import openl3
    import numpy as np

    model = openl3.models.load_embedding_model(input_repr="mel256", content_type="music",
                                               embedding_size=512)

    audio_filepath = '/path/to/file.wav'

    # Save the file to '/path/to/file.npz'
    openl3.process_file(audio_filepath, model=model)

    # Save the file to `/path/to/file_suffix.npz`
    openl3.process_file(audio_filepath, model=model, suffix='suffix')

    # Save the file to '/different/dir/file_suffix.npz'
    openl3.process_file(audio_filepath, model=model, suffix='suffix', output_dir='/different/dir')

Again, note that if a model is provided via the ``model`` parameter, then any values passed to the ``input_repr``, ``content_type`` and ``embedding_size``
parameters of ``process_file`` will be ignored.

Using the Command Line Interface (CLI)
--------------------------------------

To compute embeddings for a single file via the command line run:

.. code-block:: shell

    $ openl3 /path/to/file.wav

This will create an output file at ``/path/to/file.npz``.

You can change the output directory as follows:

.. code-block:: shell

    $ openl3 /path/to/file.wav --output /different/dir

This will create an output file at ``/different/dir/file.npz``.

You can also provide multiple input files:

.. code-block:: shell

    $ openl3 /path/to/file1.wav /path/to/file2.wav /path/to/file3.wav

which will create the output files ``/different/dir/file1.npz``, ``/different/dir/file2.npz``,
and ``different/dir/file3.npz``.

You can also provide one (or more) directories to process:

.. code-block:: shell

    $ openl3 /path/to/audio/dir

This will process all supported audio files in the directory, though it will not recursively traverse the
directory (i.e. audio files in subfolders will not be processed).

You can append a suffix to the output file as follows:

.. code-block:: shell

    $ openl3 /path/to/file.wav --suffix somesuffix

which will create the output file ``/path/to/file_somesuffix.npz``.

Arguments can also be provided to change the model used to extract the embedding including the
content type used for training (music or env), input representation (linear, mel128, mel256),
and output dimensionality (512 or 6144):

.. code-block:: shell

    $ openl3 /path/to/file.wav --content-type env --input-repr mel128 --embedding-size 512

The default value for ``--content-type`` is ``music``, for ``--input-repr`` is ``mel128`` and for ``--embedding-size`` is 512.

By default, OpenL3 will pad the beginning of the input audio signal by 0.5 seconds (half of the window size) so that the
the center of the first window corresponds to the beginning of the signal, and the timestamps correspond to the center of each window.
You can disable this centering as follows:

.. code-block:: shell

    $ openl3 /path/to/file.wav --no-centering

The hop size used to extract the embedding is 0.1 seconds by default (i.e. an embedding frame rate of 10 Hz).
In the following example we change the hop size from 0.1 (10 frames per second) to 0.5 (2 frames per second):

.. code-block:: shell

    $ openl3 /path/to/file.wav --hop-size 0.5

Finally, you can suppress non-error printouts by running:

.. code-block:: shell

    $ openl3 /path/to/file.wav --quiet
