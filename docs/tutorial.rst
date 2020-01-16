.. _tutorial:

OpenL3 tutorial
===============

Introduction
------------
Welcome to the OpenL3 tutorial! In this tutorial, we'll show how to use OpenL3
to compute embeddings for your audio files, images, and videos. The supported audio formats
are those supported by the ``pysoundfile`` library, which is used for loading the
audio (e.g. WAV, OGG, FLAC). The supported image formats are those supported by
the `scikit-image`` library (e.g. PNG, JPEG). The supported video formats are those
supported by ``moviepy`` (and therefore ``ffmpeg``) (e.g. MP4).

.. _using_library:

Using the Library
-----------------

Extracting audio embeddings
***************************


You can easily compute audio embeddings out of the box, like so:

.. code-block:: python

    import openl3
    import soundfile as sf

    audio, sr = sf.read('/path/to/file.wav')
    emb, ts = openl3.get_audio_embedding(audio, sr)

``get_audio_embedding`` returns two objects. The first object ``emb`` is a T-by-D numpy array,
where T is the number of embedding frames and D is the dimensionality
of the embedding (which can be either 6144 or 512, details below). The second object ``ts`` is a length-T
numpy array containing timestamps corresponding to each embedding frame (each timestamp corresponds
to the center of each analysis window by default).

Multiple audio arrays can be provided (with a specified model input batch size for processing, i.e. the number of audio windows the model processes at a time) as follows:

.. code-block:: python

    import openl3
    import soundfile as sf

    audio1, sr1 = sf.read('/path/to/file1.wav')
    audio2, sr2 = sf.read('/path/to/file2.wav')
    audio3, sr3 = sf.read('/path/to/file3.wav')
    audio_list = [audio1, audio2, audio3]
    sr_list = [sr1, sr2, sr3]

    # Pass in a list of audio arrays and sample rates
    emb_list, ts_list = openl3.get_audio_embedding(audio_list, sr_list, batch_size=32)
    # If all arrays use sample rate, can just pass in one sample rate
    emb_list, ts_list = openl3.get_audio_embedding(audio_list, sr1, batch_size=32)

Here, we get a list of embeddings and timestamp arrays for each of the input arrays.

By default, OpenL3 extracts audio embeddings with a model that:

* Is trained on AudioSet videos containing mostly musical performances
* Uses a mel-spectrogram time-frequency representation with 128 bands
* Returns an embedding of dimensionality 6144 for each embedding frame

These defaults can be changed via the following optional parameters:

* content_type: "env", "music" (default)
* input_repr: "linear", "mel128" (default), "mel256"
* embedding_size: 512, 6144 (default)

For example, the following code computes an embedding using a model trained on environmental
videos using a spectrogram with a linear frequency axis and an embedding dimensionality of 512:

.. code-block:: python

    emb, ts = openl3.get_audio_embedding(audio, sr, content_type="env",
                                   input_repr="linear", embedding_size=512)

By default OpenL3 will pad the beginning of the input audio signal by 0.5 seconds (half of the window size) so that the
the center of the first window corresponds to the beginning of the signal ("zero centered"), and the returned timestamps
correspond to the center of each window. You can disable this centering like this:

.. code-block:: python

    emb, ts = openl3.get_audio_embedding(audio, sr, center=False)

The hop size used to extract the embedding is 0.1 seconds by default (i.e. an embedding frame rate of 10 Hz).
In the following example we change the hop size from 0.1 (10 frames per second) to 0.5 (2 frames per second):

.. code-block:: python

    emb, ts = openl3.get_audio_embedding(audio, sr, hop_size=0.5)

Finally, you can silence the Keras printout during inference (verbosity) by changing it from 1 (default) to 0:

.. code-block:: python

    emb, ts = openl3.get_audio_embedding(audio, sr, verbose=0)

By default, the model file is loaded from disk every time ``get_audio_embedding`` is called. To avoid unnecessary I/O when
processing multiple files with the same model, you can load it manually and pass it to the function via the
``model`` parameter:

.. code-block:: python

    model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music",
                                                     embedding_size=512)
    emb1, ts1 = openl3.get_audio_embedding(audio1, sr1, model=model)
    emb2, ts2 = openl3.get_audio_embedding(audio2, sr2, model=model)

Note that when a model is provided via the ``model`` parameter any values passed to the ``input_repr``, ``content_type`` and
``embedding_size`` parameters of ``get_audio_embedding`` will be ignored.

To compute embeddings for an audio file and directly save them to disk you can use ``process_audio_file``:

.. code-block:: python

    import openl3
    import numpy as np

    audio_filepath = '/path/to/file.wav'

    # Save the embedding to '/path/to/file.npz'
    openl3.process_audio_file(audio_filepath)

    # Save the embedding to `/path/to/file_suffix.npz`
    openl3.process_audio_file(audio_filepath, suffix='suffix')

    # Save the embedding to '/different/dir/file_suffix.npz'
    openl3.process_audio_file(audio_filepath, suffix='suffix', output_dir='/different/dir')

The embddings can be loaded from disk using numpy:

.. code-block:: python

    import numpy as np

    data = np.load('/path/to/file.npz')
    emb, ts = data['embedding'], data['timestamps']

Multiple files can be processed as well (with a specified model input batch size used for processing, i.e. the number of audio windows the model processes at a time) as follows:

.. code-block:: python

    import openl3
    import numpy as np

    audio_filepath1 = '/path/to/file1.wav'
    audio_filepath2 = '/path/to/file2.wav'
    audio_filepath3 = '/path/to/file3.wav'
    audio_filepath_list = [audio_filepath1, audio_filepath2, audio_filepath3]

    # Saves embeddings to '/path/to/file1.npz', '/path/to/file2.npz', and '/path/to/file3.npz'
    openl3.process_audio_file(audio_filepath_list, batch_size=32)


As with ``get_audio_embedding``, you can load the model manually and pass it to ``process_audio_file`` to avoid loading the model multiple times:

.. code-block:: python

    import openl3
    import numpy as np

    model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music",
                                                     embedding_size=512)

    audio_filepath = '/path/to/file.wav'

    # Save the file to '/path/to/file.npz'
    openl3.process_audio_file(audio_filepath, model=model)

    # Save the file to `/path/to/file_suffix.npz`
    openl3.process_audio_file(audio_filepath, model=model, suffix='suffix')

    # Save the file to '/different/dir/file_suffix.npz'
    openl3.process_audio_file(audio_filepath, model=model, suffix='suffix', output_dir='/different/dir')

Again, note that if a model is provided via the ``model`` parameter, then any values passed to the ``input_repr``, ``content_type`` and ``embedding_size``
parameters of ``process_audio_file`` will be ignored.

Extracting image embeddings
***************************

You can easily compute image embeddings out of the box, like so:

.. code-block:: python

    import openl3
    from skimage.io import imread

    image = imread('/path/to/file.png)
    emb = openl3.get_image_embedding(image, content_type="env",
                                     input_repr="linear", embedding_size=512)

    # Preload model
    model = openl3.models.load_image_embedding_model(input_repr="mel256", content_type="music",
                                                     embedding_size=512)
    emb = openl3.get_image_embedding(image, model=model)



When given a single image, ``get_image_embedding`` returns a size D numpy array,
where D is the dimensionality of the embedding (which can be either 8192 or 512).

A sequence of images (e.g. from a video) can also be provided:

.. code-block:: python

    from moviepy.video.io.VideoFileClip import VideoFileClip

    video_filepath = '/path/to/file.mp4'

    # Load video and get image frames
    clip = VideoFileClip(video_filepath)
    images = np.array([frame for frame in clip.iter_frames()])

    emb = get_image_embedding(images, batch_size=32)

When given a sequence of images (as a numpy array), ``get_image_embedding`` returns an N-by-D numpy array,
where N is the number of embedding frames (corresponding to each video frame) and D is the dimensionality
of the embedding (which can be either 8192 or 512).

.. code-block:: python

    from moviepy.video.io.VideoFileClip import VideoFileClip

    video_filepath = '/path/to/file.mp4'

    # Load video and get image frames
    clip = VideoFileClip(video_filepath)
    images = np.array([frame for frame in clip.iter_frames()])

    # If the frame rate is provided, returns an array of timestamps
    emb, ts = get_image_embedding(images, frame_rate=clip.fps, batch_size=32)

When given a sequence of images (as a numpy array) and a frame rate, ``get_image_embedding`` returns two objects.
The first object ``emb`` is an N-by-D numpy array, where N is the number of embedding frames and D is the dimensionality
of the embedding (which can be either 8192 or 512, details below). The second object ``ts`` is a length-N
numpy array containing timestamps corresponding to each embedding frame (corresponding to each video frame).

Multiple sequences of images can be provided as well:

.. code-block:: python

    from moviepy.video.io.VideoFileClip import VideoFileClip

    video_filepath1 = '/path/to/file1.mp4'
    video_filepath2 = '/path/to/file2.mp4'
    video_filepath3 = '/path/to/file2.mp4'

    # Load video and get image frames
    clip1 = VideoFileClip(video_filepath1)
    images1 = np.array([frame for frame in clip1.iter_frames()])
    clip2 = VideoFileClip(video_filepath2)
    images2 = np.array([frame for frame in clip2.iter_frames()])
    clip3 = VideoFileClip(video_filepath3)
    images3 = np.array([frame for frame in clip3.iter_frames()])

    image_list = [images1, images2, images3]
    frame_rate_list = [clip1.fps, clip2.fps, clip3.fps]

    # If the frame rates is provided...
    emb_list, ts_list = get_image_embedding(image_list, frame_rate=frame_rate_list, batch_size=32)
    # or if a single frame rate applying to all sequences is provided, returns a list of timestamps
    emb_list, ts_list = get_image_embedding(image_list, frame_rate=clip1.fps, batch_size=32)
    # ...otherwise, just the embeddings are returnedj
    emb_list = get_image_embedding(image_list, batch_size=32)


Here, we get a list of embeddings for each sequence. If a frame rate (or list of frame rates) is given,
timestamp arrays for each of the input arrays are returned as well.

By default, OpenL3 extracts image embeddings with a model that:

* Is trained on AudioSet videos containing mostly musical performances
* Is trained using a mel-spectrogram time-frequency representation with 128 bands (for the audio embedding model)
* Returns an embedding of dimensionality 8192 for each embedding frame

These defaults can be changed via the following optional parameters:

* content_type: "env", "music" (default)
* input_repr: "linear", "mel128" (default), "mel256"
* embedding_size: 512, 8192 (default)

For example, the following code computes an embedding using a model trained on environmental
videos using an audio spectrogram with a linear frequency axis and an embedding dimensionality of 512:

.. code-block:: python

    emb = openl3.get_image_embedding(image, content_type="env",
                                     input_repr="linear", embedding_size=512)

Finally, you can silence the Keras printout during inference (verbosity) by changing it from 1 (default) to 0:

.. code-block:: python

    emb = openl3.get_image_embedding(image, verbose=0)

By default, the model file is loaded from disk every time ``get_image_embedding`` is called. To avoid unnecessary I/O when
processing multiple files with the same model, you can load it manually and pass it to the function via the
``model`` parameter:

.. code-block:: python

    model = openl3.models.load_image_embedding_model(input_repr="mel256", content_type="music",
                                                     embedding_size=512)
    emb1 = openl3.get_image_embedding(image1, model=model)
    emb2 = openl3.get_image_embedding(image2, model=model)

Note that when a model is provided via the ``model`` parameter any values passed to the ``input_repr``, ``content_type`` and
``embedding_size`` parameters of ``get_image_embedding`` will be ignored.


Image files can also be processed with just the filepath:

.. code-block:: python

    import openl3
    import numpy as np

    image_filepath = '/path/to/file.png'

    # Save the file to '/path/to/file.npz'
    openl3.process_image_file(image_filepath)

    # Preload model
    model = openl3.models.load_image_embedding_model(input_repr="mel256", content_type="music",
                                                     embedding_size=512)
    openl3.process_image_file(image_filepath, model=model)

    # Save the file to `/path/to/file_suffix.npz`
    openl3.process_image_file(image_filepath, model=model, suffix='suffix')

    # Save the file to '/different/dir/file_suffix.npz'
    openl3.process_image_file(image_filepath, model=model, suffix='suffix', output_dir='/different/dir')

The embdding can be loaded from disk using numpy:

.. code-block:: python

    import numpy as np

    data = np.load('/path/to/file.npz')
    emb = data['embedding']

Multiple files can be processed as well (with a specified model input batch size used for processing, i.e. the number of audio windows the model processes at a time) as follows:

.. code-block:: python

    import openl3
    import numpy as np

    image_filepath1 = '/path/to/file1.png'
    image_filepath2 = '/path/to/file2.png'
    image_filepath3 = '/path/to/file3.png'
    image_filepath_list = [image_filepath1, image_filepath2, image_filepath3]

    # Saves embeddings to '/path/to/file1.npz', '/path/to/file2.npz', and '/path/to/file3.npz'
    openl3.process_image_file(image_filepath_list, batch_size=32)


As with ``get_image_embedding``, you can load the model manually and pass it to ``process_image_file`` to avoid loading the model multiple times:

.. code-block:: python

    import openl3
    import numpy as np

    model = openl3.models.load_image_embedding_model(input_repr="mel256", content_type="music",
                                                     embedding_size=512)

    image_filepath = '/path/to/file.png'

    # Save the file to '/path/to/file.npz'
    openl3.process_image_file(image_filepath, model=model)

    # Save the file to `/path/to/file_suffix.npz`
    openl3.process_image_file(image_filepath, model=model, suffix='suffix')

    # Save the file to '/different/dir/file_suffix.npz'
    openl3.process_image_file(image_filepath, model=model, suffix='suffix', output_dir='/different/dir')

Again, note that if a model is provided via the ``model`` parameter, then any values passed to the ``input_repr``, ``content_type`` and ``embedding_size``
parameters of ``process_image_file`` will be ignored.


Processing video files
**********************

Video files can be processed to extract both audio and image embeddings.
Please note that the audio and image embeddings are not synchronized, so the respective timestamps for
each modality will not generally be aligned.
Image embeddings are computed for every frame of the video, while the specified audio hop size is used
for chunking the audio signal in the video. Additionally, please note that available embedding sizes differ
for audio and image embeddings. Video files can be processed as follows:

.. code-block:: python

    import openl3
    import numpy as np

    video_filepath = '/path/to/file.mp4'

    # Save audio embedding to '/path/to/file_audio.npz'
    # and image embedding to '/path/to/file_image.npz'
    openl3.process_video_file(video_filepath)

    # Preload models
    audio_model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music",
                                                           embedding_size=512)
    image_model = openl3.models.load_image_embedding_model(input_repr="mel256", content_type="music",
                                                           embedding_size=512)
    openl3.process_video_file(video_filepath, audio_model=audio_model, image_model=image_model)

    # Save audio embedding to '/path/to/file_audio_suffix.npz'
    # and image embedding to '/path/to/file_image_suffix.npz'
    openl3.process_video_file(video_filepath, audio_model=audio_model, image_model=image_model,
                              suffix='suffix')

    # Save audio embedding to '/different/dir/file_audio_suffix.npz'
    # and image embedding to '/different/dir/file_image_suffix.npz'
    openl3.process_video_file(video_filepath, audio_model=audio_model, image_model=image_model,
                              suffix='suffix', output_dir='/different/dir')

Multiple files can be processed as well (with a specified model input batch size used for processing, i.e. the number of video frames the model processes at a time) as follows:

.. code-block:: python

    import openl3
    import numpy as np

    video_filepath1 = '/path/to/file1.mp4'
    video_filepath2 = '/path/to/file2.mp4'
    video_filepath3 = '/path/to/file3.mp4'
    video_filepath_list = [video_filepath1, video_filepath2, video_filepath3]

    # Saves audio embeddings to '/path/to/file1_audio.npz', '/path/to/file2_audio.npz',
    # and '/path/to/file3_audio.npz' and saves image embeddings to
    # '/path/to/file1_image.npz', '/path/to/file2_image.npz', and '/path/to/file3_image.npz'
    openl3.process_video_file(video_filepath_list, batch_size=32)


Using the Command Line Interface (CLI)
--------------------------------------

Extracting audio embeddings
***************************

To compute embeddings for a single audio file via the command line run:

.. code-block:: shell

    $ openl3 audio /path/to/file.wav

This will create an output file at ``/path/to/file.npz``.

You can change the output directory as follows:

.. code-block:: shell

    $ openl3 audio /path/to/file.wav --output /different/dir

This will create an output file at ``/different/dir/file.npz``.

You can also provide multiple input files:

.. code-block:: shell

    $ openl3 audio /path/to/file1.wav /path/to/file2.wav /path/to/file3.wav

which will create the output files ``/different/dir/file1.npz``, ``/different/dir/file2.npz``,
and ``different/dir/file3.npz``.

You can also provide one (or more) directories to process:

.. code-block:: shell

    $ openl3 audio /path/to/audio/dir

This will process all supported audio files in the directory, though it will not recursively traverse the
directory (i.e. audio files in subfolders will not be processed).

You can append a suffix to the output file as follows:

.. code-block:: shell

    $ openl3 audio /path/to/file.wav --suffix somesuffix

which will create the output file ``/path/to/file_somesuffix.npz``.

Arguments can also be provided to change the model used to extract the embedding including the
content type used for training (music or env), input representation (linear, mel128, mel256),
and output dimensionality (512 or 6144), for example:

.. code-block:: shell

    $ openl3 audio /path/to/file.wav --content-type env --input-repr mel128 --embedding-size 512

The default value for --content-type is music, for --input-repr is mel128 and for --embedding-size is 6144.

By default, OpenL3 will pad the beginning of the input audio signal by 0.5 seconds (half of the window size) so that the
the center of the first window corresponds to the beginning of the signal, and the timestamps correspond to the center of each window.
You can disable this centering as follows:

.. code-block:: shell

    $ openl3 audio /path/to/file.wav --no-audio-centering

The hop size used to extract the embedding is 0.1 seconds by default (i.e. an embedding frame rate of 10 Hz).
In the following example we change the hop size from 0.1 (10 frames per second) to 0.5 (2 frames per second):

.. code-block:: shell

    $ openl3 audio /path/to/file.wav --audio-hop-size 0.5

You can change the batch size used as the input to the audio embedding model (i.e. the number of audio windows the model processes at a time) to one appropriate for your
computational resources:

.. code-block:: shell

    $ openl3 audio /path/to/file.wav --audio-batch-size 16

Finally, you can suppress non-error printouts by running:

.. code-block:: shell

    $ openl3 audio /path/to/file.wav --quiet

Extracting image embeddings
***************************

To compute embeddings for a single image file via the command line run:

.. code-block:: shell

    $ openl3 image /path/to/file.png

This will create an output file at ``/path/to/file.npz``.

You can change the output directory as follows:

.. code-block:: shell

    $ openl3 image /path/to/file.png --output /different/dir

This will create an output file at ``/different/dir/file.npz``.

You can also provide multiple input files:

.. code-block:: shell

    $ openl3 image /path/to/file1.png /path/to/file2.png /path/to/file3.png

which will create the output files ``/different/dir/file1.npz``, ``/different/dir/file2.npz``,
and ``different/dir/file3.npz``.

You can also provide one (or more) directories to process:

.. code-block:: shell

    $ openl3 image /path/to/image/dir

This will process all supported image files in the directory, though it will not recursively traverse the
directory (i.e. image files in subfolders will not be processed).

You can append a suffix to the output file as follows:

.. code-block:: shell

    $ openl3 image /path/to/file.png --suffix somesuffix

which will create the output file ``/path/to/file_somesuffix.npz``.

Arguments can also be provided to change the model used to extract the embedding including the
content type used for training (music or env), input representation (linear, mel128, mel256),
and output dimensionality (512 or 8192), for example:

.. code-block:: shell

    $ openl3 image /path/to/file.png --content-type env --input-repr mel128 --embedding-size 512

The default value for --content-type is music, for --input-repr is mel128 and for --embedding-size is 8192.

You can change the batch size used as the input to the
image embedding model (i.e. the number of video frames the model processes at a time) to one appropriate for your computational resources:

.. code-block:: shell

    $ openl3 image /path/to/file.png --image-batch-size 16

Finally, you can suppress non-error printouts by running:

.. code-block:: shell

    $ openl3 image /path/to/file.png --quiet



Processing video files
**********************

To compute embeddings for a single video file via the command line run:

.. code-block:: shell

    $ openl3 video /path/to/file.mp4

This will create output files at ``/path/to/file_audio.npz`` and ``/path/to/file_image.npz``
for the audio and image embeddings, respectively. Please note that the audio and image embeddings are
not synchronized, so the respective timestamps for each modality will not generally be aligned.
Image embeddings are computed for every frame of the video, while the specified audio hop size is used
for chunking the audio signal in the video. Additionally, please note that available embedding sizes differ
for audio and image embeddings.

Functionality regarding specifying models, multiple input files, verbosity, output directories, and suffixes
behave the same as with extracting audio and image embeddings. Functionality specific to audio or image embeddings
can also be specified as previously specified.

You can change the batch size used as the input to the audio embedding and image embedding models (i.e. the number of audio windows or video frames, respectively, that the model processes at a time)
to one appropriate for your computational resources:

.. code-block:: shell

    $ openl3 video /path/to/file.mp4 --audio-batch-size 16 --image-batch-size 16
