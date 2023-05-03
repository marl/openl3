.. _changes:

Changelog
---------

v0.4.2
~~~~~~
 - Fix incorrect embedding_size in ``load_image_embedding_model`` docstring
 - Add ``tensorflow.keras`` mock modules to ``docs/conf.py`` to fix docs build
 - Remove pin on ``sphinx`` version
 - Add note about training bug in README

v0.4.1
~~~~~~
 - Add librosa as an explicit dependency
 - Remove upper limit pinning for scikit-image dependency
 - Fix version number typo in README
 - Update TensorFlow information in README

v0.4.0
~~~~~~
 - Upgraded to `tensorflow>=2.0.0`. Tensorflow is now included as a dependency because of dual CPU-GPU support.
 - Upgraded to `kapre>=0.3.5`. Reverted magnitude scaling method to match `kapre<=0.1.4` as that's what the model was trained on.
 - Removed Python 2/3.5 support as they are not supported by Tensorflow 2 (and added 3.7 & 3.8)
 - Add librosa frontend, and allow frontend to be configurable between `kapre` and `librosa`
    - Added ``frontend='kapre'`` parameter to ``get_audio_embedding``, ``process_audio_file``, and ``load_audio_embedding_model``
    - Added ``audio_frontend='kapre'`` parameter to ``process_video_file`` and the CLI
    - Added `frontend='librosa'` flag to `load_audio_embedding_model` for use with a librosa or other external frontend
    - Added a `openl3.preprocess_audio` function that computes the input features needed for each frontend
 - Model .h5 no longer have Kapre layers in them and are all importable from ``tf.keras``
 - Made ``skimage`` and ``moviepy.video.io.VideoFileClip import VideoFileClip`` use lazy imports
 - Added new regression data for both Kapre 0.3.5 and Librosa
 - Parameterized some of the tests to reduce duplication
 - Added developer helpers for regression data, weight packaging, and .h5 file manipulation


v0.3.1
~~~~~~
 - Require `keras>=2.0.9,<2.3.0` in dependencies to avoid force installation of TF 2.x during pip installation.
 - Update README and installation docs to explicitly state that we do not yet support TF 2.x and to offer a working dependency combination.
 - Require `kapre==0.1.4` in dependencies to avoid installing `tensorflow>=1.14` which break regression tests.


v0.3.0
~~~~~~
 - Rename audio related embedding functions to indicate that they are specific to audio.
 - Add image embedding functionality to API and CLI.
 - Add video processing functionality to API and CLI.
 - Add batch processing functionality to API and CLI to more efficiently process multiple inputs.
 - Update documentation with new functionality.
 - Address build issues with updated dependencies.

v0.2.0
~~~~~~
 - Update embedding models with ones that have been trained with the kapre bug fixed.
 - Allow loaded models to be passed in and used in `process_file` and `get_embedding`.
 - Rename `get_embedding_model` to `load_embedding_model`.

v0.1.1
~~~~~~
 - Update kapre to fix issue with dynamic range normalization for decibel computation when computing spectrograms.

v0.1.0
~~~~~~
 - First release.
