.. _changes:

Changelog
---------

v0.4.0
~~~~~~
 - Upgrade to `tensorflow>=2.0.0`
 - Upgrade to `kapre>=`. Reverted magnitude scaling method to match `<=` as that's what the model was trained on.
 - Add librosa frontend, and allow frontend to be configurable between `kapre` and `librosa`
    - Added `include_frontend=False` flag to `load_audio_embedding_model` for use with a librosa or other external frontend
    - Added a `openl3.preprocess` function that computes the input features needed for each frontend


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
