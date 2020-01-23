.. _changes:

Changelog
---------

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
