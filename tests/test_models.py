from openl3.models import get_embedding_model, get_embedding_model_path


def test_get_embedding_model_path():
    for input_repr in ('linear', 'mel128', 'mel256'):
        for content_type in ('music', 'env'):
            for embedding_size in (512, 6144):
                model_path = get_embedding_model_path(input_repr=input_repr,
                                                      content_type=content_type,
                                                      embedding_size=embedding_size)
                assert model_path == 'openl3/openl3_audio_{}_{}.h5'.format(input_repr, content_type)
