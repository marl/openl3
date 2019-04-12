from openl3.models import load_embedding_model, load_embedding_model_path


def test_load_embedding_model_path():
    embedding_model_path = load_embedding_model_path('linear', 'music')
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'openl3/openl3_audio_linear_music.h5'

    embedding_model_path = load_embedding_model_path('linear', 'env')
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'openl3/openl3_audio_linear_env.h5'

    embedding_model_path = load_embedding_model_path('mel128', 'music')
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'openl3/openl3_audio_mel128_music.h5'

    embedding_model_path = load_embedding_model_path('mel128', 'env')
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'openl3/openl3_audio_mel128_env.h5'

    embedding_model_path = load_embedding_model_path('mel256', 'music')
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'openl3/openl3_audio_mel256_music.h5'

    embedding_model_path = load_embedding_model_path('mel256', 'env')
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'openl3/openl3_audio_mel256_env.h5'


def test_load_embedding_model():
    m = load_embedding_model('linear', 'music', 6144)
    assert m.output_shape[1] == 6144

    m = load_embedding_model('linear', 'music', 512)
    assert m.output_shape[1] == 512

    m = load_embedding_model('linear', 'env', 6144)
    assert m.output_shape[1] == 6144

    m = load_embedding_model('linear', 'env', 512)
    assert m.output_shape[1] == 512

    m = load_embedding_model('mel128', 'music', 6144)
    assert m.output_shape[1] == 6144

    m = load_embedding_model('mel128', 'music', 512)
    assert m.output_shape[1] == 512

    m = load_embedding_model('mel128', 'env', 6144)
    assert m.output_shape[1] == 6144

    m = load_embedding_model('mel128', 'env', 512)
    assert m.output_shape[1] == 512

    m = load_embedding_model('mel256', 'music', 6144)
    assert m.output_shape[1] == 6144

    m = load_embedding_model('mel256', 'music', 512)
    assert m.output_shape[1] == 512

    m = load_embedding_model('mel256', 'env', 6144)
    assert m.output_shape[1] == 6144

    m = load_embedding_model('mel256', 'env', 512)
    assert m.output_shape[1] == 512
