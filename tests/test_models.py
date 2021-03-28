from openl3.models import (
    load_audio_embedding_model, get_audio_embedding_model_path,
    load_image_embedding_model, get_image_embedding_model_path)


def test_get_audio_embedding_model_path():
    embedding_model_path = get_audio_embedding_model_path('linear', 'music')
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'openl3/openl3_audio_linear_music.h5'

    embedding_model_path = get_audio_embedding_model_path('linear', 'env')
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'openl3/openl3_audio_linear_env.h5'

    embedding_model_path = get_audio_embedding_model_path('mel128', 'music')
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'openl3/openl3_audio_mel128_music.h5'

    embedding_model_path = get_audio_embedding_model_path('mel128', 'env')
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'openl3/openl3_audio_mel128_env.h5'

    embedding_model_path = get_audio_embedding_model_path('mel256', 'music')
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'openl3/openl3_audio_mel256_music.h5'

    embedding_model_path = get_audio_embedding_model_path('mel256', 'env')
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'openl3/openl3_audio_mel256_env.h5'


def test_load_audio_embedding_model():
    import kapre

    m = load_audio_embedding_model('linear', 'music', 6144)
    # assert isinstance(m.layers[1], kapre.time_frequency.Spectrogram)
    assert m.layers[1].output_shape == (None, 257, 199, 1)
    assert m.output_shape[1] == 6144

    first_model = m

    m = load_audio_embedding_model('linear', 'music', 512)
    # assert isinstance(m.layers[1], kapre.time_frequency.Spectrogram)
    assert m.layers[1].output_shape == (None, 257, 199, 1)
    assert m.output_shape[1] == 512
    # Check model consistency
    assert isinstance(m.layers[0], type(first_model.layers[0]))
    assert len(m.layers) == len(first_model.layers)
    assert all([isinstance(l1, type(l2))
                for (l1, l2) in zip(m.layers[2:], first_model.layers[2:])])

    m = load_audio_embedding_model('linear', 'env', 6144)
    # assert isinstance(m.layers[1], kapre.time_frequency.Spectrogram)
    assert m.layers[1].output_shape == (None, 257, 199, 1)
    assert m.output_shape[1] == 6144
    # Check model consistency
    assert isinstance(m.layers[0], type(first_model.layers[0]))
    assert len(m.layers) == len(first_model.layers)
    assert all([isinstance(l1, type(l2))
                for (l1, l2) in zip(m.layers[2:], first_model.layers[2:])])

    m = load_audio_embedding_model('linear', 'env', 512)
    # assert isinstance(m.layers[1], kapre.time_frequency.Spectrogram)
    assert m.layers[1].output_shape == (None, 257, 199, 1)
    assert m.output_shape[1] == 512
    # Check model consistency
    assert isinstance(m.layers[0], type(first_model.layers[0]))
    assert len(m.layers) == len(first_model.layers)
    assert all([isinstance(l1, type(l2))
                for (l1, l2) in zip(m.layers[2:], first_model.layers[2:])])

    m = load_audio_embedding_model('mel128', 'music', 6144)
    # assert isinstance(m.layers[1], kapre.time_frequency.Melspectrogram)
    assert m.layers[1].output_shape == (None, 128, 199, 1)
    assert m.output_shape[1] == 6144
    # Check model consistency
    assert isinstance(m.layers[0], type(first_model.layers[0]))
    assert len(m.layers) == len(first_model.layers)
    assert all([isinstance(l1, type(l2))
                for (l1, l2) in zip(m.layers[2:], first_model.layers[2:])])

    m = load_audio_embedding_model('mel128', 'music', 512)
    # assert isinstance(m.layers[1], kapre.time_frequency.Melspectrogram)
    assert m.layers[1].output_shape == (None, 128, 199, 1)
    assert m.output_shape[1] == 512
    # Check model consistency
    assert isinstance(m.layers[0], type(first_model.layers[0]))
    assert len(m.layers) == len(first_model.layers)
    assert all([isinstance(l1, type(l2))
                for (l1, l2) in zip(m.layers[2:], first_model.layers[2:])])

    m = load_audio_embedding_model('mel128', 'env', 6144)
    # assert isinstance(m.layers[1], kapre.time_frequency.Melspectrogram)
    assert m.layers[1].output_shape == (None, 128, 199, 1)
    assert m.output_shape[1] == 6144
    # Check model consistency
    assert isinstance(m.layers[0], type(first_model.layers[0]))
    assert len(m.layers) == len(first_model.layers)
    assert all([isinstance(l1, type(l2))
                for (l1, l2) in zip(m.layers[2:], first_model.layers[2:])])

    m = load_audio_embedding_model('mel128', 'env', 512)
    # assert isinstance(m.layers[1], kapre.time_frequency.Melspectrogram)
    assert m.layers[1].output_shape == (None, 128, 199, 1)
    assert m.output_shape[1] == 512
    # Check model consistency
    assert isinstance(m.layers[0], type(first_model.layers[0]))
    assert len(m.layers) == len(first_model.layers)
    assert all([isinstance(l1, type(l2))
                for (l1, l2) in zip(m.layers[2:], first_model.layers[2:])])

    m = load_audio_embedding_model('mel256', 'music', 6144)
    # assert isinstance(m.layers[1], kapre.time_frequency.Melspectrogram)
    assert m.layers[1].output_shape == (None, 256, 199, 1)
    assert m.output_shape[1] == 6144
    # Check model consistency
    assert isinstance(m.layers[0], type(first_model.layers[0]))
    assert len(m.layers) == len(first_model.layers)
    assert all([isinstance(l1, type(l2))
                for (l1, l2) in zip(m.layers[2:], first_model.layers[2:])])

    m = load_audio_embedding_model('mel256', 'music', 512)
    # assert isinstance(m.layers[1], kapre.time_frequency.Melspectrogram)
    assert m.layers[1].output_shape == (None, 256, 199, 1)
    assert m.output_shape[1] == 512
    # Check model consistency
    assert isinstance(m.layers[0], type(first_model.layers[0]))
    assert len(m.layers) == len(first_model.layers)
    assert all([isinstance(l1, type(l2))
                for (l1, l2) in zip(m.layers[2:], first_model.layers[2:])])

    m = load_audio_embedding_model('mel256', 'env', 6144)
    # assert isinstance(m.layers[1], kapre.time_frequency.Melspectrogram)
    assert m.layers[1].output_shape == (None, 256, 199, 1)
    assert m.output_shape[1] == 6144
    # Check model consistency
    assert isinstance(m.layers[0], type(first_model.layers[0]))
    assert len(m.layers) == len(first_model.layers)
    assert all([isinstance(l1, type(l2))
                for (l1, l2) in zip(m.layers[2:], first_model.layers[2:])])

    m = load_audio_embedding_model('mel256', 'env', 512)
    # assert isinstance(m.layers[1], kapre.time_frequency.Melspectrogram)
    assert m.layers[1].output_shape == (None, 256, 199, 1)
    assert m.output_shape[1] == 512
    # Check model consistency
    assert isinstance(m.layers[0], type(first_model.layers[0]))
    assert len(m.layers) == len(first_model.layers)
    assert all([isinstance(l1, type(l2))
                for (l1, l2) in zip(m.layers[2:], first_model.layers[2:])])


def test_get_image_embedding_model_path():
    embedding_model_path = get_image_embedding_model_path('linear', 'music')
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'openl3/openl3_image_linear_music.h5'

    embedding_model_path = get_image_embedding_model_path('linear', 'env')
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'openl3/openl3_image_linear_env.h5'

    embedding_model_path = get_image_embedding_model_path('mel128', 'music')
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'openl3/openl3_image_mel128_music.h5'

    embedding_model_path = get_image_embedding_model_path('mel128', 'env')
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'openl3/openl3_image_mel128_env.h5'

    embedding_model_path = get_image_embedding_model_path('mel256', 'music')
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'openl3/openl3_image_mel256_music.h5'

    embedding_model_path = get_image_embedding_model_path('mel256', 'env')
    assert '/'.join(embedding_model_path.split('/')[-2:]) == 'openl3/openl3_image_mel256_env.h5'


def test_load_image_embedding_model():
    m = load_image_embedding_model('linear', 'music', 8192)
    assert m.output_shape[1] == 8192

    first_model = m

    m = load_image_embedding_model('linear', 'music', 512)
    assert m.output_shape[1] == 512
    assert len(m.layers) == len(first_model.layers)
    assert all([isinstance(l1, type(l2))
                for (l1, l2) in zip(m.layers, first_model.layers)])

    m = load_image_embedding_model('linear', 'env', 8192)
    assert m.output_shape[1] == 8192
    assert len(m.layers) == len(first_model.layers)
    assert all([isinstance(l1, type(l2))
                for (l1, l2) in zip(m.layers, first_model.layers)])

    m = load_image_embedding_model('linear', 'env', 512)
    assert m.output_shape[1] == 512
    assert len(m.layers) == len(first_model.layers)
    assert all([isinstance(l1, type(l2))
                for (l1, l2) in zip(m.layers, first_model.layers)])

    m = load_image_embedding_model('mel128', 'music', 8192)
    assert m.output_shape[1] == 8192
    assert len(m.layers) == len(first_model.layers)
    assert all([isinstance(l1, type(l2))
                for (l1, l2) in zip(m.layers, first_model.layers)])

    m = load_image_embedding_model('mel128', 'music', 512)
    assert m.output_shape[1] == 512
    assert len(m.layers) == len(first_model.layers)
    assert all([isinstance(l1, type(l2))
                for (l1, l2) in zip(m.layers, first_model.layers)])

    m = load_image_embedding_model('mel128', 'env', 8192)
    assert m.output_shape[1] == 8192
    assert len(m.layers) == len(first_model.layers)
    assert all([isinstance(l1, type(l2))
                for (l1, l2) in zip(m.layers, first_model.layers)])

    m = load_image_embedding_model('mel128', 'env', 512)
    assert m.output_shape[1] == 512
    assert len(m.layers) == len(first_model.layers)
    assert all([isinstance(l1, type(l2))
                for (l1, l2) in zip(m.layers, first_model.layers)])

    m = load_image_embedding_model('mel256', 'music', 8192)
    assert m.output_shape[1] == 8192
    assert len(m.layers) == len(first_model.layers)
    assert all([isinstance(l1, type(l2))
                for (l1, l2) in zip(m.layers, first_model.layers)])

    m = load_image_embedding_model('mel256', 'music', 512)
    assert m.output_shape[1] == 512
    assert len(m.layers) == len(first_model.layers)
    assert all([isinstance(l1, type(l2))
                for (l1, l2) in zip(m.layers, first_model.layers)])

    m = load_image_embedding_model('mel256', 'env', 8192)
    assert m.output_shape[1] == 8192
    assert len(m.layers) == len(first_model.layers)
    assert all([isinstance(l1, type(l2))
                for (l1, l2) in zip(m.layers, first_model.layers)])

    m = load_image_embedding_model('mel256', 'env', 512)
    assert m.output_shape[1] == 512
    assert len(m.layers) == len(first_model.layers)
    assert all([isinstance(l1, type(l2))
                for (l1, l2) in zip(m.layers, first_model.layers)])
