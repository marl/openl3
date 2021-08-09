import openl3.core
from openl3.models import (
    load_audio_embedding_model, get_audio_embedding_model_path,
    load_image_embedding_model, get_image_embedding_model_path,
    load_audio_embedding_model_from_path, load_image_embedding_model_from_path)
from openl3.openl3_exceptions import OpenL3Error
import pytest


AUDIO_INPUT_REPR_SIZES = {
    'linear': (None, 257, 197, 1),
    'mel128': (None, 128, 199, 1),
    'mel256': (None, 256, 199, 1),
}
CONTENT_TYPES = ['env', 'music']
VALID_AUDIO_EMBEDDING_SIZES = (6144, 512)

IMAGE_INPUT_REPR_SIZES = {
    'linear': (None, 224, 224, 3),
    'mel128': (None, 224, 224, 3),
    'mel256': (None, 224, 224, 3),
}
VALID_IMAGE_EMBEDDING_SIZES = (8192, 512)




@pytest.fixture(scope="module")
def ref_audio_model():
    input_repr, content_type, embedding_size = 'linear', 'music', 6144
    m = load_audio_embedding_model(input_repr, content_type, embedding_size)
    # assert isinstance(m.layers[1], kapre.time_frequency.Spectrogram)
    assert m.layers[1].output_shape == AUDIO_INPUT_REPR_SIZES[input_repr]
    assert m.output_shape[1] == embedding_size
    return m


@pytest.fixture(scope="module")
def ref_image_model():
    input_repr, content_type, embedding_size = 'linear', 'music', 8192
    m = load_image_embedding_model(input_repr, content_type, embedding_size)
    assert m.output_shape[1] == embedding_size
    return m


def _compare_models(m, ref_model, input_size, embedding_size, **kw):
    assert m.layers[1].output_shape == input_size
    assert m.output_shape[1] == embedding_size
    if kw.get('skip_layers') is not True:
        _compare_layers(m.layers, ref_model.layers, **kw)


def _compare_layers(layersA, layersB, skip_layers=None, compare_shapes=False):
    skip_layers = set(skip_layers or ())
    assert len(layersA) == len(layersB)
    for i, (la, lb) in enumerate(zip(layersA, layersB)):
        if i in skip_layers:
            continue
        assert isinstance(la, type(lb))
        if compare_shapes:
            assert la.input_shape == lb.input_shape
            assert la.output_shape == lb.output_shape


@pytest.mark.parametrize('input_repr', list(AUDIO_INPUT_REPR_SIZES))
@pytest.mark.parametrize('content_type', CONTENT_TYPES)
def test_get_audio_embedding_model_path(input_repr, content_type):
    embedding_model_path = get_audio_embedding_model_path(input_repr, content_type)
    assert (
        '/'.join(embedding_model_path.split('/')[-2:]) == 
        'openl3/openl3_audio_{}_{}.h5'.format(input_repr, content_type))


@pytest.mark.parametrize('input_repr', list(AUDIO_INPUT_REPR_SIZES))
@pytest.mark.parametrize('content_type', CONTENT_TYPES)
@pytest.mark.parametrize('embedding_size', VALID_AUDIO_EMBEDDING_SIZES)
def test_get_audio_embedding_model(input_repr, content_type, embedding_size, ref_audio_model):
    m = load_audio_embedding_model(input_repr, content_type, embedding_size)
    _compare_models(m, ref_audio_model, AUDIO_INPUT_REPR_SIZES[input_repr], embedding_size, skip_layers=[1])


@pytest.mark.parametrize('input_repr', list(AUDIO_INPUT_REPR_SIZES))
@pytest.mark.parametrize('content_type', CONTENT_TYPES)
@pytest.mark.parametrize('embedding_size', VALID_AUDIO_EMBEDDING_SIZES)
def test_get_audio_embedding_model_by_path(input_repr, content_type, embedding_size, ref_audio_model):
    m = load_audio_embedding_model_from_path(get_audio_embedding_model_path(input_repr, content_type), input_repr, embedding_size)
    _compare_models(m, ref_audio_model, AUDIO_INPUT_REPR_SIZES[input_repr], embedding_size, skip_layers=[1])


@pytest.mark.parametrize('input_repr', list(AUDIO_INPUT_REPR_SIZES))
def test_frontend(input_repr):
    # check spectrogram input size
    m = load_audio_embedding_model(input_repr, 'env', 512, frontend='librosa')
    assert m.input_shape == AUDIO_INPUT_REPR_SIZES[input_repr]
    m2 = load_audio_embedding_model(input_repr, 'env', 512, frontend='kapre')
    assert m2.input_shape == (None, 1, openl3.core.TARGET_SR)
    # compare all layers to model with frontend
    _compare_layers(m.layers[1:], m2.layers[2:], compare_shapes=True)

    with pytest.raises(OpenL3Error):
        load_audio_embedding_model(input_repr, 'env', 512, frontend='not-a-thing')


def test_validate_audio_frontend():
    input_repr = 'mel128'

    # test kapre
    mk = load_audio_embedding_model(input_repr, 'env', 512, frontend='kapre')
    assert len(mk.input_shape) == 3
    # assert openl3.models._validate_audio_frontend('infer', input_repr, mk) == ('kapre', input_repr)
    assert openl3.models._validate_audio_frontend('kapre', input_repr, mk) == ('kapre', input_repr)

    # test librosa validate
    ml = load_audio_embedding_model(input_repr, 'env', 512, frontend='librosa')
    assert len(ml.input_shape) == 4
    # assert openl3.models._validate_audio_frontend('infer', input_repr, ml) == ('librosa', input_repr)
    assert openl3.models._validate_audio_frontend('librosa', input_repr, ml) == ('librosa', input_repr)

    # test frontend + no input_repr
    assert openl3.models._validate_audio_frontend('kapre', None, mk) == ('kapre', 'mel256')
    with pytest.raises(OpenL3Error):
        openl3.models._validate_audio_frontend('librosa', None, ml)
    
    # test mismatched frontend/model
    with pytest.raises(OpenL3Error):
        openl3.models._validate_audio_frontend('librosa', None, mk)
    with pytest.raises(OpenL3Error):
        openl3.models._validate_audio_frontend('kapre', None, ml)


@pytest.mark.parametrize('input_repr', list(AUDIO_INPUT_REPR_SIZES))
@pytest.mark.parametrize('content_type', CONTENT_TYPES)
def test_get_image_embedding_model_path(input_repr, content_type):
    embedding_model_path = get_image_embedding_model_path(input_repr, content_type)
    assert (
        '/'.join(embedding_model_path.split('/')[-2:]) == 
        'openl3/openl3_image_{}_{}.h5'.format(input_repr, content_type))


@pytest.mark.parametrize('input_repr', list(IMAGE_INPUT_REPR_SIZES))
@pytest.mark.parametrize('content_type', CONTENT_TYPES)
@pytest.mark.parametrize('embedding_size', VALID_IMAGE_EMBEDDING_SIZES)
def test_get_image_embedding_model(input_repr, content_type, embedding_size, ref_image_model):
    m = load_image_embedding_model(input_repr, content_type, embedding_size)
    _compare_models(m, ref_image_model, IMAGE_INPUT_REPR_SIZES[input_repr], embedding_size)


@pytest.mark.parametrize('input_repr', list(IMAGE_INPUT_REPR_SIZES))
@pytest.mark.parametrize('content_type', CONTENT_TYPES)
@pytest.mark.parametrize('embedding_size', VALID_IMAGE_EMBEDDING_SIZES)
def test_get_image_embedding_model_from_path(input_repr, content_type, embedding_size, ref_image_model):
    m = load_image_embedding_model_from_path(
        get_image_embedding_model_path(input_repr, content_type), embedding_size)
    _compare_models(m, ref_image_model, IMAGE_INPUT_REPR_SIZES[input_repr], embedding_size)
