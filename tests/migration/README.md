# Tensorflow 2 Migration

## Steps

### To get the current version runnable:
install the dev version:
```bash
git clone https://github.com/beasteers/openl3.git openl3-tf2
cd openl3-tf2
git checkout tf2
pip install -e .
```
fix the weights files:
```bash
# just as a safety check - make sure it's just showing the spectrogram/melspectrogram weights:
python tests/migration/remove_layers.py
# actually patch the weights:
python tests/migration/remove_layers.py --doit
```

### Comparing Frontends

```bash
import openl3

def get_model_frontend(input_repr='linear', content_type="music", embedding_size=6144):
    model = openl3.models.load_audio_embedding_model(input_repr=input_repr, content_type=content_type, embedding_size=embedding_size)
    front = type(model)(model.inputs, model.layers[1](model.input))
    return front

# compute each frontend output
spec = get_model_frontend()
X_kapre = specmodel.predict(openl3.preprocess(audio, sr, hop_size=1))
X_librosa = openl3.preprocess(audio, sr, hop_size=1, input_repr=input_repr)
```

#### Changing Frontends
```bash
import openl3  # using dev openl3

# load audio
TEST_AUDIO_DIR = os.path.abspath(os.path.join('../data', 'audio'))
CHIRP_44K_PATH = os.path.join(TEST_AUDIO_DIR, 'chirp_44k.wav')
audio, sr = sf.read(CHIRP_44K_PATH)

# get embedding using kapre frontend 
# (if you want the legacy kapre version - import openl3 from pypi instead)
Z, ts = openl3.get_audio_embedding(audio, sr)
Z, ts = openl3.get_audio_embedding(audio, sr, frontend='kapre')  # equivalent

# get embedding using librosa frontend (matches kapre 0.3.5 implementation)
Z, ts = openl3.get_audio_embedding(audio, sr, frontend='librosa')

# get embedding using librosa frontend (matches kapre 0.1.4 implementation)
import openl3.core
openl3.core.use_librosa_v2(False)
Z, ts = openl3.get_audio_embedding(audio, sr, frontend='librosa')
openl3.core.use_librosa_v2(True)
```

#### Changing Frontends with a pre-built model.
```bash
import openl3  # using dev openl3

# load audio
TEST_AUDIO_DIR = os.path.abspath(os.path.join('../data', 'audio'))
CHIRP_44K_PATH = os.path.join(TEST_AUDIO_DIR, 'chirp_44k.wav')
audio, sr = sf.read(CHIRP_44K_PATH)

input_repr, content_type, embedding_size = 'mel128', 'music', 6144

# get embedding using kapre frontend 
# (if you want the legacy kapre version - import openl3 from pypi instead)
model = openl3.models.load_audio_embedding_model(
    input_repr, content_type, embedding_size)
Z, ts = openl3.get_audio_embedding(audio, sr, model=model)  # kapre

# include_frontend=False removes the kapre layer
# now you must make sure to include input_repr in your get_audio_embedding call

# get embedding using librosa frontend (matches kapre 0.3.5 implementation)
model = openl3.models.load_audio_embedding_model(
    input_repr, content_type, embedding_size, include_frontend=False)
Z, ts = openl3.get_audio_embedding(audio, sr, model=model, input_repr=input_repr)  # librosa

# get embedding using librosa frontend (matches kapre 0.1.4 implementation)
import openl3.core
openl3.core.use_librosa_v2(False)

model = openl3.models.load_audio_embedding_model(
    input_repr, content_type, embedding_size, include_frontend=False)
Z, ts = openl3.get_audio_embedding(audio, sr, model=model, input_repr=input_repr)  # librosa
openl3.core.use_librosa_v2(True)

```

## TF 1 - TF 2 Comparisons

### To compare the model layer outputs:
NOTE: You need two python environments: one with the live version, and one with the dev version of openl3.

Run this command in each of your environments that you want to compare:
```bash
python tests/migration/check_layer_by_layer.py compute
```

You can use `--input-repr mel256 --content-type music --embedding-size=6144` flags to compute for other types.

Defaults are: `input_repr="linear", content_type="music", embedding_size=6144, hop_size=1`

#### To Plot:
The arguments used below are substrings that are used to select colocated pickle files. So you just have to give the minimum needed to identify them uniquely. You can join multiple substrings using a `+` sign (e.g. `3.+128+mus+6144`).

So obviously it's easier when you have fewer combos computed because that's less you need to give to distinguish by.

```bash
# to compare the linear versions:
python tests/migration/check_layer_by_layer.py \
    show 3.+lin 4.+lin --limit 1
# to compare the mel256 versions:
python tests/migration/check_layer_by_layer.py \
    show 0.3.1-mel2 0.4.0-mel2 --limit 1
# to compare the mel128 versions:
python tests/migration/check_layer_by_layer.py \
    show 0.3.1-mel1 0.4.0-mel1 --limit 1
# to compare  versions:
python tests/migration/check_layer_by_layer.py \
    show 0.3.1-mel1 0.4.0-mel1 --limit 1
```

Those commands above are just showing the first layer. To show the first 3, you can remove `--limit`, specify `--limit 5`, or say `--nolimit` for them all.

See the docstring for info about other arguments (defaults: `fileindex=0, mapindex=0, limit=3, offset=0, clip=None`)

#### To compare them all numerically
This prints out stats about the differences to the terminal.

Same rules about args apply.
```bash
# to compare the linear versions:
python tests/migration/check_layer_by_layer.py \
    compare 3.+lin 4.+lin
```