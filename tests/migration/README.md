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