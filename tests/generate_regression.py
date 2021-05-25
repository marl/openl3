import os
import glob
import openl3.cli


fglob = lambda *f: glob.glob(os.path.join(*f))


def just_keep_swimming(func):
    def inner(*a, **kw):
        try:
            return func(*a, **kw)
        except Exception:
            print('failure while running {}({}, {}) (u get the idea)'.format(func.__name__, a, kw))
            import traceback
            traceback.print_exc()
            print('continuing on...')
            print()
    return inner

def sfxs(*xs):
    return '_'.join(map(str, (x for x in xs if x)))

@just_keep_swimming
def run_default(modality, fname, output_dir, **kw):
    openl3.cli.run(modality, fname, output_dir=output_dir, overwrite=True, verbose=True, **kw)

@just_keep_swimming
def run_linear_audio(fname, output_dir, suffix='', **kw):
    openl3.cli.run(
        'audio', fname, output_dir=output_dir, suffix=sfxs(suffix, 'linear'), input_repr='linear',
        content_type='env', audio_embedding_size=512, audio_center=False, audio_hop_size=0.5,
        overwrite=True, verbose=False, **kw)

@just_keep_swimming
def run_linear_video(fname, output_dir, suffix='', **kw):
    openl3.cli.run(
        'video', fname, output_dir=output_dir, suffix=sfxs(suffix, 'linear'), input_repr='linear',
        content_type='env', audio_embedding_size=512, image_embedding_size=512, audio_center=False, audio_hop_size=0.5,
        overwrite=True, verbose=False, **kw)

@just_keep_swimming
def run_linear_img(fname, output_dir, suffix='', **kw):
    openl3.cli.run(
        'image', fname, output_dir=output_dir, suffix=sfxs(suffix, 'linear'), input_repr='linear',
        content_type='env', image_embedding_size=512, overwrite=True, verbose=True, **kw)


def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    out_dir = os.path.join(data_dir, 'regression')

    frontends = ['kapre', 'librosa']#

    for front in frontends:
        mod = 'audio'
        for f in fglob(data_dir, mod, '*'):
            run_default(mod, f, out_dir, audio_frontend=front, suffix=front)#
            run_linear_audio(f, out_dir, audio_frontend=front, suffix=front)

        mod = 'video'
        for f in fglob(data_dir, mod, '*'):
            run_default(mod, f, out_dir, audio_frontend=front, suffix=front)#, suffix=front
            run_linear_video(f, out_dir, audio_frontend=front, suffix=front)

    # mod = 'image'
    # for f in fglob(data_dir, mod, '*'):
    #     run_linear_img(f, out_dir)

if __name__ == '__main__':
    import fire
    fire.Fire(main)