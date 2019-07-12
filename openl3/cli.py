from __future__ import print_function
import os
import sys
import sklearn.decomposition
from openl3 import process_audio_file, process_image_file, process_video_file
from openl3.models import load_audio_embedding_model, load_image_embedding_model
from openl3.openl3_exceptions import OpenL3Error
from argparse import ArgumentParser, RawDescriptionHelpFormatter, ArgumentTypeError
from collections import Iterable
from six import string_types


def positive_float(value):
    """An argparse type method for accepting only positive floats"""
    try:
        fvalue = float(value)
    except (ValueError, TypeError) as e:
        raise ArgumentTypeError('Expected a positive float, error message: '
                                '{}'.format(e))
    if fvalue <= 0:
        raise ArgumentTypeError('Expected a positive float')
    return fvalue


def get_file_list(input_list):
    """Get list of files from the list of inputs"""
    if not isinstance(input_list, Iterable) or isinstance(input_list, string_types):
        raise ArgumentTypeError('input_list must be iterable (and not string)')
    file_list = []
    for item in input_list:
        if os.path.isfile(item):
            file_list.append(os.path.abspath(item))
        elif os.path.isdir(item):
            for fname in os.listdir(item):
                path = os.path.join(item, fname)
                if os.path.isfile(path):
                    file_list.append(path)
        else:
            raise OpenL3Error('Could not find {}'.format(item))

    return file_list


def run(modality, inputs, output_dir=None, suffix=None,
        input_repr="mel256", content_type="music",
        audio_embedding_size=6144, audio_center=True, audio_hop_size=0.1,
        image_embedding_size=8192, verbose=False):
    """
    Computes and saves L3 embedding for given inputs.

    Parameters
    ----------
    modality : str
        String to specify the modalities to be processed: audio, image, or video
    inputs : list of str, or str
        File/directory path or list of file/directory paths to be processed
    output_dir : str or None
        Path to directory for saving output files. If None, output files will
        be saved to the directory containing the input file.
    suffix : str or None
        String to be appended to the output filename, i.e. <base filename>_<suffix>.npy.
        If None, then no suffix will be added, i.e. <base filename>.npy.
    input_repr : "linear", "mel128", or "mel256"
        Spectrogram representation used for model.
    content_type : "music" or "env"
        Type of content used to train embedding.
    audio_embedding_size : 6144 or 512
        Audio embedding dimensionality.
    audio_center : boolean
        If True, pads beginning of signal so timestamps correspond
        to center of window.
    audio_hop_size : float
        Hop size in seconds.
    image_embedding_size : 8192 or 512
        Image embedding dimensionality.
    verbose : boolean
        If True, print verbose messages.

    Returns
    -------
    """

    if isinstance(inputs, string_types):
        file_list = [inputs]
    elif isinstance(inputs, Iterable):
        file_list = get_file_list(inputs)
    else:
        raise OpenL3Error('Invalid input: {}'.format(str(inputs)))

    if len(file_list) == 0:
        print('openl3: No files found in {}. Aborting.'.format(str(inputs)))
        sys.exit(-1)

    # Load model
    if modality == 'audio':
        model = load_audio_embedding_model(input_repr, content_type,
                                           audio_embedding_size)

        # Process all files in the arguments
        process_audio_file(file_list,
                           output_dir=output_dir,
                           suffix=suffix,
                           model=model,
                           center=audio_center,
                           hop_size=audio_hop_size,
                           verbose=verbose)
    elif modality == 'image':
        model = load_image_embedding_model(input_repr, content_type,
                                           image_embedding_size)

        # Process all files in the arguments
        process_image_file(file_list,
                           output_dir=output_dir,
                           suffix=suffix,
                           model=model,
                           verbose=verbose)
    elif modality == 'video':
        audio_model = load_audio_embedding_model(input_repr, content_type,
                                                 audio_embedding_size)
        image_model = load_image_embedding_model(input_repr, content_type,
                                                 image_embedding_size)

        # Process all files in the arguments
        process_video_file(file_list,
                           output_dir=output_dir,
                           suffix=suffix,
                           audio_model=audio_model,
                           image_model=image_model,
                           audio_embedding_size=audio_embedding_size,
                           audio_center=audio_center,
                           audio_hop_size=audio_hop_size,
                           image_embedding_size=image_embedding_size,
                           verbose=verbose)

    if verbose:
        print('openl3: Done!')


def parse_args(args):
    parser = ArgumentParser(sys.argv[0], description=main.__doc__,
                            formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument('modality',
                        choices=['audio', 'image', 'video'],
                        help='String to specify the modality to the '
                             'embedding model, audio, image, or video.')

    parser.add_argument('inputs', nargs='+',
                        help='Path or paths to files to process, or path to '
                             'a directory of files to process.')

    parser.add_argument('--output-dir', '-o', default=None,
                        help='Directory to save the ouptut file(s); '
                             'if not given, the output will be '
                             'saved to the same directory as the input WAV '
                             'file(s).')

    parser.add_argument('--suffix', '-x', default=None,
                        help='String to append to the output filenames.'
                             'If not provided, no suffix is added.')

    parser.add_argument('--input-repr', '-i', default='mel256',
                        choices=['linear', 'mel128', 'mel256'],
                        help='String specifying the time-frequency input '
                             'representation for the audio embedding model.')

    parser.add_argument('--content-type', '-c', default='music',
                        choices=['music', 'env'],
                        help='Content type used to train embedding model.')

    parser.add_argument('--audio-embedding-size', '-as', type=int, default=6144,
                        choices=[6144, 512],
                        help='Audio embedding dimensionality.')

    parser.add_argument('--no-audio-centering', '-n', action='store_true',
                        default=False,
                        help='Used for audio embeddings. Do not pad signal; '
                             'timestamps will correspond to '
                             'the beginning of each analysis window.')

    parser.add_argument('--audio-hop-size', '-t', type=positive_float, default=0.1,
                        help='Used for audio embeddings. '
                             'Hop size in seconds for processing audio files.')

    parser.add_argument('--image-embedding-size', '-is', type=int, default=8192,
                        choices=[8192, 512],
                        help='Image embedding dimensionality.')

    parser.add_argument('--quiet', '-q', action='store_true', default=False,
                        help='Suppress all non-error messages to stdout.')

    return parser.parse_args(args)


def main():
    """
    Extracts audio embeddings from models based on the Look, Listen, and Learn models (Arandjelovic and Zisserman 2017).
    """
    args = parse_args(sys.argv[1:])

    run(args.modality,
        args.inputs,
        output_dir=args.output_dir,
        suffix=args.suffix,
        input_repr=args.input_repr,
        content_type=args.content_type,
        audio_embedding_size=args.audio_embedding_size,
        audio_center=not args.no_audio_centering,
        audio_hop_size=args.audio_hop_size,
        image_embedding_size=args.image_embedding_size,
        verbose=not args.quiet)
