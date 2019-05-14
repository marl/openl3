from __future__ import print_function
import os
import sys
import sklearn.decomposition
from openl3 import process_file
from openl3.models import load_embedding_model
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


def run(inputs, output_dir=None, suffix=None, input_repr="mel256", content_type="music",
        embedding_size=6144, center=True, hop_size=0.1, verbose=False):
    """
    Computes and saves L3 embedding for given inputs.

    Parameters
    ----------
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
    embedding_size : 6144 or 512
        Embedding dimensionality.
    center : boolean
        If True, pads beginning of signal so timestamps correspond
        to center of window.
    hop_size : float
        Hop size in seconds.
    quiet : boolean
        If True, suppress all non-error output to stdout

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
        print('openl3: No WAV files found in {}. Aborting.'.format(str(inputs)))
        sys.exit(-1)

    # Load model
    model = load_embedding_model(input_repr, content_type, embedding_size)

    # Process all files in the arguments
    for filepath in file_list:
        if verbose:
            print('openl3: Processing: {}'.format(filepath))
        process_file(filepath,
                     output_dir=output_dir,
                     suffix=suffix,
                     model=model,
                     center=center,
                     hop_size=hop_size,
                     verbose=verbose)
    if verbose:
        print('openl3: Done!')


def parse_args(args):
    parser = ArgumentParser(sys.argv[0], description=main.__doc__,
                            formatter_class=RawDescriptionHelpFormatter)

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
                             'representation for the embedding model.')

    parser.add_argument('--content-type', '-c', default='music',
                        choices=['music', 'env'],
                        help='Content type used to train embedding model.')

    parser.add_argument('--embedding-size', '-s', type=int, default=6144,
                        help='Embedding dimensionality.')

    parser.add_argument('--no-centering', '-n', action='store_true', default=False,
                        help='Do not pad signal; timestamps will correspond to '
                             'the beginning of each analysis window.')

    parser.add_argument('--hop-size', '-t', type=positive_float, default=0.1,
                        help='Hop size in seconds for processing audio files.')

    parser.add_argument('--quiet', '-q', action='store_true', default=False,
                        help='Suppress all non-error messages to stdout.')

    return parser.parse_args(args)


def main():
    """
    Extracts audio embeddings from models based on the Look, Listen, and Learn models (Arandjelovic and Zisserman 2017).
    """
    args = parse_args(sys.argv[1:])

    run(args.inputs,
        output_dir=args.output_dir,
        suffix=args.suffix,
        input_repr=args.input_repr,
        content_type=args.content_type,
        embedding_size=args.embedding_size,
        center=not args.no_centering,
        hop_size=args.hop_size,
        verbose=not args.quiet)
