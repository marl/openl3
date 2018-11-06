import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter, ArgumentTypeError


def positive_float(value):
    """An argparse type method for accepting only positive floats"""
    fvalue = float(value)
    if fvalue <= 0:
        raise ArgumentTypeError('expected a positive float')
    return fvalue


def main():
    """
    """
    parser = ArgumentParser(sys.argv[0], description=main.__doc__,
                            formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument('input', nargs='+',
                        help='Path or paths to files to process, or path to '
                             'a directory of files to process.')

    parser.add_argument('--output', '-o', default=None,
                        help='Directory to save the ouptut file(s); '
                             'if not given, the output will be '
                             'saved to the same directory as the input WAV '
                             'file(s).')

    parser.add_argument('--input-repr', '-i', default='mel256',
                        choices=['linear', 'mel128', 'mel256'],
                        help='String specifying the time-frequency input '
                             'representation for the embedding model.')

    parser.add_argument('--content-type', '-c', default='music',
                        choices=['music', 'env'],
                        help='Content type used to train embedding model.')

    parser.add_argument('--embedding-size', '-s', default=6144,
                        help='Embedding dimensionality.')

    parser.add_argument('--no-centering', '-n', action='store_true',
                        help='Do not pad signal; timestamps will correspond to '
                             'the beginning of each analysis window.')

    parser.add_argument('--hop-size', '-t', type=positive_float,
                        help=' Hop size in seconds for processing audio files.')

    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress all non-error messages to stdout.')

    raise NotImplementedError()
