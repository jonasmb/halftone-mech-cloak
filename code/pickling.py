"""Pickle helper functions."""

import lzma
import os
import pickle

samples_extension = "_samples.pickle.xz"


def dump_samples(filename_json, samples, prealias=""):
    """
    Dump the samples into a compressed pickled file.

    Args:
        filename_json: filename of the output json file.
        samples: list of samples.
    """
    filename_samples = os.path.splitext(filename_json)[0] + prealias + samples_extension
    with lzma.open(filename_samples, "wb") as lzmafile:
        pickle.dump(samples, lzmafile)


def load_samples(filename_json, prealias=""):
    """
    Load the samples from a compressed pickled file.

    Args:
        filename_json: filename of the input json file.

    Returns:
        List of samples.
    """
    filename_samples = os.path.splitext(filename_json)[0] + prealias + samples_extension
    with lzma.open(filename_samples, "rb") as lzmafile:
        return pickle.load(lzmafile)
