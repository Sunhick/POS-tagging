#!/usr/bin/python3

"""
Part of speech tagging in python using Hidden Markov Model.

POS tagging using Hidden Markov model (Viterbi algorithm)
"""

__author__ = "Sunil"
__copyright__ = "Copyright (c) 2017 Sunil"
__license__ = "MIT License"
__email__ = "suba5417@colorado.edu"
__version__ = "0.1"

class Bigrams(Ngrams):
    """
    Represents the bigrams tokens. This is a
    vanilla class that inherits from Ngrams.
    """
    def __init__(self, words):
        super(Bigrams, self).__init__(words, n = 2)