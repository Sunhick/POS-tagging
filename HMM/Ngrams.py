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

class Ngrams(object):
    """
    Generates the ngrams from the list of words.
    """

    def __init__(self, words, n):
        self.Ngrams = list(zip(*[words[i:] for i in range(n)]))
        self.__index = 0

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self):
        if self.__index >= len(self.Ngrams):
            raise StopIteration
        else:
            self.__index += 1
            return self.Ngrams[self.__index-1]