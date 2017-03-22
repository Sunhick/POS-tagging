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

class WordTag(namedtuple('WordTag', ['word', 'tag'], verbose=False)):
    """
    Represents the word tag pair. Inherits from namedtuple so that i can 
    unpack the class in word, tag pair easily in the iterations.
    """

    # can't do __init__ in case of namedtuple
    # def __init__(self, word, tag):
    #     # TODO: shoud i ignore the case of word?
    #     self.word = word.lower()
    #     self.tag = tag

    def __new__(cls, word, tag):
        word = word.lower()
        self = super(WordTag, cls).__new__(cls, word, tag)
        return self

    def IsLastWord(self):
        return self.word == Constants.kSENTENCE_END

    def IsFirstWord(self):
        return self.word == Constants.kSENTENCE_BEGIN