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

class ProbEntry(object):
    """
    Represents path probability entry in viterbi matrix 
    """

    # store log probability for easier calculations, and also
    # we don't lose the floating point precision.

    def __init__(self, probability=0.0, tag=None, backpointer=None):
        self.probability = probability
        self.backpointer = backpointer
        self.tag = tag
        self.word = None

    def __str__(self):
        """
        string representation of ProbEntry.
        """
        backptr = id(self.backpointer) if self.backpointer else None
        return "Prob={0} id={2} tag={3} word={4} BackPtr={1}".     \
            format(self.probability, backptr, id(self), self.tag, self.word)