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

class Line(object):
    """
    Represents the sentence as collection of words and thier 
    corresponding tag sequences.
    """

    def __init__(self):
        self.words = []
        self.__index = 0

    def AddWordTag(self, word, tag):
        wordTag = WordTag(word, tag)
        self.words.append(wordTag)

    @property
    def Sentence(self):
        """
        Get the sentence representation of line as a string
        """
        words = [wt.word if (not wt.IsFirstWord() and not wt.IsLastWord()) else ""
                     for wt in self.words]
        return " ".join(words).strip()

    def __iter__(self):
        # called once before iteration. reset index pointer
        self.__index = 0
        return self

    def __next__(self):
        """
        Iterate through the list of words
        """
        if self.__index >= len(self.words):
            raise StopIteration
        else:
            self.__index += 1
            return self.words[self.__index-1]

    def __eq__(self, other):
        """
        compare if two given sentences are same. 
        Returns True if two sentences have same (word, tag) pair in same order.
        """
        for wt in zip(self.words, other.words):
            w1, t1 = wt[0]
            w2, t2 = wt[0]
            if (w1 != w2 and t1 != t2):
                return False
        return True