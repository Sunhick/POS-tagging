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

import os
import sys
import random

# not required for now. as we are not splitting contractions ourself.
# but requried when testing with sentences.
# from nltk.tokenize import *

# to create bigram partial class from ngram
# from functools import partial
# Bigrams = partial(Ngrams, n = 2)

from abc import ABCMeta, abstractmethod

# for calculating log probabilities
from math import log10
from copy import deepcopy
# for getting the max in tuple based on key
from operator import itemgetter
# for getting the max of ProbEntry 
from operator import attrgetter
from collections import defaultdict
from collections import namedtuple

# progress bar
from status import printProgressBar

import _pickle as cPickle

def main(args):
    filename = args[0]
    # train(filename)
    data = POSFile(filename)

    # split data as 80:20 for train and test 
    train, test = data.RandomSplit(80)
    tagger = HMMTagger(k = .0000001, decoder = Viterbi())
    tagger.Train(train)

    formatter = lambda word, tag: "{0}\t{1}\n".format(word, tag)
    endOfSentence = ".\t.{newline}{newline}".format(newline=os.linesep)

    current = 0
    total = len(test)

    expectedTags = []
    predictedTags = []

    # decode = tagger.Decode
    print("\nDecoding tag sequence for test data:\n")
    with open("berp-key.txt", "w") as goldFile,         \
         open("berp-out.txt", "w") as outFile:
        for line in test:
            sentence = line.Sentence
            words = sentence.split()
            if not words:
                continue
            tagSequence = tagger.Decode(sentence)

            predictedTags.extend(tagSequence)
            # assert len(tagSequence) == len(words),   \
            #         "total tag sequence and len of words in sentence should be equal"

            for wt in line:
                if not wt.IsFirstWord() and not wt.IsLastWord():
                    w, t = wt
                    goldFile.write(formatter(w, t))
                    expectedTags.append(t)
            goldFile.write(endOfSentence)

            for w, t in zip(words, tagSequence):
                outFile.write(formatter(w, t))
            outFile.write(endOfSentence)

            current += 1
            printProgressBar(current, total, prefix="Progress:", suffix="completed", length=50)

    # TODO: get top-N mistagged words.

    # from confusion_matrix import plotcnf
    # plotcnf(expectedTags, predictedTags)
    # with open('exp.pkl', 'wb') as out:
    #     cPickle.dump(expectedTags, out, protocol=2)

    # with open('pred.pkl', 'wb') as out:
    #     cPickle.dump(predictedTags, out, protocol=2)

if __name__ == "__main__":
    main(sys.argv[1:])