#!/usr/bin/python3

"""
helper file for POSTagger
"""

__author__ = "Sunil"
__copyright__ = "Copyright (c) 2017 Sunil"
__license__ = "MIT License"
__email__ = "suba5417@colorado.edu"
__version__ = "0.1"

import os
import sys
import math

import POSTagger as pt
from collections import defaultdict

def generate(filename):
    s = set()
    with open(filename, 'r') as file:
        for line in file:
            if not line.strip():
                continue
            word, tag = line.split()
            if word == "." and tag == ".":
                continue
            s.add(tag)
    return s

class DummyDecoder(pt.Decoder):
    """
    Example of dummy decoder.
    """
    def __call__(self, tagger, sentence):
        print(sentence)
        return ['This', 'is', 'dummy', 'tag']

# hmm = pt.HMMTagger(decoder = DummyDecoder())

hmm = pt.HMMTagger()

tt = defaultdict(lambda: defaultdict(float))
tt["<s>"]["VB"] = 0.019
tt["<s>"]["TO"] = 0.0043
tt["<s>"]["NN"] = 0.041
tt["<s>"]["PPSS"] = 0.067

tt["VB"]["VB"] = 0.0038
tt["VB"]["TO"] = 0.035
tt["VB"]["NN"] = 0.047
tt["VB"]["PPSS"] = 0.0070

tt["TO"]["VB"] = 0.83
tt["TO"]["TO"] = 0.0
tt["TO"]["NN"] = 0.00047
tt["TO"]["PPSS"] = 0.0

tt["NN"]["VB"] = 0.0040
tt["NN"]["TO"] = 0.016
tt["NN"]["NN"] = 0.087
tt["NN"]["PPSS"] = 0.0045

tt["PPSS"]["VB"] = 0.23
tt["PPSS"]["TO"] = 0.00079
tt["PPSS"]["NN"] = 0.0012
tt["PPSS"]["PPSS"] = 0.00014

li = defaultdict(lambda: defaultdict(float))
li["VB"]["I"] = 0
li["VB"]["want"] = 0.0093
li["VB"]["to"] = 0
li["VB"]["race"] = 0.00012

li["TO"]["I"] = 0
li["TO"]["want"] = 0
li["TO"]["to"] = 0.99
li["TO"]["race"] = 0

li["NN"]["I"] = 0
li["NN"]["want"] = 0.000054
li["NN"]["to"] = 0
li["NN"]["race"] = 0.00057

li["PPSS"]["I"] = 0.37
li["PPSS"]["want"] = 0
li["PPSS"]["to"] = 0
li["PPSS"]["race"] = 0

tagset = ["VB", "TO", "NN", "PPSS"]

hmm.tagset = set(tagset)
hmm.V = len(tagset)
hmm.tagTransitions = tt
hmm.likelihood = li
sentence = "I want to race"

tags = hmm.Decode(sentence)
print(tags)

# http://www.katrinerk.com/courses/python-worksheets/hidden-markov-models-for-pos-tagging-in-python
# https://github.com/mkvisakan/NLP-Python/tree/master/POS_Tagging
