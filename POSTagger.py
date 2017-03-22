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

class PennTreebank(object):
    """
    Dictionary of all tags in the Penn tree bank. 
    
    Can be used to look up penn tree bank codeword to human 
    understandable Parts of speech
    """
    tagset = defaultdict (
            lambda: '#unknown#',
            {
                "CC"    : "Coordinating conjunction",
                "CD"    : "Cardinal number",
                "DT"    : "Determiner",
                "EX"    : "Existential there",
                "FW"    : "Foreign word",
                "IN"    : "Preposition or subordinating conjunction",
                "JJ"    : "Adjective",
                "JJR"   : "Adjective, comparative",
                "JJS"   : "Adjective, superlative",
                "LS"    : "List item marker",
                "MD"    : "Modal",
                "NN"    : "Noun, singular or mass",
                "NNS"   : "Noun, plural",
                "NNP"   : "Proper noun, singular",
                "NNPS"  : "Proper noun, plural",
                "PDT"   : "Predeterminer",
                "POS"   : "Possessive ending",
                "PRP"   : "Personal pronoun",
                "PRP$"  : "Possessive pronoun",
                "RB"    : "Adverb",
                "RBR"   : "Adverb, comparative",
                "RBS"   : "Adverb, superlative",
                "RP"    : "Particle",
                "SYM"   : "Symbol",
                "TO"    : "to",
                "UH"    : "Interjection",
                "VB"    : "Verb, base form",
                "VBD"   : "Verb, past tense",
                "VBG"   : "Verb, gerund or present participle",
                "VBN"   : "Verb, past participle",
                "VBP"   : "Verb, non-3rd person singular present",
                "VBZ"   : "Verb, 3rd person singular present",
                "WDT"   : "Wh-determiner",
                "WP"    : "Wh-pronoun",
                "WP$"   : "Possessive wh-pronoun",
                "WRB"   : "Wh-adverb"
            }
        )

    @classmethod
    def lookup(cls, codedTag):
        """
        look up coded tag and return human understanable POS.
        No exception hanlding required because of defaultdict
        """
        return cls.tagset[codedTag]

class POSError(Exception):
    """
    Defines the POS tagger application error
    """
    pass

class Constants(object):
    """
    POS tagger constants
    """
    kSENTENCE_BEGIN = "<s>"
    kSENTENCE_END = "</s>"

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

class POSFile(object):
    """
    Abstraction over words-tags, lines as a POSfile.
    """

    def __init__(self, filename):
        # os.path.exists is a false positive if we are looking for file.
        # os.path.exists only checks if there's an inode entry in dir and doesn't
        # check the type of file in that directory.
        if not os.path.isfile(filename):
            raise POSError("{0} is a invalid file.".format(filename))

        self.lines = []
        self.__read(filename)

    def __read(self, filename):
        sentence = Line()
        sentence.AddWordTag(Constants.kSENTENCE_BEGIN, Constants.kSENTENCE_BEGIN)
        with open(filename, 'r') as file:
            for line in file:
                if not line.strip():
                    # end of sentence
                    self.lines.append(sentence)

                    # create a new line holder
                    sentence = Line()
                    # add the word begin marker
                    sentence.AddWordTag(Constants.kSENTENCE_BEGIN, Constants.kSENTENCE_BEGIN)
                    continue

                word, tag = line.split()

                # TODO: Should i ignore the periods?
                # Marks the last word in the sentence
                if word == "." and tag == ".":
                    # add the word end marker
                    sentence.AddWordTag(Constants.kSENTENCE_END, Constants.kSENTENCE_END)
                else:
                    sentence.AddWordTag(word, tag)

    @property
    def Lines(self):
        return self.lines

    def Split(self, train_percent):
        """
        Split the lines into train and test set.
        """
        size = len(self.lines)
        train_len = int(size*train_percent/100)
        test_len = size - train_len
        return self.lines[:train_len], self.lines[-test_len:]

    def RandomSplit(self, train_percent):
        """
        Shuffle & split the lines into train and test set.
        """
        # lines = deepcopy(self.lines)
        random.shuffle(self.lines)
        size = len(self.lines)
        train_len = int(size*train_percent/100)
        test_len = size - train_len
        return self.lines[:train_len], self.lines[-test_len:]

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

class Bigrams(Ngrams):
    """
    Represents the bigrams tokens. This is a
    vanilla class that inherits from Ngrams.
    """
    def __init__(self, words):
        super(Bigrams, self).__init__(words, n = 2)

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

class Decoder(object):
    """
    Decoder interface 
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, tagger, sentence):
        raise NotImplementedError("Should implement __call__(...) method")

class Viterbi(Decoder):
    """
    Stateless viterbi algorithm to decode the sequence using bigram model.
    """

    def __init__(self):
        self.start = "start"
        self.end = "end"

    def __backTrack(self, viterbi, T):
        """
        Return the tag sequence by following back pointers to states
        back in time from viterbi.backpointers
        """
        tagSequence = []
        pointer = viterbi[T+1][self.end][self.end].backpointer
        
        # traverse the back pointers
        while (pointer):
            # print("TAG=%s WORD=%s" % (pointer.tag, pointer.word))
            tagSequence.append(pointer.tag)
            pointer = pointer.backpointer

        # reverse the tag sequence as they are 
        # traced from back to front
        tagSequence.reverse()
        return tagSequence

    def __call__(self, tagger, sentence):
        """
        callable method on instance. Takes a hmm tagger instance and sentence.
        hmm tagger instance provides the tag transition probability and 
        likelihood probability requried for calculating the tag sequence.
        """
        tagSequence = []
        # implement viterbi algorithm here
        N = tagger.V
        tokens = sentence.split()
        T = len(tokens)
        # viterbi = [[ProbEntry() for j in range(T)] for i in range(N+2)]

        viterbi = [defaultdict(lambda: defaultdict(ProbEntry)) for i in range(T+2)]

        # viterbi[i] makes sures entries are unique even if 
        # a given (word, tag) occurs multiple times in the sentence.
        # Without viterbi[i], there will be circular back pointers, thereby 
        # not leading to a infinite set of tags.
        viterbi[0][self.start][self.start] = None

        # performance: avoid using dots in loops to gain perf.
        Log10TagTransitionProbability = tagger.Log10TagTransitionProbability
        Log10LikelihoodProbability = tagger.Log10LikelihoodProbability
        tagset = tagger.tagset

        # initialization step
        word = tokens[0]
        for state in tagger.tagset:
            viterbi[1][state][word].probability = log10(1)                                    \
                + tagger.Log10TagTransitionProbability(Constants.kSENTENCE_BEGIN, state)      \
                + tagger.Log10LikelihoodProbability(state, tokens[0])
            viterbi[1][state][word].tag = state
            viterbi[1][state][word].word = word
            viterbi[1][state][word].backpointer = viterbi[0][self.start][self.start]

        # recursion step
        for time in range(1, T):
            cword = tokens[time]
            pword = tokens[time-1]
            for state in tagset:
                # tuple of (previous prob. entry and prob. value)
                prbs = [
                    (viterbi[time][sp][pword], 
                        viterbi[time][sp][pword].probability                                \
                        + Log10TagTransitionProbability(sp, state)                          \
                        + Log10LikelihoodProbability(state, cword))
                    for sp in tagset 
                    ]

                backptr, logprob = max(prbs, key=itemgetter(1))
                viterbi[time+1][state][cword].probability = logprob
                viterbi[time+1][state][cword].tag = state
                viterbi[time+1][state][cword].word = cword
                viterbi[time+1][state][cword].backpointer = backptr

        # termination step
        final = max([viterbi[T][s][tokens[T-1]] for s in tagset],      \
                        key=attrgetter("probability"))
        viterbi[T+1][self.end][self.end].backpointer = final

        # return the backtrace path by following back pointers to states
        # back in time from viterbi.backpointers
        tagSequence = self.__backTrack(viterbi, T)
        return tagSequence

class FastViterbi(Decoder):
    """
    Stateless faster viterbi algorithm to decode the sequence using bigram model.
    """
    end = start = None

    def __init__(self):
        self.start = "start"
        self.end = "end"

    def __call__(self, tagger, sentence):
        """
        callable method on instance. Takes a hmm tagger instance and sentence.
        hmm tagger instance provides the tag transition probability and 
        likelihood probability requried for calculating the tag sequence.
        """
        #raise POSError("Use regular Viterbi! This has to be implemented" 
        #    + " correctly and faster than regular viterbi.")

        tagSequence = []
        # implement viterbi algorithm here
        N = tagger.V
        tokens = sentence.split()
        T = len(tokens)

        viterbi = [defaultdict(lambda: defaultdict(ProbEntry)) for i in range(2)]

        # viterbi[i] makes sures entries are unique even if 
        # a given (word, tag) occurs multiple times in the sentence.
        # Without viterbi[i], there will be circular back pointers, thereby 
        # not leading to a infinite set of tags.
        level = 0
        viterbi[level][self.start][self.start] = None

        level = 1
        # initialization step
        for state in tagger.tagset:
            viterbi[level][state][tokens[0]].probability = log10(1)                                 \
                + tagger.Log10TagTransitionProbability(Constants.kSENTENCE_BEGIN, state)            \
                + tagger.Log10LikelihoodProbability(state, tokens[0])
            viterbi[level][state][tokens[0]].tag = state
            viterbi[level][state][tokens[0]].word = tokens[0]
            viterbi[level][state][tokens[0]].backpointer = viterbi[0][self.start][self.start]

        maxd = max([viterbi[level][state][tokens[0]] for state in tagger.tagset],
                        key = attrgetter("probability"))
        tagSequence.append(maxd.tag)
        # recursion step
        for time in range(1, T):
            max_entry = None
            for state in tagger.tagset:
                # tuple of (prob. entry and prob. value)
                prbs = [
                    (viterbi[level][sp][tokens[time-1]], 
                        viterbi[level][sp][tokens[time-1]].probability                          \
                        + tagger.Log10TagTransitionProbability(sp, state)                       \
                        + tagger.Log10LikelihoodProbability(state, tokens[time]))
                    for sp in tagger.tagset 
                    ]

                level = (level+1)%2
                # reset the previous level.
                viterbi[level] = defaultdict(lambda: defaultdict(ProbEntry))

                backptr, prob = max(prbs, key=itemgetter(1))
                viterbi[level][state][tokens[time]].probability = prob
                viterbi[level][state][tokens[time]].tag = state
                viterbi[level][state][tokens[time]].word = tokens[time]
                viterbi[level][state][tokens[time]].backpointer = backptr
                
                entry = viterbi[level][state][tokens[time]]
                if not max_entry:
                    max_entry = entry
                else:
                    max_entry = entry if entry.probability > max_entry.probability else max_entry
                # max_entry = entry if max_entry and entry.probability > max_entry.probability else max_entry

            tagSequence.append(max_entry.tag)

        # termination step
        level = (level+1)%2
        final = max([viterbi[level][s][tokens[T-1]] for s in tagger.tagset],      \
                        key=attrgetter("probability"))
        viterbi[level][self.end][self.end].backpointer = final
        
        # no need to add final tag. It's going to be none
        # tagSequence.append(final.tag)
        return tagSequence

class HMMTagger(object):
    """
    POS tagger using HMM. Each word may have more tags assosicated with it.
    """

    def __init__(self, k = 0.0001, decoder = Viterbi()):
        """
        Initialize the varibles. decoder is paramterized and default decoder is viterbi.
        Default viterbi uses bigram model sequence. If you want use your own decoder, then 
        define it and pass it the HMM tagger.

        Defining a decoder: It should be callable on object instance i.e impement 
        __call__() method. Signature : def __call__(self, hmm_instance, sentence)
        """

        # make sure decoder instance implements Decoder Interface
        if not issubclass(type(decoder), Decoder):
            raise POSError("{0} doesn't implement interface {1}".format(decoder, Decoder))

        self.tagTransitions = defaultdict(lambda: defaultdict(float))
        self.likelihood = defaultdict(lambda: defaultdict(float))
        self.__decoder = decoder
        self.k = k # maybe i have to fine tune this to get better accuracy.
        self.tagset = set() # vocabulary set/ different POS tags
        self.V = 0  # vocabulary size. Total count of tags

    def Train(self, trainData):
        """
        Train the HMM using train data. i.e calculate the 
        likelihood probabilities and tag transition probabilities.
        """
        for line in trainData:
            # update the likelihood probabilities.
            # No need to track begin/end of sentences for likelihood.
            for wordtag in line:
                if wordtag.IsFirstWord() or wordtag.IsLastWord(): 
                    continue
                word, tag = wordtag
                # unpack word and tag. I can do this becusae of namedtuple
                self.likelihood[tag][word] += 1
                self.tagset.add(tag)

            words = line.words
            # update the tag transition probabilties
            for first, second in Bigrams(words).Ngrams:
                if second.IsLastWord():
                    continue
                _, fromTag = first
                _, toTag = second
                self.tagTransitions[fromTag][toTag] += 1

        # Normalize probablities
        self.__normalize()

    def __normalize(self):
        """
        Normalize the tag transition table and likelihood proabibility table.
        For easier and faster look up.
        """
        # -1 because of <s>
        # self.tagset = set(self.tagTransitions).remove(Constants.kSENTENCE_BEGIN)
        self.V = len(self.tagset)

        # If i normalize the tag transition table, 
        # I can directly use it and no need for 
        # below two methods.
        # TODO: To be implemented.

    def GetTagTransitionProbability(self, fromTag, toTag):
        """
        estimate tag transition probability using MLE with add-k smoothing -

                                  C(tag[i-1], tag[i]) + k
        P(tag[i] | tag[i-1]) =   __________________________
                                        C(tag[i-1]) + Vk

        Use add-k with k = 0.0001 as default value
        """
        prob = 0.0
        cxy = self.tagTransitions[fromTag][toTag]
        cx = sum(self.tagTransitions[fromTag].values())
        prob = (cxy + self.k) / (cx + (self.k * self.V))
        return float(prob)

    def GetLikelihoodProbability(self, tag, word):
        """
        Estimate maximum likelihood (MLE) with smoothing -

                            C(tag, word) + k
        P(word | tag) =    ___________________
                              C(tag) + Vk

        Use add-k with k = 0.0001 as default value
        """
        prob = 0.0
        ctagword = self.likelihood[tag][word]
        ctag = sum(self.likelihood[tag].values())
        prob = (ctagword + self.k) / (ctag + (self.k * self.V))
        return float(prob)

    def Log10TagTransitionProbability(self, fromTag, toTag):
        """
        Estimate log10 tag transition table. This method will never throw exception. 
        GetTagTransitionProbability will never be zero because of add-k smoothing.

        returns math.log10(P(tag[i] | tag[i-1]))
        """
        try:
            # return log10(self.tagTransitions[fromTag][toTag]+.0000001)
            return log10(self.GetTagTransitionProbability(fromTag, toTag))
        except ValueError:
            # If there's any math domain error. Just return probability as 0.
            return float(0)

    def Log10LikelihoodProbability(self, tag, word):
        """
        Estimate log10 likelihood transition table.
        GetLikelihoodProbability will never be zero because of add-k smoothing.

        returns math.log10(P(word | tag))
        """
        try:
            # return log10(self.likelihood[tag][word]+.00000001)
            return log10(self.GetLikelihoodProbability(tag,word))
        except ValueError:
            # If there's any math domain error. Just return probability as 0.
            return float(0)

    def Decode(self, sentence):
        """
        Get the POS tagging sequence for the given sentence
        by calling decoder instance. default decoder is viterbi with bigram model.
        """
        return self.__decoder(self, sentence)

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