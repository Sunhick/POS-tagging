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