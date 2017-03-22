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