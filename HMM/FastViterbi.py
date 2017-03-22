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