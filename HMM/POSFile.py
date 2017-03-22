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