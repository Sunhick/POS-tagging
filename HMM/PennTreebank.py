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