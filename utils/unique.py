#!/usr/bin/python3

"""
helper file for POSTagger
"""

__author__ = "Sunil"
__copyright__ = "Copyright (c) 2017 Sunil"
__license__ = "MIT License"
__email__ = "suba5417@colorado.edu"
__version__ = "0.1"

import POSTagger as pt

def main():
    file = pt.POSFile('berp-POS-train.txt')
    ct = []
    for line in file.Lines:
        sentence = line.Sentence
        if [sentence] not in ct:
            ct.append([sentence])
    print(len(ct))

if __name__ == "__main__":
    main()