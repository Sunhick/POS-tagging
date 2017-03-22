#!/usr/bin/python3

"""
Show progress bar

reference url: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
"""

import os

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Show progress bar
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    fLength = int(length * iteration // total)
    bars = fill * fLength + '-' * (length - fLength)
    
    # progressbar = \
    #     lambda prefix, bars, precent, suffix: "\r%s |%s| %s%% %s".format(prefix, bars, percent, suffix)
    progressbar = ("\r{0} |{1}| {2}% {3}").format(prefix, bars, percent, suffix)
    print(progressbar, end = '\r')
    
    if iteration == total: 
        print(os.linesep)