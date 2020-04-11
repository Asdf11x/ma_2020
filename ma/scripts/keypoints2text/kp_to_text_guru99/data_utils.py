"""
data_utils.py: script for data processing during runtime
"""

import torch
from torch.utils import data
import pandas as pd
import numpy as np
import numbers


class DataUtils:

    def __init__(self, path_to_vocab_file):
        self.path_to_vocab_file = path_to_vocab_file

    def vocab_word2int(self, path_to_vocab_file):
        word2int = {}
        indx = 0
        with open(path_to_vocab_file) as f:
            for line in f:
                word2int[line.strip()] = indx
                indx += 1
        return word2int

    def vocab_int2word(self, path_to_vocab_file):
        word2int = {}
        indx = 0
        with open(path_to_vocab_file) as f:
            for line in f:
                word2int[line.strip()] = indx
                indx += 1
        int2word = {v: k for k, v in word2int.items()}
        return int2word

    def int2text(self, indices, int2word):
        result = []
        for element in indices:
            if element in int2word:
                result.append(int2word[element])
            else:
                result.append("<unk>")
        return result
