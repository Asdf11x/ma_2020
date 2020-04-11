"""
data_utils.py: script for data processing during runtime
"""

import torch
from torch.utils import data
import pandas as pd
import numpy as np
import numbers


class DataUtils:

    # def __init__(self, int2word, word2int):
    #     self.int2word = int2word
    #     self.word2int = word2int

    def int2text(self, indices, int2word):
        result = []
        for element in indices:
            if element in int2word:
                result.append(int2word[element])
            else:
                result.append("<unk>")
        return result
