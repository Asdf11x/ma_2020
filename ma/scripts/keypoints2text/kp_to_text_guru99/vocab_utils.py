"""
vocab_utils.py: script for processing data before runtime

features:
    - read text from sentences.csv
    - build vocab file, containg all vocabs used in the sentences.csv file

"""

import torch
from torch.utils import data
import pandas as pd
import numpy as np
import numbers

class VocabUtils:

    def __init__(self, path_to_numpy_file, path_to_csv,):
        'Initialization'
        self.path_to_numpy_file = path_to_numpy_file
        self.path_to_csv = path_to_csv

    def getData(self):
        df_text = pd.read_csv(self.path_to_csv)
        saved_column = df_text['text']