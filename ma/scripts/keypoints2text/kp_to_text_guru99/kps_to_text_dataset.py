"""
kps_to_text_dataset.py:
Based on that tutorial
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

Features:
- not able to handle "null", skip if reading "null"

"""

import torch
from torch.utils import data
import pandas as pd
import numpy as np
import numbers
from keypoints2text.kp_to_text_guru99.data_utils import DataUtils

class TextKeypointsDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, path_to_numpy_file, path_to_csv, transform=None):
        'Initialization'
        self.path_to_numpy_file = path_to_numpy_file
        self.path_to_csv = path_to_csv
        self.transform = transform
        self.int2word = {}
        old = np.load
        np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)

    def __len__(self):
        'Denotes the total number of samples'
        with open(self.path_to_csv) as f:
            return sum(1 for line in f) - 1  # subtract 1, because of header line

    def __getitem__(self, index):
        df = pd.read_csv(self.path_to_csv)
        saved_column = df['keypoints']
        'Generates one sample of data'
        # Select sample
        X = []

        # load from .npy file
        all_files = np.load(self.path_to_numpy_file).item()

        # get specific subdirectory corresponding to the index
        subdirectory = saved_column[index]

        keys = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
        keys_x = []
        keys_y = []
        for file in all_files[subdirectory]:
            temp_df = all_files[subdirectory][file]
            # init dictionaries & write x, y values into dictionary
            for k in keys:
                keys_x.extend(temp_df['people'][0][k][0::3])
                keys_y.extend(temp_df['people'][0][k][1::3])

        # get x and y values and concat the values
        keys_x = [x for x in keys_x if isinstance(x, numbers.Number)]
        keys_y = [x for x in keys_y if isinstance(x, numbers.Number)]
        X.append(keys_x + keys_y)
        X = X[0]  # remove one parenthesis
        # TODO Padding to max length

        # TODO check if EOS token is necessary
        # X.append(1.0)  # append EOS

        df_text = pd.read_csv(self.path_to_csv)
        saved_column = df_text['text']
        print(saved_column)

        processed_data = self.preprocess_data(saved_column)  # preprocess
        processed_line = [int(i) for i in processed_data[index]]
        processed_line.append(1)  # append EOS

        # Set padding length (uncomment following 3 lines for padding)
        # TODO: Padding to max length
        padding_length = 20
        processed_line += ['0'] * (padding_length - len(processed_line))
        processed_line = [int(i) for i in processed_line]

        if self.transform:
            X = self.transform(X)
            processed_line = self.transform(processed_line)
        return X, processed_line

    def preprocess_data(self, data):
        unique_words = self.word2dictionary(data)  # get array of unique words
        int2word = {0: "UNK", 1: "SOS", 2: "EOS"}  # add Start/End of sentence
        int2word.update(dict(enumerate(unique_words, start=3)))  # map array of unique words to numbers
        self.int2word = int2word
        # e.g. print: 0 : 'who'
        word2int = {char: ind for ind, char in int2word.items()}  # map numbers to unique words
        # e.g. print: 'who': 0
        text2index = self.text2index(data, word2int)  # map sentences to words from dictionaries above
        single_sentence_tensor = torch.tensor(text2index[0]).view(-1, 1)  # get one sentence and turn it into a tensor

        # DataUtils().int2text([9, 16, 4, 4, 70, 4], int2word)

        return text2index

    def word2dictionary(self, text_array):
        """
        All words used in the texts split into an array of unique words
        :param text_array: array containing texts
        :return: set of all unique words used in the csv file
        """
        words = set()
        for sentence in text_array:
            for word in sentence.split(' '):
                words.add(word)
        return words

    def text2index(self, text_array, word2int):
        """
        use a word2int representation to turn an array of word sentences into an array of indices
        :param text_array:
        :param word2int:
        :return:
        """
        text2index = []
        for sentence in text_array:
            indexes = []
            for word in sentence.split(' '):
                indexes.append(word2int.get(word))
            text2index.append(indexes)
        return text2index


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(np.array(sample))
