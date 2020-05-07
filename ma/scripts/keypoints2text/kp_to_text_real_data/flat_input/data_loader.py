"""
data_loader.py:
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
from keypoints2text.kp_to_text_real_data.data_utils import DataUtils


class TextKeypointsDataset(data.Dataset):
    """
    Characterizes a dataset for PyTorch
    """

    def __init__(self, path_to_numpy_file, path_to_csv, path_to_vocab_file, transform=None, kp_max_len=0,
                 text_max_len=0):
        self.path_to_numpy_file = path_to_numpy_file
        self.path_to_csv = path_to_csv
        self.path_to_vocab_file = path_to_vocab_file
        self.transform = transform
        self.kp_max_len = kp_max_len
        self.text_max_len = text_max_len

        # init variables
        self.int2word = {}
        self.df_kp_text_train = pd.DataFrame()

        # needs to be done or .load throws error
        old = np.load
        np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)

        # load csv containing kp and text
        self.df_kp_text_train = pd.read_csv(self.path_to_csv)

        # load keypoints
        self.saved_column_kp = self.df_kp_text_train['keypoints']
        self.all_files = np.load(self.path_to_numpy_file).item()

        # load text
        self.saved_column_text = self.df_kp_text_train['text']

        # load vocab dictionaries
        self.word2int = DataUtils().vocab_word2int(self.path_to_vocab_file)  # e.g. print: 'who': 0
        np.load = old  # reset np.load back or pickle error

    def __len__(self):
        """Denotes the total number of samples"""
        with open(self.path_to_csv) as f:
            return sum(1 for line in f) - 1  # subtract 1, because of header line

    def __getitem__(self, index):
        """
        Generates one sample of data
        :param index:
        :return:
        """
        # init keypoints
        keypoints = []

        # get specific subdirectory corresponding to the index
        subdirectory = self.saved_column_kp[index]

        keys = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
        keys_x = []
        keys_y = []
        for file in self.all_files[subdirectory]:
            temp_df = self.all_files[subdirectory][file]
            # init dictionaries & write x, y values into dictionary
            for k in keys:
                keys_x.extend(temp_df['people'][0][k][0::3])
                keys_y.extend(temp_df['people'][0][k][1::3])

        # get x and y values and concat the values
        keys_x = [x for x in keys_x if isinstance(x, numbers.Number)]
        keys_y = [x for x in keys_y if isinstance(x, numbers.Number)]

        padding_length = int(self.kp_max_len / 2)
        keys_x += [0.0] * (padding_length - len(keys_x))
        keys_y += [0.0] * (padding_length - len(keys_y))

        keypoints.append(keys_x + keys_y)
        keypoints = keypoints[0]  # remove one parenthesis

        # load sentences
        # take the sentence column of .csv file and the word2int representation
        # -> transform sentence to index and take one line of it
        sentence = [int(i) for i in DataUtils().text2index(self.saved_column_text, self.word2int)[index]]
        sentence.append(self.word2int["<eos>"])  # append EOS (should be int(3))

        # Set padding length (uncomment following 2 lines for padding)
        padding_length = self.text_max_len
        sentence += [0] * (padding_length - len(sentence))

        # transform to tensor via ToTensor TODO remove class and implement here?
        if self.transform:
            keypoints = self.transform(keypoints)
            sentence = self.transform(sentence)
        return keypoints, sentence


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(np.array(sample))
