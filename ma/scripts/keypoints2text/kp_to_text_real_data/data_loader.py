"""
data_loader_flat.py:
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
import random

try:
    from keypoints2text.kp_to_text_real_data.data_utils import DataUtils
except ImportError:  # server uses different imports than local
    from data_utils import DataUtils


class TextKeypointsDataset(data.Dataset):
    """
    Characterizes a dataset for PyTorch
    """

    def __init__(self, path_to_numpy_file, path_to_csv, path_to_vocab_file, input_length, transform=None, kp_max_len=0,
                 text_max_len=0):
        self.path_to_numpy_file = path_to_numpy_file
        self.path_to_csv = path_to_csv
        self.path_to_vocab_file = path_to_vocab_file
        self.input_length = input_length
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

        # get amount of data
        self.amount_of_files = DataUtils().get_file_length(self.path_to_csv)

        np.load = old  # reset np.load back or pickle error

    def __len__(self):
        """Denotes the total number of samples"""
        with open(self.path_to_csv) as f:
            return sum(1 for line in f) - 1  # subtract 1, because of header line
            #####################################################################
            # TESTING SETTING REMOVE WHEN NOT USED ANYMORE
            ################################################
            # return 20  # subtract 1, because of header line

    def __getitem__(self, index):
        """
        Generates one sample of data
        :param index:
        :return:
        """
        # init keypoints
        keypoints = []
        keys_per_folder = []
        while 1:
            # get specific subdirectory corresponding to the index
            subdirectory = self.saved_column_kp[index]
            keys = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
            keys_per_folder = []
            keys_per_folder_256 = []

            for file in self.all_files[subdirectory]:
                temp_df = self.all_files[subdirectory][file]
                # init dictionaries & write x, y values into dictionary
                keys_x = []
                keys_y = []
                keys_x_256 = []
                keys_y_256 = []
                for k in keys:
                    if k == "pose_keypoints_2d":
                        # keys_x.extend(temp_df['people'][0][k][0::3])
                        # keys_y.extend(temp_df['people'][0][k][1::3])

                        keys_x_256 = temp_df['people'][0][k][0:25:3]
                        keys_x_256.extend(temp_df['people'][0][k][45:55:3])
                        keys_y_256 = temp_df['people'][0][k][1:26:3]
                        keys_y_256.extend(temp_df['people'][0][k][46:56:3])
                    else:
                        # keys_x.extend(temp_df['people'][0][k][0::3])
                        # keys_y.extend(temp_df['people'][0][k][1::3])

                        keys_x_256.extend(temp_df['people'][0][k][0::3])
                        keys_y_256.extend(temp_df['people'][0][k][1::3])

                # keys_per_folder.append(keys_x + keys_y)
                keys_per_folder.append(keys_x_256 + keys_y_256 + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # transform to tensor here
            if self.transform:
                keys_per_folder = self.transform(keys_per_folder)

            # if padding is not used, break the loop here
            if self.kp_max_len == 0:
                break

            # if the data size is smaller than the padding size, then break the loop
            if keys_per_folder.size(0) <= self.kp_max_len:
                break
            # if the data size is bigger than the padding size, search for new data (might just cut the data?)
            else:
                index = random.randint(0, self.amount_of_files - 2)

        # keypoints padding
        # check if padding is activated (kp_max_len must be greater than 0)
        if self.kp_max_len > 0:
            if self.kp_max_len < keys_per_folder.size(0):
                temp_max_len = keys_per_folder.size(0)
            else:
                temp_max_len = self.kp_max_len
            length = keys_per_folder.size(0)
            keys = torch.zeros(temp_max_len, self.input_length)
            source = keys_per_folder
            keys[:length, :] = source


        else:
            keys = keys_per_folder

        # load sentences
        # take the sentence column of .csv file and the word2int representation
        # -> transform sentence to index and take one line of it
        # print(self.saved_column_text[index])
        sentence = [int(i) for i in DataUtils().text2index([self.saved_column_text[index]], self.word2int)[0]]
        sentence.append(self.word2int["<eos>"])  # append EOS (should be int(3))

        # Set padding length (uncomment following 2 lines for padding)
        padding_length = self.text_max_len
        sentence += [0] * (padding_length - len(sentence))
        # transform to tensor via ToTensor TODO remove class and implement here?
        if self.transform:
            sentence = self.transform(sentence)
        # print(sentence)
        return keys, sentence


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(np.array(sample))
