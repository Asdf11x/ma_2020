"""
d_loader_test_1.py:
trying from this tutorial
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

"""

import torch
from torch.utils import data
import os
from pathlib import Path
import json
import csv
import pandas as pd
import numpy as np

class Keypoints(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, path_to_json, path_to_csv, transform=None):
        'Initialization'
        self.path_to_json = path_to_json
        self.path_to_csv = path_to_csv
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        with open(self.path_to_csv) as f:
            return sum(1 for line in f) - 1  # subtract 1, because of header line

    def __getitem__(self, index):
        df = pd.read_csv(self.path_to_csv)
        saved_column = df['keypoints']
        # print(saved_column)

        'Generates one sample of data'
        # Select sample
        X = []
        data_dir_origin = Path(self.path_to_json)
        subdirectories = saved_column
        keys = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']

        for subdir in subdirectories:
            json_files = [pos_json for pos_json in os.listdir(data_dir_origin / subdir)
                          if pos_json.endswith('.json')]
            # load files from one folder into dictionary
            keys_x = []
            keys_y = []
            for file in json_files:
                temp_df = json.load(open(data_dir_origin / subdir / file))
                # init dictionaries & write x, y values into dictionary
                for k in keys:
                    keys_x.extend(temp_df['people'][0][k][0::3])
                    keys_y.extend(temp_df['people'][0][k][1::3])

            # [[x0, y0],[x1, y1]]
            # X.append(list(map(list, zip(keys_x, keys_y))))

            # [x0, xn, y0, yn]
            X.append(keys_x[:500] + keys_y[:500])

        if self.transform:
            X = self.transform(X)

        return X[index]

class Text(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, path_to_json, path_to_csv, transform):
        'Initialization'
        self.path_to_json = path_to_json
        self.path_to_csv = path_to_csv
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        with open(self.path_to_csv) as f:
            return sum(1 for line in f) - 1  # subtract 1, because of header line

    def __getitem__(self, index):
        df = pd.read_csv(self.path_to_csv)
        saved_column = df['text']
        # print(saved_column)

        processed_data = self.preprocess_data(saved_column)  # preprocess
        processed_line = [float(i) for i in processed_data[index]]
        processed_line += ['0'] * (10 - len(processed_line))
        processed_line = [float(i) for i in processed_line]
        print(processed_line)

        if self.transform:
            processed_line = self.transform(processed_line)
        return processed_line

    def preprocess_data(self, data):
        dict_DE = {}
        dict_DE = self.word2dictionary(data)
        int2word_DE = dict(enumerate(dict_DE))
        word2int_DE = {char: ind for ind, char in int2word_DE.items()}
        text2index_DE = self.text2index(data, word2int_DE)
        te_DE = torch.tensor(text2index_DE[0]).view(-1, 1)
        return text2index_DE

    # split words into dictionary
    def word2dictionary(self, text_array):
        words = set()
        for sentence in text_array:
            for word in sentence.split(' '):
                words.add(word)
        return words

    # use a word2int representation to turn an array of word sentences into an array
    # of indices
    def text2index(self, text_array, word2int):
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
        # image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        # return {'image': torch.from_numpy(image),
        #         'landmarks': torch.from_numpy(landmarks)}
        # print(sample.dtype)
        return torch.from_numpy(np.array(sample))
