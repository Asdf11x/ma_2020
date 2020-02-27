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

class Keypoints(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, path_to_json, path_to_csv):
        'Initialization'
        self.path_to_json = path_to_json
        self.path_to_csv = path_to_csv

    def __len__(self):
        'Denotes the total number of samples'
        with open(self.path_to_csv) as f:
            return sum(1 for line in f)

    def __getitem__(self, index):
        df = pd.read_csv(self.path_to_csv)
        saved_column = df['keypoints']
        print(saved_column)

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

            X.append(keys_x + keys_y)

        return X[index]

class Text(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, path_to_json, path_to_csv):
        'Initialization'
        self.path_to_json = path_to_json
        self.path_to_csv = path_to_csv

    def __len__(self):
        'Denotes the total number of samples'
        with open(self.path_to_csv) as f:
            return sum(1 for line in f)

    def __getitem__(self, index):
        df = pd.read_csv(self.path_to_csv)
        saved_column = df['text']
        return saved_column[index]