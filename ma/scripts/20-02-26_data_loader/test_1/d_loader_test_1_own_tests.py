"""
d_loader_test_1.py:
trying from this tutorial
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

"""

import os
from pathlib import Path
import json
import csv
import pandas as pd
import numpy as np
from d_loader_test_1_own import Keypoints
from d_loader_test_1_own import ToTensor
from d_loader_test_1_own import Text
import matplotlib.pyplot as plt
import torch
import torch.utils
import torch.utils.data

plt.ion()




class Testy():

    def random_tests(self):

        # output_size = (56465,321,321)
        # assert isinstance(output_size, (int, tuple))

        path_to_csv = r"C:\Eigene_Programme\Git-Data\Own_Repositories\ma_2020\ma\scripts\20-02-26_data_loader\test_1\own_data\sentences.csv"

        with open(path_to_csv) as f:
            print(sum(1 for line in f))

        print("- " * 5)

        df = pd.read_csv(path_to_csv)
        saved_column = df['keypoints']
        print(saved_column)


        path_to_json = r"C:\Eigene_Programme\Git-Data\Own_Repositories\ma_2020\ma\scripts\20-02-26_data_loader\test_1\own_data\json"

        subdirectories = [x[1] for x in os.walk(path_to_json)]
        subdirectories = saved_column

        print(len(subdirectories))

        X = []
        i = 0
        data_dir_origin = Path(path_to_json)
        # subdirectories = [x[1] for x in os.walk(path_to_json)]
        # subdirectories = subdirectories[0]
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

        print(X[i])

        df = pd.read_csv(path_to_csv)
        saved_column = df['text']
        print(saved_column[i])

    def datasets_tests(self):
        keypoints = Keypoints(path_to_json="own_data/json", path_to_csv='own_data/sentences.csv', transform=ToTensor())
        text = Text(path_to_json="own_data/json", path_to_csv='own_data/sentences.csv', transform=ToTensor())
        keypoints_loader = torch.utils.data.DataLoader(keypoints, batch_size=2, shuffle=True, num_workers=4)
        text_loader = torch.utils.data.DataLoader(text, batch_size=2, shuffle=True, num_workers=4)




        fig = plt.figure()
        print(len(text))
        for i in range(len(text)):
            sample = keypoints[i]
            sample_text = text[i]
            print(sample)
            print(torch.tensor(sample, dtype=torch.long).view(-1, 1))
            print(sample_text.shape)

            # print(i, sample.size())

            # ax = plt.subplot(1, 4, i + 1)
            # plt.tight_layout()
            # ax.set_title('Sample #{}'.format(i))
            # ax.axis('off')
            # show_landmarks(**sample)

            # if i == 3:
            #     plt.show()
            #     break


if __name__ == '__main__':
    testy = Testy()
    testy.datasets_tests()