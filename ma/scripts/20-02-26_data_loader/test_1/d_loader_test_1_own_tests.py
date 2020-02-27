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
subdirectories = [x[1] for x in os.walk(path_to_json)]
subdirectories = subdirectories[0]
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