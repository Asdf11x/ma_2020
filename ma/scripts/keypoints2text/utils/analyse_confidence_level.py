"""
analyse_confidence_level.py: prints the confidence levels of the datasets

28.07.20: init

Tasks:
    - read numpy file of dataset (train, val or test)
    - print confidence level for
        - pose, face, hand_l/r
    -

"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import json


class ConfidenceAnalysis:

    def __init__(self, path_to_numpy_file, path_to_target):
        self.path_to_numpy_file = Path(path_to_numpy_file)
        self.path_to_target = Path(path_to_target)
        self.keys = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
        self.confidences = {"pose_keypoints_2d": [], "face_keypoints_2d": [], "hand_left_keypoints_2d": [], "hand_right_keypoints_2d": []}
        old = np.load
        np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)

    def main(self):
        self.get_confidence()

    def get_confidence(self):
        """ load from .npy file """
        kp_files = np.load(self.path_to_numpy_file).item()

        df_kp = pd.DataFrame(kp_files.keys(), columns=["keypoints"])
        kp2sentence = []
        print(df_kp["keypoints"])
        # get confidence from subdir -> file -> keys
        for subdir in df_kp["keypoints"]:
            for file in kp_files[subdir]:
                for k in self.keys:
                    self.confidences[k].append(np.mean(kp_files[subdir][file]['people'][0][k][2::3]))

        # mean from all subdir means
        for k in self.keys:
            self.confidences[k] = np.mean(self.confidences[k])

        print(self.path_to_numpy_file)
        print(self.confidences)

        # print keys
        with open(self.path_to_target / "confidence_mean.txt", 'a') as f:
            f.write(str(self.path_to_numpy_file) + "\n")
            f.write(str(self.confidences))
            f.write("\n")


if __name__ == '__main__':
    # file with sentences
    if len(sys.argv) > 1:
        path_to_numpy_file = sys.argv[1]
    else:
        print("Set path to npy file")
        sys.exit()

    # target
    if len(sys.argv) > 2:
        path_to_target = sys.argv[2]
    else:
        print("Set path to target folder")
        sys.exit()
    npy = ConfidenceAnalysis(path_to_numpy_file, path_to_target)
    npy.main()
