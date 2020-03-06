import os
from pathlib import Path
import json
import csv
import pandas as pd
import numpy as np
from Dataloader.text_to_kp.text_to_kps_dataset import TextKeypointsDataset
from Dataloader.text_to_kp.text_to_kps_dataset import ToTensor
import matplotlib.pyplot as plt
import torch
import torch.utils
import torch.utils.data

plt.ion()

class Testy():

    def datasets_tests(self):
        text2kp = TextKeypointsDataset(path_to_numpy_file="own_data/all_files_normalized.npy", path_to_csv='own_data/sentences.csv', transform=ToTensor())
        keypoints_loader = torch.utils.data.DataLoader(text2kp, batch_size=1, shuffle=True, num_workers=0)

        it = iter(keypoints_loader)
        first = next(it)
        # print(first)
        print(len(first))
        print(first[0].size())
        print(first[1].size())
        print(first[1][0][0][:20])



if __name__ == '__main__':
    testy = Testy()
    testy.datasets_tests()