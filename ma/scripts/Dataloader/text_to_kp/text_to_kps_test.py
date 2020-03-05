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
        print(first[0])
        print(first[1].size())
        print(first[1][0][:20])

        # second = next(it)

        # text = Text(path_to_json="own_data/json", path_to_csv='own_data/sentences.csv', transform=ToTensor())
        # text_loader = torch.utils.data.DataLoader(text, batch_size=2, shuffle=True, num_workers=4)




        # fig = plt.figure()
        # print(len(text))
        # for i in range(len(text)):
        #     sample = keypoints[i]
        #     sample_text = text[i]
        #     print(sample)
        #     print(torch.tensor(sample, dtype=torch.long).view(-1, 1))
        #     print(sample_text.shape)

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