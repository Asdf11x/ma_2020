"""
text_to_kps_test.py: Test and print the computed data from text_to_kps_dataset.py

- use .npy file (using full dictionary of .json files)
- use .csv file (containing the connection between text <-> keypoint folder)
"""

from Dataloader.text_to_kp.text_to_kps_dataset import TextKeypointsDataset
from Dataloader.text_to_kp.text_to_kps_dataset import ToTensor
import torch
import torch.utils
import torch.utils.data


class PrintDataset():

    def datasets_tests(self):
        # set path to .npy file (using full dictionary of .json files)
        # set path to .csv file, containing the connection between text <-> keypoint folder
        # load Dataset, Dataloader and Iterator
        text2kp = TextKeypointsDataset(path_to_numpy_file="own_data/all_files_normalized.npy",
                                       path_to_csv='own_data/sentences.csv', transform=ToTensor())
        keypoints_loader = torch.utils.data.DataLoader(text2kp, batch_size=1, shuffle=True, num_workers=0)
        it = iter(keypoints_loader)
        epochs = 10

        for i in range(epochs):
            try:
                iterator_data = next(it)
                # get size and content of text
                print("Text:")
                print(iterator_data[0].size())
                print(iterator_data[0])

                # get size and content of keypoints (limit print to first 20 keypoints)
                print("Keypoints:")
                print(iterator_data[1].size())
                print(iterator_data[1][0][0][:20])
                print("--- " * 10 + "\n")

            except StopIteration:  # reinitialize data loader if epochs > amount of data
                it = iter(keypoints_loader)


if __name__ == '__main__':
    testy = PrintDataset()
    testy.datasets_tests()
