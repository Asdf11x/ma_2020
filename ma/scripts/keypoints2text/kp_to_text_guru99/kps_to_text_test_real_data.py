"""
kps_to_text_test.py: Test and print the computed data from kps_to_text_dataset.py

- use .npy file (using full dictionary of .json files)
- use .csv file (containing the connection between text <-> keypoint folder)
"""

from keypoints2text.kp_to_text_guru99.kps_to_text_dataset_real_files import TextKeypointsDataset
from keypoints2text.kp_to_text_guru99.kps_to_text_dataset_real_files import ToTensor
import torch
import torch.utils
import torch.utils.data


class PrintDataset():

    def datasets_tests(self):
        # set path to .npy file (using full dictionary of .json files)
        # set path to .csv file, containing the connection between text <-> keypoint folder
        # load Dataset, Dataloader and Iterator
        text2kp = TextKeypointsDataset(path_to_numpy_file="own_data/all_files_normalized.npy",
                                       path_to_csv=r'C:\Users\Asdf\Downloads\How2Sign_samples\text_vocab\how2sign.test.id_transformed_sample.txt',
                                       path_to_vocab_file=r'C:\Users\Asdf\Downloads\How2Sign_samples\text_vocab\how2sign.test.id_vocab.txt',
                                       transform=ToTensor())
        keypoints_loader = torch.utils.data.DataLoader(text2kp, batch_size=1, shuffle=True, num_workers=0)
        it = iter(keypoints_loader)
        epochs = 10

        for i in range(epochs):
            try:
                iterator_data = next(it)
                # get size and content of text
                print("Keypoints:")
                print(iterator_data[0].size())
                print(iterator_data[0][0][:20])

                # get size and content of keypoints (limit print to first 20 keypoints)
                print("Text:")
                print(iterator_data[1].size())
                print(iterator_data[1])
                print("--- " * 10 + "\n")

            except StopIteration:  # reinitialize data loader if epochs > amount of data
                it = iter(keypoints_loader)


if __name__ == '__main__':
    testy = PrintDataset()
    testy.datasets_tests()
