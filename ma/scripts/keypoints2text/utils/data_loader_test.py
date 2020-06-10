"""
kps_to_text_test.py: Test and print the computed data from kps_to_text_dataset.py

- use .npy file (using full dictionary of .json files)
- use .csv file (containing the connection between text <-> keypoint folder)
"""


import torch
import torch.utils
import torch.utils.data
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
try:
    from keypoints2text.kp_to_text_real_data.data_loader import TextKeypointsDataset
    from keypoints2text.kp_to_text_real_data.data_loader import ToTensor
except ImportError:  # server uses different imports than local
    from data_loader import TextKeypointsDataset
    from data_loader import ToTensor

class PrintDataset:

    def datasets_tests(self):
        # set path to .npy file (using full dictionary of .json files)
        # set path to .csv file, containing the connection between text <-> keypoint folder
        # load Dataset, Dataloader and Iterator
        text2kp = TextKeypointsDataset(
            path_to_numpy_file="C:\\Users\\Asdf\\Downloads\\How2Sign_samples\\all_files_normalized.npy",
            path_to_csv="C:\\Users\\Asdf\\Downloads\\How2Sign_samples\\text\\3_linked_to_npy\\how2sign.test.id_transformed.txt_2npy.txt",
            path_to_vocab_file="C:\\Users\\Asdf\\Downloads\\How2Sign_samples\\text\\1_vocab_list\\vocab_merged.txt",
            input_length=256,
            transform=ToTensor(),
            kp_max_len=0,
            text_max_len=0)


        keypoints_loader = torch.utils.data.DataLoader(text2kp, batch_size=1, shuffle=True, num_workers=0)
        epochs = 5
        self.print_sample_data(keypoints_loader, epochs)
        self.get_length_show_data(keypoints_loader)

    def print_sample_data(self, keypoints_loader, epochs):
        it = iter(keypoints_loader)

        for i in range(epochs):
            try:
                iterator_data = next(it)

                # get size and content of keypoints (limit print to first 20 keypoints)
                print("Keypoints:")
                print(iterator_data[0].size())
                print(iterator_data[0][0][:20])

                # get size and content of text
                print("Text:")
                print(iterator_data[1].size())
                print(iterator_data[1])
                print("--- " * 10 + "\n")
                # show out_ten as flat list
                # out_ten = torch.as_tensor(iterator_data[1], dtype=torch.long).view(-1, 1)
                # flat_list = []
                # for sublist in out_ten.tolist():
                #     for item in sublist:
                #         flat_list.append(item)

            except StopIteration:  # reinitialize data loader if epochs > amount of data
                it = iter(keypoints_loader)

    def get_length_show_data(self, keypoints_loader):
        source_max = 0
        target_max = 0
        source_max_saved = []
        target_max_saved = []
        it = iter(keypoints_loader)
        while 1:
            try:
                iterator_data = next(it)
            except StopIteration:  # if StopIteration is raised, all data of a loader is used
                break
            source_len = torch.as_tensor(iterator_data[0], dtype=torch.float).size(1)
            target_len = torch.as_tensor(iterator_data[1], dtype=torch.long).size(1)

            source_max_saved.append(source_len)
            target_max_saved.append(target_len)

            if source_len > source_max:
                source_max = source_len

            if target_len > target_max:
                target_max = target_len

        plt.hist(source_max_saved)
        plt.title("Amount of frames")
        plt.ylabel("Folders")
        plt.xlabel("Frames")
        plt.savefig('source_max_saved_test.png')
        # plt.show()

        plt.figure()
        plt.hist(target_max_saved)
        plt.title("Amount of sentences")
        plt.ylabel("Amount of sentences")
        plt.xlabel("Amount of Words")
        plt.savefig('target_max_saved_test.png')
        # plt.show()


if __name__ == '__main__':
    testy = PrintDataset()
    testy.datasets_tests()
