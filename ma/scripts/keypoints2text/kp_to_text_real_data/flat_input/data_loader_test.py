"""
kps_to_text_test.py: Test and print the computed data from kps_to_text_dataset.py

- use .npy file (using full dictionary of .json files)
- use .csv file (containing the connection between text <-> keypoint folder)
"""

from keypoints2text.kp_to_text_real_data.flat_input.data_loader import TextKeypointsDataset
from keypoints2text.kp_to_text_real_data.flat_input.data_loader import ToTensor
import torch
import torch.utils
import torch.utils.data
from matplotlib import pyplot as plt


class PrintDataset:

    def datasets_tests(self):
        # set path to .npy file (using full dictionary of .json files)
        # set path to .csv file, containing the connection between text <-> keypoint folder
        # load Dataset, Dataloader and Iterator
        text2kp = TextKeypointsDataset(
            path_to_numpy_file=r"C:\Users\Asdf\Downloads\How2Sign_samples\all_files_normalized.npy",
            path_to_csv=r"/keypoints2text/data/text/3_linked_to_npy/how2sign.test.id_transformed.txt_2npy.txt",
            path_to_vocab_file=r"/keypoints2text/data/text/1_vocab_list/how2sign.test.id_vocab.txt",
            kp_max_len=120000,
            text_max_len=120,
            transform=ToTensor())

        keypoints_loader = torch.utils.data.DataLoader(text2kp, batch_size=1, shuffle=True, num_workers=0)
        epochs = 5
        self.print_sample_data(keypoints_loader, epochs)
        # self.get_length_show_data(keypoints_loader)

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
            source_len = torch.as_tensor(iterator_data[0], dtype=torch.float).view(-1, 1).size()[0]
            target_len = torch.as_tensor(iterator_data[1], dtype=torch.long).view(-1, 1).size()[0]

            source_max_saved.append(source_len)
            target_max_saved.append(target_len)

            if source_len > source_max:
                source_max = source_len

            if target_len > target_max:
                target_max = target_len

        plt.hist(source_max_saved)
        plt.savefig('source_max_saved_test.png')
        plt.show()

        plt.hist(target_max_saved)
        plt.savefig('target_max_saved_test.png')
        plt.show()


if __name__ == '__main__':
    testy = PrintDataset()
    testy.datasets_tests()
