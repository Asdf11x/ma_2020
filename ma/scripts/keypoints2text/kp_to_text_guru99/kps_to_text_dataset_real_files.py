"""
kps_to_text_dataset_real_files.py:
Based on that tutorial
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

Features:
- not able to handle "null", skip if reading "null"

"""

import torch
from torch.utils import data
import pandas as pd
import numpy as np
import numbers
from keypoints2text.kp_to_text_guru99.data_utils import DataUtils
import csv


class TextKeypointsDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, path_to_numpy_file, path_to_csv, path_to_vocab_file, transform=None):
        """Initialization"""
        self.path_to_numpy_file = path_to_numpy_file
        self.path_to_csv = path_to_csv
        self.path_to_vocab_file = path_to_vocab_file
        self.transform = transform
        self.int2word = {}
        old = np.load
        np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)
        self.df_kp_text_train = pd.DataFrame()
        self.keypoints2sentence()
        self.get_vocab_file()

    def keypoints2sentence(self):
        # load from .npy file
        kp_files = np.load(self.path_to_numpy_file).item()
        df_kp = pd.DataFrame(kp_files.keys(), columns=["keypoints"])
        kp2sentence = []
        # print(df_kp)

        d = {'keypoints': [], 'text': []}
        with open(self.path_to_csv) as f:
            for line in f:
                d['keypoints'].append(line.split(" ")[0])
                d['text'].append(" ".join(line.split()[1:]))
        df_text = pd.DataFrame(d)

        speaker = []
        text_en = []
        for kp in df_kp["keypoints"]:
            vid_speaker = kp[:11] + kp[11:].split('-')[0]
            speaker.append(vid_speaker)

            for idx in range(len(df_text['keypoints'])):
                if vid_speaker in df_text['keypoints'][idx]:
                    kp2sentence.append([kp, df_text['text'][idx]])
                    break
        self.df_kp_text_train = pd.DataFrame(kp2sentence, columns=["keypoints", "text"])
        # dffucc.to_csv(r'kp2sentence.txt', index=False)
        # print(self.df_kp_text_train)

    def __len__(self):
        """Denotes the total number of samples"""
        with open(self.path_to_csv) as f:
            return sum(1 for line in f) - 1  # subtract 1, because of header line

    def __getitem__(self, index):
        """Generates one sample of data"""
        d = {'keypoints': [], 'text': []}
        with open(self.path_to_csv) as f:
            for line in f:
                d['keypoints'].append(line.split()[0])
                d['text'].append(" ".join(line.split()[1:]))
        df = pd.DataFrame(d)

        # saved_column = df['keypoints']
        saved_column = self.df_kp_text_train['keypoints']  # new one
        # Select sample
        X = []

        # load from .npy file
        all_files = np.load(self.path_to_numpy_file).item()

        # get specific subdirectory corresponding to the index
        subdirectory = saved_column[index]

        keys = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
        keys_x = []
        keys_y = []
        for file in all_files[subdirectory]:
            temp_df = all_files[subdirectory][file]
            # init dictionaries & write x, y values into dictionary
            for k in keys:
                keys_x.extend(temp_df['people'][0][k][0::3])
                keys_y.extend(temp_df['people'][0][k][1::3])

        # get x and y values and concat the values
        keys_x = [x for x in keys_x if isinstance(x, numbers.Number)]
        keys_y = [x for x in keys_y if isinstance(x, numbers.Number)]
        X.append(keys_x + keys_y)
        X = X[0]  # remove one parenthesis

        # saved_column = df['text']
        saved_column = self.df_kp_text_train['text']  # new one

        processed_data = self.preprocess_data(saved_column)  # preprocess
        processed_line = [int(i) for i in processed_data[index]]
        processed_line.append(2)  # append EOS

        # Set padding length (uncomment following 2 lines for padding)
        # padding_length = 50
        # processed_line += ['0'] * (padding_length - len(processed_line))
        processed_line = [int(i) for i in processed_line]

        if self.transform:
            X = self.transform(X)
            processed_line = self.transform(processed_line)
        return X, processed_line

    def get_vocab_file(self):
        int2word = {}
        indx = 0
        with open(self.path_to_vocab_file) as f:
            for line in f:
                int2word[line.strip()] = indx
                indx += 1
        return int2word


    def preprocess_data(self, data):
        unique_words = self.word2dictionary(data)  # get array of unique words

        int2word = {0: "<unk>", 1: "<sos>", 2: "<eos>", 3: "."}  # add Start/End of sentence
        int2word.update(dict(enumerate(unique_words, start=4)))  # map array of unique words to numbers
        self.int2word = int2word

        # e.g. print: 0 : 'who'
        word2int = {char: ind for ind, char in int2word.items()}  # map numbers to unique words
        # e.g. print: 'who': 0
        text2index = self.text2index(data, self.get_vocab_file())  # map sentences to words from dictionaries above
        single_sentence_tensor = torch.tensor(text2index[0]).view(-1, 1)  # get one sentence and turn it into a tensor

        # DataUtils().int2text([9, 16, 4, 4, 70, 4], int2word)

        return text2index

    def word2dictionary(self, text_array):
        """
        All words used in the texts split into an array of unique words
        :param text_array: array containing texts
        :return: set of all unique words used in the csv file
        """
        words = set()
        for sentence in text_array:
            for word in sentence.split(' '):
                words.add(word)
        return words

    def text2index(self, text_array, word2int):
        """
        use a word2int representation to turn an array of word sentences into an array of indices
        :param text_array:
        :param word2int:
        :return:
        """
        text2index = []
        for sentence in text_array:
            indexes = []
            for word in sentence.split(' '):
                indexes.append(word2int.get(word))
            text2index.append(indexes)
        return text2index


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(np.array(sample))
