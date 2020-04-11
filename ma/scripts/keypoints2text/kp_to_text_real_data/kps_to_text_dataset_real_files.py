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


class TextKeypointsDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, path_to_numpy_file, path_to_csv, path_to_vocab_file, transform=None):
        """Initialization"""
        self.path_to_numpy_file = path_to_numpy_file
        self.path_to_csv = path_to_csv
        self.path_to_vocab_file = path_to_vocab_file
        self.transform = transform
        self.int2word = {}
        self.df_kp_text_train = pd.DataFrame()
        old = np.load
        np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)
        self.load_data()
        # self.keypoints2sentence()
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
        for kp in df_kp["keypoints"]:
            vid_speaker = kp[:11] + kp[11:].split('-')[0]
            speaker.append(vid_speaker)

            for idx in range(len(df_text['keypoints'])):
                if vid_speaker in df_text['keypoints'][idx]:
                    kp2sentence.append([kp, df_text['text'][idx]])
                    break
        self.df_kp_text_train = pd.DataFrame(kp2sentence, columns=["keypoints", "text"])

    def __len__(self):
        """Denotes the total number of samples"""
        with open(self.path_to_csv) as f:
            return sum(1 for line in f) - 1  # subtract 1, because of header line

    def load_data(self):
        self.df_kp_text_train = pd.read_csv(self.path_to_csv)
        # print(self.df_kp_text_train['text'][0])
        # print(type(self.df_kp_text_train['text'][0]))
        # print(self.df_kp_text_train)

        # load from keypoints
        self.saved_column_kp = self.df_kp_text_train['keypoints']  # new one
        self.all_files = np.load(self.path_to_numpy_file).item()

        # load for text
        self.saved_column_text = self.df_kp_text_train['text']  # new one


        self.processed_data = self.preprocess_data(self.saved_column_text)  # preprocess

    def __getitem__(self, index):
        """Generates one sample of data"""

        X = []

        # get specific subdirectory corresponding to the index
        subdirectory = self.saved_column_kp[index]

        keys = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
        keys_x = []
        keys_y = []
        for file in self.all_files[subdirectory]:
            temp_df = self.all_files[subdirectory][file]
            # init dictionaries & write x, y values into dictionary
            for k in keys:
                keys_x.extend(temp_df['people'][0][k][0::3])
                keys_y.extend(temp_df['people'][0][k][1::3])

        # get x and y values and concat the values
        keys_x = [x for x in keys_x if isinstance(x, numbers.Number)]
        keys_y = [x for x in keys_y if isinstance(x, numbers.Number)]
        X.append(keys_x + keys_y)
        X = X[0]  # remove one parenthesis



        processed_line = [int(i) for i in self.processed_data[index]]
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
        word2int = {}
        indx = 0
        with open(self.path_to_vocab_file) as f:
            for line in f:
                word2int[line.strip()] = indx
                indx += 1
        return word2int

    def preprocess_data(self, data):
        # TODO remove the computation of the vocab file here since its already done in data_utils

        word2int = self.get_vocab_file()
        int2word = {v: k for k, v in self.get_vocab_file().items()}

        # unique_words = sorted(self.word2dictionary(data))  # get array of unique words
        # int2word = {1: "<unk>", 2: "<sos>", 3: "<eos>", 4: "."}  # add Start/End of sentence
        # int2word.update(dict(enumerate(unique_words, start=5)))  # map array of unique words to numbers
        # print(int2word)
        # inv_map = {v: k for k, v in self.get_vocab_file().items()}
        self.int2word = int2word
        print(self.int2word)

        # e.g. print: 0 : 'who'
        # word2int = {char: ind for ind, char in int2word.items()}  # map numbers to unique words
        # e.g. print: 'who': 0
        text2index = self.text2index(data, word2int)  # map sentences to words from dictionaries above
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
        :param text_array: array of words
        :param word2int: a dictionary word2int
        :return: int representation of a sentence
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
