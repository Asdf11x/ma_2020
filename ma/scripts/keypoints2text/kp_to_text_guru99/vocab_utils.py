"""
vocab_utils.py: script for processing data before runtime

features:
    - read text from sentences.csv
    - build vocab file, containg all vocabs used in the sentences.csv file

"""

import torch
from torch.utils import data
import pandas as pd
import numpy as np
import numbers
import time
from pathlib import Path
import os
import sys
import spacy
from spacy.attrs import ORTH, LEMMA, NORM, TAG
import re
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex

# path_to_numpy_files = Path(sys.argv[1])
path_hard_coded = r"C:\Users\Asdf\Downloads\How2Sign_samples\text\how2sign.val.id.en"


# path_hard_coded = r"C:\Users\Asdf\Downloads\How2Sign_samples\text\aaa.txt"

def custom_tokenizer(nlp):
    infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"~]''')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                     suffix_search=suffix_re.search,
                     infix_finditer=infix_re.finditer,
                     token_match=None)


nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = custom_tokenizer(nlp)


def test_spacy():
    # nlp = en_core_web_sm.load()
    doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
    for token in doc:
        print(token.text)


# test_spacy()

class VocabUtils:

    def __init__(self, path_to_sentences, path_to_target_dir):
        self.path_to_sentences = Path(path_to_sentences)
        self.path_to_target_dir = path_to_target_dir
        self.path_to_target_file = ""
        self.create_folders()
        self.getData_spacy()

    def getData_spacy(self):
        contractions = {
            "ain't": "are not",
            "aren't": "are not",
            "can't": "can not",
            "can't've": "can not have",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he had",
            "he'd've": "he would have",
            "he'll": "he will",
            "he'll've": "he will have",
            "he's": "he is",
            "how'd": "how did",
            "how'd'y": "how do you",
            "how'll": "how will",
            "how's": "how is",
            "i'd": "I would",
            "i'd've": "I would have",
            "i'll": "I will",
            "i'll've": "I will have",
            "i'm": "I am",
            "i've": "I have",
            "isn't": "is not",
            "it'd": "it would",
            "it'd've": "it would have",
            "it'll": "it will",
            "it'll've": "it will have",
            "it's": "it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "mightn't've": "might not have",
            "must've": "must have",
            "mustn't": "must not",
            "mustn't've": "must not have",
            "needn't": "need not",
            "needn't've": "need not have",
            "o'clock": "of the clock",
            "oughtn't": "ought not",
            "oughtn't've": "ought not have",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "shan't've": "shall not have",
            "she'd": "she would",
            "she'd've": "she would have",
            "she'll": "she will",
            "she'll've": "she will have",
            "she's": "she is",
            "should've": "should have",
            "shouldn't": "should not",
            "shouldn't've": "should not have",
            "so've": "so have",
            "so's": "so is",
            "that'd": "that had",
            "that'd've": "that would have",
            "that's": "that is",
            "there'd": "there would",
            "there'd've": "there would have",
            "there's": "there is",
            "they'd": "they would",
            "they'd've": "they would have",
            "they'll": "they will",
            "they'll've": "they will have",
            "they're": "they are",
            "they've": "they have",
            "to've": "to have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'd've": "we would have",
            "we'll": "we will",
            "we'll've": "we will have",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what'll've": "what will have",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "when's": "when is",
            "when've": "when have",
            "where'd": "where did",
            "where's": "where is",
            "where've": "where have",
            "who'll": "who will",
            "who'll've": "who will have",
            "who's": "who is",
            "who've": "who have",
            "why's": "why is",
            "why've": "why have",
            "will've": "will have",
            "won't": "will not",
            "won't've": "will not have",
            "would've": "would have",
            "wouldn't": "would not",
            "wouldn't've": "would not have",
            "y'all": "you all",
            "y'all'd": "you all would",
            "y'all'd've": "you all would have",
            "y'all're": "you all are",
            "y'all've": "you all have",
            "you'd": "you would",
            "you'd've": "you would have",
            "you'll": "you will",
            "you'll've": "you will have",
            "you're": "you are",
            "you've": "you have",
            "'ll": "will",
            "'m": "am",
            "'re": "are",
            "'s": "is",
            "'ve": "have",
            "n't": "not"
        }

        TOKENIZER_EXCEPTIONS = {
            # do
            "don't": [
                {ORTH: "do", LEMMA: "do"},
                {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
            "doesn't": [
                {ORTH: "does", LEMMA: "do"},
                {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
            "didn't": [
                {ORTH: "did", LEMMA: "do"},
                {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
            # can
            "can't": [
                {ORTH: "ca", LEMMA: "can"},
                {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
            "couldn't": [
                {ORTH: "could", LEMMA: "can"},
                {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
            # have
            "I've'": [
                {ORTH: "I", LEMMA: "I"},
                {ORTH: "'ve'", LEMMA: "have", NORM: "have", TAG: "VERB"}],
            "haven't": [
                {ORTH: "have", LEMMA: "have"},
                {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
            "hasn't": [
                {ORTH: "has", LEMMA: "have"},
                {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
            "hadn't": [
                {ORTH: "had", LEMMA: "have"},
                {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
            # will/shall will be replaced by will
            "I'll'": [
                {ORTH: "I", LEMMA: "I"},
                {ORTH: "'ll'", LEMMA: "will", NORM: "will", TAG: "VERB"}],
            "he'll'": [
                {ORTH: "he", LEMMA: "he"},
                {ORTH: "'ll'", LEMMA: "will", NORM: "will", TAG: "VERB"}],
            "she'll'": [
                {ORTH: "she", LEMMA: "she"},
                {ORTH: "'ll'", LEMMA: "will", NORM: "will", TAG: "VERB"}],
            "it'll'": [
                {ORTH: "it", LEMMA: "it"},
                {ORTH: "'ll'", LEMMA: "will", NORM: "will", TAG: "VERB"}],
            "won't": [
                {ORTH: "wo", LEMMA: "will"},
                {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
            "wouldn't": [
                {ORTH: "would", LEMMA: "will"},
                {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
            # be
            "I'm'": [
                {ORTH: "I", LEMMA: "I"},
                {ORTH: "'m'", LEMMA: "be", NORM: "am", TAG: "VERB"}]
        }

        # doc1 = nlp(u"Oh no he didn't. I can't and I won't. I'll know what I'm gonna do.")
        # for token in doc1:
        #     print(token.text, token.lemma_)

        corrections = {
            "'ll": "will",
            "'m": "am",
            "'re": "are",
            "'s": "is",
            "'ve": "have",
            "n't": "not"
        }
        unique_words = set()

        with open(self.path_to_sentences, encoding='utf-8') as f:
            for line in f:
                # tokenize
                doc = nlp(line)
                words = [token.text for token in doc]
                for token in words[1:]:
                    unique_words.add(token.lower())
                    # if token in contractions:
                    #     for element in contractions[token].split():
                    #         unique_words.add(element.lower())
                    # if token.isalpha():
                    #     unique_words.add(token.lower())

            print(sorted(unique_words))

            # clean tokenize
            unique_copy = unique_words.copy()
            for word in unique_words:
                if word in contractions:
                    unique_copy.remove(word)
                    for element in contractions[word].split():
                        unique_copy.add(element.lower())
                    continue
                if not word.isalpha():
                    unique_copy.remove(word)
            # unique_words = unique_copy.copy()
            # for word in unique_words:

            unique_words = unique_copy.copy()
            sorted_words = sorted(unique_words)

            sorted_words.insert(0, "UNK")
            sorted_words.insert(1, "SOS")
            sorted_words.insert(2, "EOS")

            print(len(sorted_words))
            print(sorted_words)

        with open(self.path_to_target_file, 'w') as f:
            for item in sorted_words:
                f.write("%s\n" % item)

    def getData(self):
        with open(path_hard_coded) as f:
            # data = open(path_hard_coded)
            unique_words = set()
            for line in f:
                # unique_words.add(nlp(line))

                for element in line.split()[1:]:
                    # for element in line.split():
                    #     print(element)
                    unique_words.add(element.lower())
                # print(set(line.split()))

                # print(line.split())
                # print(line)
                # text = set(word.split() for word in line)
            # df_text = pd.read_csv(path_hard_coded, header=None, sep=" ")
            sorted_words = sorted(unique_words)
            print(sorted_words)

        with open(self.path_to_target_file, 'w') as f:
            for item in sorted_words:
                f.write("%s\n" % item)

    def create_folders(self):
        """
        Create folders and filenames
        """
        # if no target dir is set, move one folder up and create folder with "_vocab attached"
        if self.path_to_target_dir == "":
            data_dir_target = self.path_to_sentences.parent.parent / (
                        str(self.path_to_sentences.parent.name) + str("_vocab"))
        else:
            data_dir_target = Path(self.path_to_target_dir)

        # create new target directory, the files will be saved there
        if not os.path.exists(data_dir_target):
            os.makedirs(data_dir_target)

        # vocab file is in target dir "_vocab.txt" attached to the original name
        self.path_to_target_file = data_dir_target / (str(self.path_to_sentences.stem) + str("_vocab.txt"))


if __name__ == '__main__':
    # file with sentences
    if len(sys.argv) > 1:
        path_to_sentences = sys.argv[1]
    else:
        print("Set path to file containing sentences")
        sys.exit()

    # target directory
    path_to_target_dir = ""
    if len(sys.argv) > 2:
        path_to_target_dir = sys.argv[2]
    start_time = time.time()
    vocab = VocabUtils(path_to_sentences, path_to_target_dir)
    print("--- %.4s seconds ---" % (time.time() - start_time))
    # vocab.getData()
