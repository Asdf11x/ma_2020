"""
vocab_utils.py: create a vocab list for a text file

usage:
vocab_utils.py path_to_text_file transform_file path_to_target_dir
example
vocab_utils.py ~/workspace/how2sign.train.id.en 0 .


features:
    - read text from file
        - each line contains: ID Sentence
        - ID is first element in each sentence, followed by a whitespace and then the actual sentence starts
    - build vocab file, containg all vocabs used in the sentences.csv file
    - Vocab list DOES contain: 3 special tokens in the beginning & only words
        - words like won't / don't / he's will be split up into: will not / do not / he is and added seperately to the
        vocab list
    - Vocab list does NOT contain: special characters & numbers
    - standard feature: create vocab list
    - additional feature: transform text file to lowercase / no non-character words / dont't -> do not
        - set transform_file=1 in start parameters
"""

import os
import re
import sys
import time
from pathlib import Path
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_suffix_regex


class VocabUtils:

    def __init__(self, path_to_sentences, path_to_target_dir, transform_file):
        self.path_to_sentences = Path(path_to_sentences)
        self.path_to_target_dir = path_to_target_dir
        self.path_to_vocab_file = ""
        self.path_to_trans_file = ""
        self.transform_file = transform_file
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.tokenizer = self.custom_tokenizer(self.nlp)
        self.contractions = {
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

    def main(self):
        self.create_folders()
        self.create_vocab_file()

        if self.transform_file:
            self.transform_sentence_file()

    def custom_tokenizer(self, nlp):
        """
        Custom tokenizer:
            - won't -> won't
        before:
            - won't -> wo + n't

        Afterwards all words like won't are compared to a dictionary containing all possibilities.
        the option before (won't -> wo + n't) cannot detect if "wo" was an actual word or a part from won't.
        :param nlp:
        :return:
        """
        infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"~]''')
        prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
        suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

        return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                         suffix_search=suffix_re.search,
                         infix_finditer=infix_re.finditer,
                         token_match=None)

    def transform_sentence_file(self):
        """
        transform text file to lowercase / no non-character words / dont't -> do not
        :return:
        """
        all_sentences = []

        with open(self.path_to_sentences, encoding='utf-8') as f:
            for line in f:
                # tokenize
                doc = self.nlp(line)
                sentence_tok = []

                # check if first character is "_", if so build first char manually
                if doc[0].text == "_":
                    sentence_tok.append("_" + doc[1].text)
                    for idx in range(2, len(doc)):
                        sentence_tok.append(doc[idx].text.lower())
                else:
                    sentence_tok.append(doc[0].text)
                    for idx in range(1, len(doc)):
                        sentence_tok.append(doc[idx].text.lower())

                sentence_trans = [str(sentence_tok[0])]

                # clean tokenize
                for word in sentence_tok[1:]:
                    # check if word is contracted (e.g. won't, don't, ...)
                    if word in self.contractions:
                        # if is contracted, expand (e.g. won't -> will not) and add both words to the set
                        for element in self.contractions[word].split():
                            sentence_trans.append(element.lower())
                        continue
                    # check if word does contain only alphanummeric
                    if word.isalpha():
                        sentence_trans.append(word)

                all_sentences.append(sentence_trans)

        with open(self.path_to_trans_file, 'w') as f:
            f.write('\n'.join([' '.join(i) for i in all_sentences]))
        print("Transformed file.")

    def create_vocab_file(self):
        """
        create vocab list from a file
            - excluding first element, since its the video ID
        :return:
        """
        unique_words = set()

        with open(self.path_to_sentences, encoding='utf-8') as f:
            for line in f:
                # tokenize
                doc = self.nlp(line)
                sentence_tok = [token.text for token in doc]
                # print(line)
                # print(sentence_tok)
                for token in sentence_tok[1:]:
                    unique_words.add(token.lower())

            # clean tokenize
            unique_copy = unique_words.copy()
            for word in unique_words:
                # check if word is contracted (e.g. won't, don't, ...)
                if word in self.contractions:
                    unique_copy.remove(word)
                    # if is contracted, expand (e.g. won't -> will not) and add both words to the set
                    for element in self.contractions[word].split():
                        unique_copy.add(element.lower())
                    continue
                # check if word does contain only alphanummeric
                if not word.isalpha():
                    unique_copy.remove(word)

            unique_words = unique_copy.copy()
            sorted_words = sorted(unique_words)

            # add special tokens to the beginning
            sorted_words.insert(0, "<pad>")
            sorted_words.insert(1, "<unk>")
            sorted_words.insert(2, "<sos>")
            sorted_words.insert(3, "<eos>")
            sorted_words.insert(4, ".")

            print("Unique words (incl. UNK/EOS/SOS ): %d" % len(sorted_words))

        with open(self.path_to_vocab_file, 'w') as f:
            for item in sorted_words:
                f.write("%s\n" % item)
        print("Created vocab file.")

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
        self.path_to_vocab_file = data_dir_target / (str(self.path_to_sentences.stem) + str("_vocab.txt"))

        # transformed file is in target dir "_transformed.txt" attached to the original name
        self.path_to_trans_file = data_dir_target / (str(self.path_to_sentences.stem) + str("_transformed.txt"))


if __name__ == '__main__':
    # file with sentences
    if len(sys.argv) > 1:
        path_to_sentences = sys.argv[1]
    else:
        print("Set path to file containing sentences")
        sys.exit()

    # transform file?
    # 0:no, 1:yes
    transform_file = 1
    if len(sys.argv) > 2:
        transform_file = sys.argv[2]

    # target directory
    path_to_target_dir = ""
    if len(sys.argv) > 3:
        path_to_target_dir = sys.argv[3]


    start_time = time.time()
    vocab = VocabUtils(path_to_sentences, path_to_target_dir, transform_file)
    vocab.main()
    print("--- %.4s seconds ---" % (time.time() - start_time))
