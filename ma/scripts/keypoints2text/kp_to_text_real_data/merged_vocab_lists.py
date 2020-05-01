"""
merged_vocab_lists.py: merge vocab lists containing unique words

features:
    - take all vocab Lists from a directory and add a merge vocab list
        - files need to be cleaned:
            - one word per line
    - resulting file contains unique words from each file
"""

import os
import sys
import time
from pathlib import Path


class VocabUtils:

    def __init__(self, path_to_sentences, path_to_target_dir):
        self.path_to_files = Path(path_to_sentences)
        self.path_to_target_dir = path_to_target_dir
        self.path_to_target_file = ""

    def main(self):
        self.create_folders()
        self.getData()

    def getData(self):
        unique_words = set()

        vocab_files = [pos_json for pos_json in os.listdir(self.path_to_files) if pos_json.endswith('.txt')]
        unique_words = set()

        for file in vocab_files:
            with open(self.path_to_files / file, encoding='ISO-8859-1', errors=None) as f:
                for line in f:
                    unique_words.add(line.lower().strip())

        sorted_words = sorted(unique_words)

        # make sure the beginning is correct
        sorted_words.remove("<pad>")
        sorted_words.remove("<unk>")
        sorted_words.remove("<sos>")
        sorted_words.remove("<eos>")
        sorted_words.remove(".")
        sorted_words.insert(0, "<pad>")
        sorted_words.insert(1, "<unk>")
        sorted_words.insert(2, "<sos>")
        sorted_words.insert(3, "<eos>")
        sorted_words.insert(4, ".")

        print("Unique words (incl. <unk>, <sos>, <eos>, . ): %d" % len(sorted_words))

        with open(self.path_to_target_file, 'w') as f:
            for item in sorted_words:
                f.write("%s\n" % item)

    def create_folders(self):
        """
        Create folders and filenames
        """
        # if no target dir is set, create file in the directory where the vocab list files are
        if self.path_to_target_dir == "":
            data_dir_target = self.path_to_files
        else:
            data_dir_target = Path(self.path_to_target_dir)

        # create new target directory, the files will be saved there
        if not os.path.exists(data_dir_target):
            os.makedirs(data_dir_target)

        # vocab file is in target dir "_vocab.txt" attached to the original name
        self.path_to_target_file = data_dir_target / "vocab_merged.txt"


if __name__ == '__main__':
    # file with sentences
    if len(sys.argv) > 1:
        path_to_sentences = sys.argv[1]
    else:
        print("Set path to folder")
        sys.exit()

    # target directory
    path_to_target_dir = ""
    if len(sys.argv) > 2:
        path_to_target_dir = sys.argv[2]
    start_time = time.time()
    vocab = VocabUtils(path_to_sentences, path_to_target_dir)
    vocab.main()
    print("--- %.4s seconds ---" % (time.time() - start_time))
