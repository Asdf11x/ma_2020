"""
data_utils.py: script for data processing during runtime
"""

class DataUtils:

    def vocab_word2int(self, path_to_vocab_file):
        """
        Create a word2int dictionary from a vocab file
        e.g. print: {'who': 0}
        :param path_to_vocab_file:
        :return:
        """
        word2int = {}
        indx = 0
        with open(path_to_vocab_file) as f:
            for line in f:
                word2int[line.strip()] = indx
                indx += 1
        return word2int

    def vocab_int2word(self, path_to_vocab_file):
        """
        Transform word2int dictionary into an int2word dictionary
        e.g. print: {'0': word}
        :param path_to_vocab_file:
        :return:
        """
        word2int = self.vocab_word2int(path_to_vocab_file)
        int2word = {v: k for k, v in word2int.items()}
        return int2word

    def int2text(self, indices, int2word):
        """
        Transform a list of indices according to a lookup dictionary
        :param indices: List of indices representing words, according to a vocab file
        :param int2word: An int2word dictionary
        :return: list of words (corresponding to the indices)
        """
        result = []
        for element in indices:
            if element in int2word:
                result.append(int2word[element])
            else:
                result.append("<unk>")
        return result

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

    def get_file_length(self, path_to_vocab_file):
        """
        Get file length of a vocab file
        -- !! Assuming each line contains ONE SINGLE UNIQUE WORD !! --
        :param path_to_vocab_file:
        :return: Amount of single unique words in a file
        """
        count = 0
        with open(path_to_vocab_file, 'r') as f:
            for line in f:
                count += 1
        return count
