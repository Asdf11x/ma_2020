"""
data_utils.py: script for data processing during runtime
"""

class DataUtils:

    def vocab_word2int(self, path_to_vocab_file):
        word2int = {}
        indx = 0
        with open(path_to_vocab_file) as f:
            for line in f:
                word2int[line.strip()] = indx
                indx += 1
        return word2int

    def vocab_int2word(self, path_to_vocab_file):
        word2int = {}
        indx = 0
        with open(path_to_vocab_file) as f:
            for line in f:
                word2int[line.strip()] = indx
                indx += 1
        int2word = {v: k for k, v in word2int.items()}
        return int2word

    def int2text(self, indices, int2word):
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