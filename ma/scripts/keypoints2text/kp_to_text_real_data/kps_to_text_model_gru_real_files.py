"""
gru99_model.py:
https://www.guru99.com/seq2seq-model.html

01-04-20:
features /todos:
    - kps to text model
    - remove embedding from input
    - decoder is the same as original gru99_model.py
    - reverse from source to target

"""

from __future__ import unicode_literals, division
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
import torch.nn.functional as F

from keypoints2text.kp_to_text_real_data.kps_to_text_dataset_real_files import TextKeypointsDataset
from keypoints2text.kp_to_text_real_data.kps_to_text_dataset_real_files import ToTensor
from keypoints2text.kp_to_text_guru99.data_utils import DataUtils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embbed_dim, num_layers):
        super(Encoder, self).__init__()

        # set the encoder input dimesion , embbed dimesion, hidden dimesion, and number of layers
        self.input_dim = input_dim
        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # initialize the embedding layer with input and embbed dimention

        # self.embedding = nn.Embedding(input_dim + 1, self.embbed_dim)

        # intialize the GRU to take the input dimetion of embbed, and output dimention of hidden and
        # set the number of gru layers
        self.gru = nn.GRU(1, self.hidden_dim, num_layers=self.num_layers)

    def forward(self, src):
        outputs, hidden = self.gru(src.view(1, 1, -1))
        return outputs, hidden  # outputs = hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, embbed_dim, num_layers):
        super(Decoder, self).__init__()

        # set the encoder output dimension, embed dimension, hidden dimension, and number of layers
        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # initialize every layer with the appropriate dimension. For the decoder layer, it will consist of an embedding, GRU, a Linear layer and a Log softmax activation function.
        self.embedding = nn.Embedding(output_dim, self.embbed_dim)
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)
        self.out = nn.Linear(self.hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # reshape the input to (1, batch_size)
        input = input.view(1, -1)
        embedded = F.relu(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)
        prediction = self.softmax(self.out(output[0]))

        return prediction, hidden


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super().__init__()

        # initialize the encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):

        input_length = source.size(0)  # get the input length (number of words in sentence)
        batch_size = target.shape[1]
        target_length = target.shape[0]
        vocab_size = self.decoder.output_dim

        # initialize a variable to hold the predicted outputs
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

        # encode every word in a sentence
        for i in range(input_length):
            # encoder_output = encoder_hidden = Encoder.forward.outputs/hidden
            # .size() => (hidden_size = 512) => [1, 1, 512]
            encoder_output, encoder_hidden = self.encoder(source[i])

        # use the encoder’s hidden layer as the decoder hidden
        decoder_hidden = encoder_hidden.to(device)

        # add a token before the first predicted word
        decoder_input = torch.tensor([SOS_token], device=device)  # SOS

        # topk is used to get the top K value over a list
        # predict the output word from the current target word. If we enable the teaching force,
        # then the #next decoder input is the next word, else, use the decoder output highest value.
        for t in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            topv, topi = decoder_output.topk(1)
            input = (target[t] if teacher_force else topi)
            if (teacher_force == False and input.item() == EOS_token):
                break

        return outputs


class RunModel:

    def init_model(self, input_dim, output_dim, hidden_size, embed_size, num_layers):
        # create encoder-decoder model
        encoder = Encoder(input_dim, hidden_size, embed_size, num_layers)
        decoder = Decoder(output_dim, hidden_size, embed_size, num_layers)
        model = Seq2Seq(encoder, decoder, device).to(device)
        return model

    def train_helper(self, model, keypoints_loader, num_iteration):
        model.train()
        model_optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.L1Loss()
        total_loss_iterations = 0
        it = iter(keypoints_loader)

        for idx in range(1, num_iteration + 1):
            try:
                iterator_data = next(it)
            except StopIteration:  # reinitialize data loader if num_iteration > amount of data
                it = iter(keypoints_loader)

            source_ten = torch.as_tensor(iterator_data[0], dtype=torch.float).view(-1,
                                                                                   1)  # [:20] TODO: remove (20 for testing)
            target_ten = torch.as_tensor(iterator_data[1], dtype=torch.long).view(-1, 1)
            print("source_ten.size: %d, target_ten.size: %d" % (source_ten.size()[0], target_ten.size()[0]))

            loss = run_model_own.train_own(model, source_ten, target_ten, model_optimizer, criterion)
            total_loss_iterations += loss

            if idx % 1 == 0:
                avarage_loss = total_loss_iterations / 1
                total_loss_iterations = 0
                print('Epoch %d, average loss: %.2f' % (idx, avarage_loss))

    # train
    def train_own(self, model, source_tensor, target_tensor, model_optimizer, criterion):
        model_optimizer.zero_grad()
        loss = 0.0
        output = model(source_tensor, target_tensor)
        num_iter = output.size(0)

        # calculate the loss from a predicted sentence with the expected result
        for ot in range(num_iter):
            # print("output %s, target %s" % (str(output.size()), str(target_tensor.size())))
            # print("output %s, target %s" % (str(output[ot].size()), str(target_tensor[ot].size())))
            # print(output)
            loss += criterion(output[ot], target_tensor[ot])
        loss.backward()
        model_optimizer.step()
        epoch_loss = loss.item() / num_iter

        # print(epoch_loss)
        return epoch_loss

    def evaluate_model_own(self, text2kp):
        # evaluate (kommt da was sinnvolles raus?)
        it = iter(keypoints_loader)

        for idx in range(1, num_iteration_eval + 1):
            try:
                iterator_data = next(it)
            except StopIteration:  # reinitialize data loader if num_iteration > amount of data
                it = iter(keypoints_loader)

            with torch.no_grad():
                in_ten = torch.as_tensor(iterator_data[0], dtype=torch.float).view(-1, 1)
                out_ten = torch.as_tensor(iterator_data[1], dtype=torch.long).view(-1, 1)
                print("---" * 10)

                flat_list = []
                for sublist in out_ten.tolist():
                    for item in sublist:
                        flat_list.append(item)

                print(flat_list)
                print(DataUtils().int2text(flat_list, text2kp.int2word))
                print("in_ten.size: %d, out_ten.size: %d" % (in_ten.size()[0], out_ten.size()[0]))
                # print("in_ten: %s, out_ten: %s" % (str(in_ten), str(out_ten)))
                decoded_words = []

                output = model(in_ten, out_ten)
                # print("output.size: %d" % output.size(0))
                # print(output)
                print("---" * 10)
                for ot in range(output.size(0)):
                    topv, topi = output[ot].topk(1)

                    # print(topv)
                    # print(topi)
                    # print(topi[0].item())
                    if topi[0].item() == EOS_token:
                        print("HALT STOP JETZT REDE ICH")
                        decoded_words.append('<EOS>')
                        break
                    else:
                        decoded_words.append(topi[0].item())

                print(decoded_words)
                print(DataUtils().int2text(decoded_words, text2kp.int2word))


def get_src_trgt_sizes():
    source_max = 0
    target_max = 0
    source_max_saved = []
    target_max_saved = []
    it = iter(keypoints_loader)

    while 1:
        try:
            iterator_data = next(it)
        except StopIteration:  # reinitialize data loader if num_iteration > amount of data
            return source_max, target_max
        source_len = torch.as_tensor(iterator_data[0], dtype=torch.float).view(-1, 1).size()[0]
        target_len = torch.as_tensor(iterator_data[1], dtype=torch.long).view(-1, 1).size()[0]

        source_max_saved.append(source_len)
        target_max_saved.append(target_len)

        if source_len > source_max:
            source_max = source_len

        if target_len > target_max:
            target_max = target_len


if __name__ == '__main__':
    teacher_forcing_ratio = 0.5
    embed_size = 16
    hidden_size = 512
    num_layers = 1
    num_iteration = 150
    num_iteration_eval = 20
    UNK_token = 1
    SOS_token = 2
    EOS_token = 3

    text2kp = TextKeypointsDataset(
        path_to_numpy_file=r"C:\Users\Asdf\Downloads\How2Sign_samples\all_files_normalized.npy",
        path_to_csv=r"C:\Users\Asdf\Downloads\How2Sign_samples\text\3_linked_to_npy\how2sign.test.id_transformed.txt_2npy.txt",
        path_to_vocab_file=r"C:\Users\Asdf\Downloads\How2Sign_samples\text_vocab\how2sign.test.id_vocab.txt",
        transform=ToTensor())
    keypoints_loader = torch.utils.data.DataLoader(text2kp, batch_size=1, shuffle=True, num_workers=0)

    # get max lengths
    # source_dim, target_dim = get_src_trgt_sizes()
    # print("source_dim: %d, target_dim: %d" % (source_dim, target_dim))
    # test set
    # source_dim_max:  291536
    # target_dim_max: 120

    # TODO skip too long data
    source_dim = 100000  # length of source keypoints TODO: get automatically
    target_dim = 50  # length of target

    # it = iter(keypoints_loader)
    run_model_own = RunModel()

    model = run_model_own.init_model(source_dim, target_dim, hidden_size, embed_size, num_layers)
    run_model_own.train_helper(model, keypoints_loader, num_iteration)

    run_model_own.evaluate_model_own(text2kp)
