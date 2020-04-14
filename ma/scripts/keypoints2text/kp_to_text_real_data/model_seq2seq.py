"""
model_seq2seq.py: Implement the basic seq2seq model
"""

from __future__ import unicode_literals, division
import random
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.nn.functional as F

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
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embbed_dim = embbed_dim
        self.num_layers = num_layers

        # initialize every layer with the appropriate dimension. For the decoder layer, it will consist of an embedding, GRU, a Linear layer and a Log softmax activation function.
        self.embedding = nn.Embedding(self.output_dim, self.embbed_dim)  # TODO vocab size input for embedding + 1
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # reshape the input to (1, batch_size)
        input = input.view(1, -1)
        embedded = F.relu(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)
        prediction = self.softmax(self.out(output[0]))

        return prediction, hidden


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device, SOS_token, EOS_token):
        super().__init__()

        # initialize the encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token

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

        # use the encoder’s hidden layer as the decoder hidden (context vector)
        decoder_hidden = encoder_hidden.to(device)

        # add a token before the first predicted word
        decoder_input = torch.tensor([self.SOS_token], device=device)  # SOS

        # topk is used to get the top K value over a list
        # predict the output word from the current target word. If we enable the teaching force,
        # then the #next decoder input is the next word, else, use the decoder output highest value.
        for t in range(target_length):
            # print(decoder_input)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            topv, topi = decoder_output.topk(1)
            decoder_input = (target[t] if teacher_force else topi)
            if (teacher_force == False and decoder_input.item() == self.EOS_token):
                break
        return outputs
