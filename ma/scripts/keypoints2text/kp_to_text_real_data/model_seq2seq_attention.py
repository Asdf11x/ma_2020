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

    # def forward(self, src):
    #     outputs, hidden = self.gru(src.view(1, 1, -1))
    #     return outputs, hidden  # outputs = hidden

    def forward(self, input):
        output, hidden = self.gru(input.view(1, 1, -1))
        return output, hidden

    # def initHidden(self):
    #     return torch.zeros(1, 1, self.hidden_dim, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=50):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


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

        input_length = source.size(0)
        target_length = target.size(0)
        batch_size = target.shape[1]
        vocab_size = 3600

        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

        # encoder_hidden = self.encoder.initHidden()
        # encoder_outputs = torch.zeros(100000, self.encoder.hidden_dim, device=device)

        loss = 0
        print(input_length)

        for i in range(input_length):
            # encoder_output = encoder_hidden = Encoder.forward.outputs/hidden
            # .size() => (hidden_size = 512) => [1, 1, 512]
            encoder_output, encoder_hidden = self.encoder(source[i])

        decoder_input = torch.tensor([[self.SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_output)
                outputs[di] = decoder_output
                decoder_input = target[di]  # Teacher forcing
        return outputs
