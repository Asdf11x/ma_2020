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
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class AttnEncoder(nn.Module):
    def __init__(self, hidden_dim, embbed_dim, num_layers):
        super(AttnEncoder, self).__init__()

        # set the encoder input dimesion , embbed dimesion, hidden dimesion, and number of layers
        # self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 274 fixed input dim, since currently are used 274 keypoints
        self.gru = nn.GRU(274, self.hidden_dim)

    def forward(self, input, hidden):
        input = input.view(1, 1, -1)
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_dim, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_dim, dropout_p=0.1, max_length=350):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_dim, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # print("///" * 10)
        # print(embedded[0].size())
        # print(hidden[0].size())
        # print(hidden.size())
        # print(hidden)

        # print(embedded)
        # print(hidden)
        # print(torch.cat((embedded[0], hidden[0]), 1).size())
        # print(torch.cat((embedded[0], hidden[0])))
        # print(self.attn(torch.cat((embedded[0], hidden[0]),1)))

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1).view(1,-1)), dim=1)
        # print(attn_weights)
        # print(attn_weights.unsqueeze(0).size())
        # print(encoder_outputs.unsqueeze(0).size())

        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnSeq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device, SOS_token, EOS_token):
        super().__init__()

        # initialize the encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token

    def forward(self, source_tensor, target_tensor, teacher_forcing_ratio=0.5, max_length=350):

        input_length = source_tensor.size(0)
        target_length = target_tensor.size(0)
        batch_size = target_tensor.shape[1]
        vocab_size = self.decoder.output_dim

        encoder_outputs = torch.zeros(max_length, self.encoder.hidden_dim, device=device)
        encoder_hidden = self.encoder.initHidden()
        # encoder_outputs = torch.zeros(100000, self.encoder.hidden_dim, device=device)
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

        loss = 0
        # print("input length: %d" % input_length)
        # print("encoder outputs size: %s" % str(encoder_outputs.size()))
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(source_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[self.SOS_token]], device=device)

        # print("encoder hidden size")
        # print(encoder_hidden.size())
        #
        # print("encoder output size")
        # print(encoder_outputs.size())

        decoder_hidden = encoder_hidden[-1].view(1, 1, -1)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden,
                                                                                 encoder_outputs)
                outputs[di] = decoder_output
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden,
                                                                                 encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                outputs[di] = decoder_output
                if decoder_input.item() == self.EOS_token:
                    break
        return outputs
