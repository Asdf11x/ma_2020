"""
model_seq2seq.py: Implement the basic transformer model
from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
here: 20-02-26_data_loader\test_2_online\transformer_tutorial.py

"""

from __future__ import unicode_literals, division
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        super(TransformerModel, self).__init__()

        ################ OLD

        # from torch.nn import TransformerEncoder, TransformerEncoderLayer
        # self.model_type = 'Transformer'
        # self.src_mask = None
        # self.pos_encoder = PositionalEncoding(ninp, dropout)
        # encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, ninp)
        # self.ninp = ninp
        # self.decoder = nn.Linear(ninp, ntoken)
        # self.init_weights()

        ########### NEW

        # nhead = nhid // 64

        # self.encoder = nn.Embedding(intoken, nhid)
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        self.decoder = nn.Embedding(ntoken, 256)
        self.pos_decoder = PositionalEncoding(256, dropout)

        # self.inscale = math.sqrt(intoken)
        # self.outscale = math.sqrt(ntoken)
        import torch.nn.functional
        self.transformer = nn.Transformer(d_model=256, nhead=nhead, num_encoder_layers=nlayers,
                                          num_decoder_layers=nlayers, dim_feedforward=256, dropout=dropout,
                                          activation='relu')
        self.fc_out = nn.Linear(nhid, ntoken)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, trg):

        ############### OLD
        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #     device = src.device
        #     mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #     self.src_mask = mask
        # # print(src.size())
        # # src = self.encoder(src) * math.sqrt(self.ninp)
        # # print(src.size())
        # #
        # src = self.pos_encoder(src)
        # # print(src.size())
        #
        # output = self.transformer_encoder(src, self.src_mask)
        # output = self.decoder(output)
        # return output


        ############### NEW

        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)

        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #     self.src_mask = self.generate_square_subsequent_mask(len(src)).to(src.device)

        src_pad_mask = self.make_len_mask(src)
        trg_pad_mask = self.make_len_mask(trg)

        # src = self.encoder(src)
        src = self.pos_encoder(src)

        trg = self.decoder(trg)
        # trg_pad_mask = self.make_len_mask(trg)
        trg = self.pos_decoder(trg)
        #
        # with open('log.txt', 'a') as f:
        #     f.write("\n\nmodel_trans.py:")
        #     f.write("\n")
        #
        #     f.write(str(src.size()))
        #     f.write("\n")
        #     f.write("src_pad_mask: ")
        #     f.write(str(src_pad_mask.shape))
        #     f.write("\n")
        #
        #     f.write(str(trg.size()))
        #     f.write("\n")
        #     f.write("trg_pad_mask: ")
        #     f.write(str(trg_pad_mask.shape))
        #     f.write("\n")

        # output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask, memory_mask=self.memory_mask,
        #                           src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask,
        #                           memory_key_padding_mask=src_pad_mask)

        output = self.transformer(src, trg, tgt_mask=self.trg_mask)

        output = self.fc_out(output)

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
