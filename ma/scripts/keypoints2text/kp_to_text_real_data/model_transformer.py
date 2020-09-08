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
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm
import torch.nn.functional
import torch.nn.functional as F

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
        self.ninp = ninp
        # self.encoder = nn.Embedding(intoken, nhid)
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        self.decoder = nn.Embedding(ntoken, self.ninp)
        self.pos_decoder = PositionalEncoding(self.ninp, dropout)

        # self.inscale = math.sqrt(intoken)
        # self.outscale = math.sqrt(ntoken)
        self.transformer = nn.Transformer(d_model=256, nhead=nhead, num_encoder_layers=nlayers,
                                          num_decoder_layers=nlayers, dim_feedforward=256, dropout=dropout,
                                          activation='relu')
        encoder_layer = TransformerEncoderLayer(self.ninp, nhead, nhid, dropout)
        encoder_norm = LayerNorm(self.ninp)
        self.encoder = TransformerEncoder(encoder_layer, nlayers, encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=nhead).to(device)
        decoder_norm = LayerNorm(self.ninp)
        self.decoder_full = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=nlayers, norm=decoder_norm).to(device)
        self.decoder_emb = nn.Embedding(ntoken, self.ninp)

        self.fc_out = nn.Linear(ninp, ntoken)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)
        # return (inp == 0).unsqueeze(-2)

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

        padding_tensor = src.mean(2)
        src_pad_mask = self.make_len_mask(padding_tensor)
        src_pad_mask = ~src_pad_mask
        # src_pad_mask = src_pad_mask.permute(1, 0)

        trg_pad_mask = self.make_len_mask(trg)

        # src = self.encoder(src)
        src = self.pos_encoder(src)

        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)

        # with open('log_batches.txt', 'a') as f:
        #     f.write("###" * 20 + "\n")
        #     f.write("src_pad_mask\n")
        #     f.write(str(src_pad_mask))
        #     f.write("\n")

        ## FUL TRANSFORMER
        output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask, memory_mask=self.memory_mask,
                                  src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask,
                                  memory_key_padding_mask=src_pad_mask)

        # Running without masks
        # output = self.transformer(src, trg, tgt_mask=self.trg_mask)


        ## TRANSFORMER ENCODER
        # output = self.encoder(src, mask=self.src_mask)

        # for i in range(tgt_seq_len):
        #     decoder_input = self.decoder_emb(trg[:, :i + 1]).transpose(0, 1)
        #     decoder_input = self.pos_encoder(decoder_input)
        #     tgt_mask = transformer_model.generate_square_subsequent_mask(i + 1).to(device)
        #     decoder_output = transformer_model.decoder(
        #         tgt=decoder_input,
        #         memory=encoder_hidden_states,
        #         tgt_mask=tgt_mask,
        #         memory_key_padding_mask=src_key_padding_mask)
        #     decoder_output = self.generator(decoder_output)[-1]
        #     decoder_inputs[:, i + 1] = decoder_output.max(1)[1]
        # return decoder_inputs


        ### RUNNING FULL TRANSFORMER
        # if 1:
        #     tgt_emb = self.decoder_emb(trg).transpose(0, 1)
        #     tgt_mask = torch.nn.Transformer().generate_square_subsequent_mask(tgt_emb.size(0)).to(device).transpose(0,1)
        #     decoder_output = self.decoder_full(tgt=tgt_emb,
        #                                   tgt_mask=tgt_mask,
        #                                   memory=output,
        #                                   memory_key_padding_mask=src_pad_mask)
        #     return self.fc_out(decoder_output)



        # for t in range(1, 50):
        #     tgt_emb = self.decoder_emb(output[:, :t]).transpose(0, 1)
        #     tgt_mask = torch.nn.Transformer().generate_square_subsequent_mask(t).to(device).transpose(0, 1)
        #     decoder_output = self.decoder_full(tgt=tgt_emb,
        #                              memory=output,
        #                              tgt_mask=tgt_mask)
        #
        #     pred_proba_t = self.fc_out(decoder_output)[-1, :, :]
        #     output_t = pred_proba_t.data.topk(1)[1].squeeze()
        #     output[:, t] = output_t

        output = self.fc_out(output)

        return output

        # return F.log_softmax(output, dim=-1)

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
