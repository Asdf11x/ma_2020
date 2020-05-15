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


class AttnEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, bi_encoder, batch_size):
        super(AttnEncoder, self).__init__()

        # set the encoder input dimesion , embbed dimesion, hidden dimesion, and number of layers
        # self.input_dim = input_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bi_encoder = True if bi_encoder else False
        self.batch_size = batch_size

        # 274 input dim, since currently are used 274 keypoints
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=self.bi_encoder)
        self.fc = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

    def forward(self, input):
        input = input.view(-1, self.batch_size, 274)
        output, hidden = self.gru(input)
        if self.bi_encoder:
            # output = (output[:, :, :self.hidden_dim] + output[:, :, self.hidden_dim:])
            hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return output, hidden


class AttnDecoderRNN(nn.Module):
    def __init__(self, output_dim, hidden_size, num_layers, dropout_p, max_length, bi_encoder, batch_size):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_layers = num_layers
        self.bi_encoder = bi_encoder
        self.batch_size = batch_size

        self.embedding = nn.Embedding(self.output_dim, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=self.num_layers * (self.bi_encoder + 1))
        self.out = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, self.batch_size, -1)
        embedded = self.dropout(embedded)
        hidden = hidden.view(1, self.batch_size, -1)

        # print("self.output_dim: %s " % str(self.output_dim))
        # print("self.hidden_size: %s " % str(self.hidden_size))
        # print("input: %s " % str(input))
        # print("input.size(): %s " % str(input.size()))
        # print("embedding_size: %s " % str(embedded.size()))
        # print("hidden_size: %s " % str(hidden.size()))
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1).view(1, -1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights


class AttnSeq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device, teacher_forcing, max_length, batch_size, SOS_token, EOS_token):
        super().__init__()

        # initialize the encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing = teacher_forcing
        self.max_length = max_length
        self.batch_size = batch_size
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token

    def forward(self, source_tensor, target_tensor):

        input_length = source_tensor.size(0)
        target_length = target_tensor.size(0)
        batch_size = target_tensor.shape[1]
        vocab_size = self.decoder.output_dim

        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_dim*2, device=device)
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

        # for ei in range(input_length):
        #     # sum() of tensor is only 0 when padding element,
        #     if torch.sum(source_tensor[ei]) == 0:
        #         break
        #     encoder_output, encoder_hidden = self.encoder(source_tensor[ei])
        #     # print(encoder_output.size())
        #     # print("___" * 10)
        #     encoder_outputs[ei] = encoder_output[0, 0]
        encoder_outputs, encoder_hidden = self.encoder(source_tensor)
        decoder_input = torch.tensor([[self.SOS_token] * self.batch_size], device=device)
        decoder_hidden = encoder_hidden


        # for t in range(target_length):
        #     decoder_output, decoder_hidden,decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
        #     outputs[t] = decoder_output
        #     teacher_force = random.random() < self.teacher_forcing
        #     topv, topi = decoder_output.topk(1)
        #     decoder_input = (target_tensor[t] if teacher_force else topi)
        #     if (teacher_force == False and decoder_input.item() == self.EOS_token):
        #         break
        use_teacher_forcing = True if random.random() < self.teacher_forcing else False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            # print("tf")
            for di in range(target_length):
                # print(target_tensor)
                # print(target_tensor[di])
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden,
                                                                                 encoder_outputs)
                outputs[di] = decoder_output
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # print("no tf")
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                # print(target_tensor)
                # print(target_tensor[di])
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # print("decoder_output.size(): %s " % str(decoder_output.size()))
                topv, topi = decoder_output.data.topk(1)
                # eg for batch_size = 2, but doesnt make sense to just double the results
                # decoder_input = torch.tensor([topi.squeeze().detach(), topi.squeeze().detach()])   # detach from history as input
                decoder_input = topi.squeeze().detach()
                outputs[di] = decoder_output
                if decoder_input.item() == self.EOS_token:
                    break
        return outputs
