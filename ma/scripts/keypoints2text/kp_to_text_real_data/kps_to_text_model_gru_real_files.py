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
import os
import time
from keypoints2text.kp_to_text_real_data.kps_to_text_dataset_real_files import TextKeypointsDataset
from keypoints2text.kp_to_text_real_data.kps_to_text_dataset_real_files import ToTensor
from keypoints2text.kp_to_text_real_data.model_seq2seq import Encoder
from keypoints2text.kp_to_text_real_data.model_seq2seq import Decoder
# from keypoints2text.kp_to_text_real_data.model_seq2seq_attention import AttnDecoderRNN
from keypoints2text.kp_to_text_real_data.model_seq2seq import Seq2Seq
from keypoints2text.kp_to_text_guru99.data_utils import DataUtils
import datetime
import nltk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RunModel:

    def __init__(self):
        # TODO create settings file
        # model settings
        self.teacher_forcing_ratio = 0.5
        self.embed_size = 256  # vocab list size
        # TODO different hidden size for encoder and decoder, now: same size for both
        self.hidden_size = 512
        self.num_layers = 1

        # train settings
        self.use_epochs = 1  # 0: time, 1: epochs
        self.num_iteration = 10
        self.hours = 0
        self.minutes = 30

        # eval settings
        self.num_iteration_eval = 20


        # variable setting
        # TODO set "final" _tokens when not changing implementation of vocabs anymore
        self.UNK_token = 1
        self.SOS_token = 2
        self.EOS_token = 3
        self.path_to_numpy_file = r"C:\Users\Asdf\Downloads\How2Sign_samples\all_files_normalized.npy"
        self.path_to_csv = r"C:\Users\Asdf\Downloads\How2Sign_samples\text\3_linked_to_npy\how2sign.test.id_transformed.txt_2npy.txt"
        self.path_to_vocab_file = r"C:\Users\Asdf\Downloads\How2Sign_samples\text_vocab\how2sign.test.id_vocab.txt"

        text2kp = TextKeypointsDataset(
            path_to_numpy_file=self.path_to_numpy_file,
            path_to_csv=self.path_to_csv,
            path_to_vocab_file=self.path_to_vocab_file,
            transform=ToTensor())
        self.keypoints_loader = torch.utils.data.DataLoader(text2kp, batch_size=1, shuffle=True, num_workers=0)

        # get max lengths
        # TODO skip too long data?
        # source_dim, target_dim = get_src_trgt_sizes()
        # print("source_dim: %d, target_dim: %d" % (source_dim, target_dim))
        # test set
        # source_dim_max:  291536
        # target_dim_max: 120

        count = 0
        with open(self.path_to_vocab_file, 'r') as f:
            for line in f:
                count += 1
        print("count: %d " %count)
        self.input_dim = 100000  # length of source keypoints TODO: get automatically
        self.output_dim = count + 1  # length of target

        self.model = self.init_model(self.input_dim, self.output_dim, self.hidden_size, self.embed_size,
                                     self.num_layers)

    def main(self):

        # if os.path.exists("model.pt"):
        #     self.model = torch.load("model.pt")

        print(self.model)
        self.train_helper(self.keypoints_loader, self.num_iteration)

        # torch.save(self.model, "model.pt")

        self.evaluate_model_own()

    def init_model(self, input_dim, output_dim, hidden_size, embed_size, num_layers):
        # create encoder-decoder model
        encoder = Encoder(input_dim, hidden_size, embed_size, num_layers)
        decoder = Decoder(output_dim, hidden_size, embed_size, num_layers)
        model = Seq2Seq(encoder, decoder, device, self.SOS_token, self.EOS_token).to(device)
        return model

    def train_helper(self, keypoints_loader, num_iteration):
        self.model.train()
        model_optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.L1Loss()
        total_loss_iterations = 0
        it = iter(keypoints_loader)

        # TODO add save/load here

        t_end = time.time() + 60 * self.minutes + 60 * 60 * self.hours

        # TODO shorten if/else
        if self.use_epochs:
            for idx in range(1, num_iteration + 1):
                try:
                    iterator_data = next(it)
                except StopIteration:  # reinitialize data loader if num_iteration > amount of data
                    it = iter(keypoints_loader)

                source_ten = torch.as_tensor(iterator_data[0], dtype=torch.float).view(-1, 1)
                target_ten = torch.as_tensor(iterator_data[1], dtype=torch.long).view(-1, 1)
                print("source_ten.size: %d, target_ten.size: %d" % (source_ten.size()[0], target_ten.size()[0]))

                loss = self.train_own(source_ten, target_ten, model_optimizer, criterion)
                total_loss_iterations += loss

                if idx % 1 == 0:
                    avarage_loss = total_loss_iterations / 1
                    total_loss_iterations = 0
                    print('Epoch %d, average loss: %.2f' % (idx, avarage_loss))
        else:
            idx_t = 0
            while time.time() < t_end:
                try:
                    iterator_data = next(it)
                except StopIteration:  # reinitialize data loader if num_iteration > amount of data
                    it = iter(keypoints_loader)

                source_ten = torch.as_tensor(iterator_data[0], dtype=torch.float).view(-1, 1)
                target_ten = torch.as_tensor(iterator_data[1], dtype=torch.long).view(-1, 1)
                print("source_ten.size: %d, target_ten.size: %d" % (source_ten.size()[0], target_ten.size()[0]))

                loss = self.train_own(source_ten, target_ten, model_optimizer, criterion)
                total_loss_iterations += loss

                if idx_t % 1 == 0:
                    avarage_loss = total_loss_iterations / 1
                    total_loss_iterations = 0
                    print('Epoch %d, average loss: %.2f' % (idx_t, avarage_loss))

                    print('Remaining time: %s' % str(datetime.timedelta(seconds=int(t_end - time.time()))))
                idx_t += 1



    # train
    def train_own(self, source_tensor, target_tensor, model_optimizer, criterion):
        model_optimizer.zero_grad()
        loss = 0.0
        output = self.model(source_tensor, target_tensor)
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

    def evaluate_model_own(self):
        # evaluate (kommt da was sinnvolles raus?)
        it = iter(self.keypoints_loader)

        for idx in range(1, self.num_iteration_eval + 1):
            try:
                iterator_data = next(it)
            except StopIteration:  # reinitialize data loader if num_iteration > amount of data
                it = iter(self.keypoints_loader)

            with torch.no_grad():
                in_ten = torch.as_tensor(iterator_data[0], dtype=torch.float).view(-1, 1)
                out_ten = torch.as_tensor(iterator_data[1], dtype=torch.long).view(-1, 1)
                print("---" * 10)

                flat_list = []  # sentence representation in int
                for sublist in out_ten.tolist():
                    for item in sublist:
                        flat_list.append(item)

                hypothesis = DataUtils().int2text(flat_list, DataUtils().vocab_int2word(self.path_to_vocab_file))
                hyp_str = " ".join(hypothesis)

                print("in_ten.size: %d, out_ten.size: %d" % (in_ten.size()[0], out_ten.size()[0]))
                # print("in_ten: %s, out_ten: %s" % (str(in_ten), str(out_ten)))
                decoded_words = []

                output = self.model(in_ten, out_ten)
                # print("output.size: %d" % output.size(0))
                # print(output)
                print("---" * 10)
                for ot in range(output.size(0)):
                    topv, topi = output[ot].topk(1)

                    if topi[0].item() == self.EOS_token:
                        print("FOUND EOS STOP")
                        decoded_words.append('<EOS>')
                        break
                    else:
                        decoded_words.append(topi[0].item())

                reference = DataUtils().int2text(decoded_words, DataUtils().vocab_int2word(self.path_to_vocab_file))
                ref_str = " ".join(reference)

                print("Hyp: %s" % hyp_str)
                print("Ref: %s" % ref_str)

                if len(hypothesis) >= 4 and len(reference) >= 4:
                    # there may be several references
                    bleu_score = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
                    print("BLEU score: %d" % bleu_score)


if __name__ == '__main__':
    runny = RunModel()
    runny.main()
