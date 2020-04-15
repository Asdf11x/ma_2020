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
from keypoints2text.kp_to_text_real_data.data_loader import TextKeypointsDataset
from keypoints2text.kp_to_text_real_data.data_loader import ToTensor
from keypoints2text.kp_to_text_real_data.model_seq2seq import Encoder
from keypoints2text.kp_to_text_real_data.model_seq2seq import Decoder
# from keypoints2text.kp_to_text_real_data.model_seq2seq_attention import AttnDecoderRNN
from keypoints2text.kp_to_text_real_data.model_seq2seq import Seq2Seq
from keypoints2text.kp_to_text_guru99.data_utils import DataUtils
from keypoints2text.kp_to_text_real_data.run_model_helper import Helper, Save, Mode
import datetime
import nltk
from copy import deepcopy

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
        self.num_iteration = 15
        self.hours = 0
        self.minutes = 30
        self.show_after_epochs = 5

        # eval settings
        self.num_iteration_eval = 1

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

        # save / load
        self.save_model = 1
        self.save_model_file_path = r"C:\Eigene_Programme\Git-Data\Own_Repositories\ma_2020\ma\scripts\keypoints2text\kp_to_text_real_data\saved_models\2020-04-15_23-19\model.pt"  # if not empty use path, else create new folder, use only when documentation
        # exists
        self.save_model_folder_path = r"C:\Eigene_Programme\Git-Data\Own_Repositories\ma_2020\ma\scripts\keypoints2text\kp_to_text_real_data\saved_models"
        self.save_loss = []  # list to save loss results
        self.save_eval = []  # list to save evaluation results
        self.save_epoch = 5  # save each x epoch

        if self.save_model_file_path == "":
            self.save_state = Save.new
        else:
            self.save_state = Save.update

        self.documentation = {"epochs_total": 0,
                              "time_total_s": 0,
                              "time_total_readable": "",
                              "loss": [],
                              "loss_time_epoch": [],
                              "hypothesis": [],
                              "reference": [],
                              "BLEU": [],
                              }
        self.elapsed_time_sum = 0
        self.idx_save = 0
        self.load_dic = deepcopy(self.documentation)
        self.load_model = 0
        self.load_model_path = ""
        self.load_json_file_once = 1  # load json file once to get origin values
        # get max lengths
        # TODO skip too long data?
        # source_dim, target_dim = get_src_trgt_sizes()
        # print("source_dim: %d, target_dim: %d" % (source_dim, target_dim))
        # test set
        # source_dim_max: 291536
        # target_dim_max: 120
        count = 0
        with open(self.path_to_vocab_file, 'r') as f:
            for line in f:
                count += 1
        print("amount of unique words in vocab file: %d " % count)
        # TODO: get input_dim automatically?
        # TODO: crop max input_dim?
        self.input_dim = 100000  # length of source keypoints
        self.output_dim = count + 1  # length of target

        self.model = self.init_model(self.input_dim, self.output_dim, self.hidden_size, self.embed_size,
                                     self.num_layers)

    def main(self):
        if self.load_model:
            if os.path.exists(self.load_model_path):
                self.model = torch.load(self.load_model_path)

        print(self.model)
        self.train_run(self.keypoints_loader, self.num_iteration)

        # self.save_helper()

        self.evaluate_model_own()

    def init_model(self, input_dim, output_dim, hidden_size, embed_size, num_layers):
        # create encoder-decoder model
        encoder = Encoder(input_dim, hidden_size, embed_size, num_layers)
        decoder = Decoder(output_dim, hidden_size, embed_size, num_layers)
        model = Seq2Seq(encoder, decoder, device, self.SOS_token, self.EOS_token).to(device)
        return model

    def train_run(self, keypoints_loader, num_iteration):
        """
        the outer most train method, the whole train procesdure is started here
        :param keypoints_loader:
        :param num_iteration:
        :return:
        """
        self.model.train()
        model_optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.L1Loss()
        total_loss_iterations = 0
        it = iter(keypoints_loader)

        start_time = time.time()  # used for tracking time when saving, refreshed each run
        t_end = time.time() + 60 * self.minutes + 60 * 60 * self.hours  # remaining training time

        # TODO shorten if/else
        if self.use_epochs:
            for idx in range(1, num_iteration + 1):
                self.train_helper(criterion, idx, it, keypoints_loader, model_optimizer, total_loss_iterations,
                                  start_time)
                start_time = time.time()

        else:
            idx = 1
            while time.time() < t_end:
                self.train_helper(criterion, idx, it, keypoints_loader, model_optimizer, total_loss_iterations,
                                  start_time, t_end)
                start_time = time.time()
                idx += 1

    def train_helper(self, criterion, idx, it, keypoints_loader, model_optimizer, total_loss_iterations, start_time,
                     t_end=0.0):
        """
        main -> train_run -> train_helper -> train_model
        use this model to help with computing the loss and reduce train_run
        :param criterion:
        :param idx:
        :param it:
        :param keypoints_loader:
        :param model_optimizer:
        :param total_loss_iterations:
        :return:
        """
        try:
            iterator_data = next(it)
        except StopIteration:  # reinitialize data loader if num_iteration > amount of data
            it = iter(keypoints_loader)

        source_ten = torch.as_tensor(iterator_data[0], dtype=torch.float).view(-1, 1)  # TODO remove [:20]
        target_ten = torch.as_tensor(iterator_data[1], dtype=torch.long).view(-1, 1)
        print("source_ten.size: %d, target_ten.size: %d" % (source_ten.size()[0], target_ten.size()[0]))
        loss = self.train_model(source_ten, target_ten, model_optimizer, criterion)
        total_loss_iterations += loss

        if idx % self.show_after_epochs == 0:
            elapsed_time_s = int(time.time() - start_time)
            self.elapsed_time_sum += elapsed_time_s

            remaining_time = int(t_end - time.time())
            average_loss = total_loss_iterations / self.show_after_epochs
            total_loss_iterations = 0

            print('Epoch %d, average loss: %.2f, elapsed time: %s'
                  % (idx, average_loss, str(datetime.timedelta(seconds=self.elapsed_time_sum))))

            if t_end != 0.0:
                print('Remaining time: %s' % str(datetime.timedelta(seconds=remaining_time)))

        if idx % self.save_epoch == 0:
            elapsed_time_s = int(time.time() - start_time)

            print('Saving at epoch %d, average loss: %.2f' % (idx, average_loss))

            # refresh idx_t each time saving is callled
            idx_t = idx - self.idx_save

            self.documentation["epochs_total"] = idx_t
            self.documentation["time_total_s"] = elapsed_time_s
            self.documentation["loss"] = [round(average_loss, 2)]

            self.save_helper(self.save_state, Mode.train)
            self.idx_save = idx

    def train_model(self, source_tensor, target_tensor, model_optimizer, criterion):
        """
        the inner most method to train the model, the actual training is implemented here
        :param source_tensor:
        :param target_tensor:
        :param model_optimizer:
        :param criterion:
        :return:
        """

        model_optimizer.zero_grad()
        loss = 0.0
        output = self.model(source_tensor, target_tensor)
        num_iter = output.size(0)

        # calculate the loss from a predicted sentence with the expected result
        for ot in range(num_iter):  # creating user warning of tensor
            loss += criterion(output[ot], target_tensor[ot].view(1, -1))

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
                        decoded_words.append('<eos>')
                        break
                    else:
                        decoded_words.append(topi[0].item())

                reference = DataUtils().int2text(decoded_words, DataUtils().vocab_int2word(self.path_to_vocab_file))
                ref_str = " ".join(reference)

                if len(hypothesis) >= 4 or len(reference) >= 4:
                    # there may be several references
                    bleu_score = round(nltk.translate.bleu_score.sentence_bleu([reference], hypothesis),2)
                    print("BLEU score: %d" % bleu_score)
                    self.documentation["BLEU"].append(bleu_score)

                print("Hyp: %s" % hyp_str)
                print("Ref: %s" % ref_str)
                self.documentation["hypothesis"].append(hyp_str)
                self.documentation["reference"].append(ref_str)

                self.save_helper(self.save_state, Mode.eval)

    def save_helper(self, save, mode):

        if self.save_model:

            # save in new file and model
            if save == Save.new:
                self.save_model_file_path = Helper().save_model(self.model, self.save_model_folder_path,
                                                                self.save_model_file_path,
                                                                self.documentation, self.save_state, mode)
                self.save_state = Save.update

            # update file and model
            else:
                Helper().save_model(self.model, self.save_model_folder_path, self.save_model_file_path,
                                    self.documentation, self.save_state, mode)

            # reset variables
            self.documentation = {"epochs_total": 0,
                                  "time_total_s": 0,
                                  "time_total_readable": "",
                                  "loss": [],
                                  "loss_time_epoch": [],
                                  "hypothesis": [],
                                  "reference": [],
                                  "BLEU": [],
                                  }


if __name__ == '__main__':
    runny = RunModel()
    runny.main()
