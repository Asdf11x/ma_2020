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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
import os
import time
from keypoints2text.kp_to_text_real_data.other_input.data_loader_flat import TextKeypointsDataset, ToTensor
from keypoints2text.kp_to_text_real_data.model_seq2seq import Encoder, Decoder, Seq2Seq
from keypoints2text.kp_to_text_real_data.data_utils import DataUtils
from keypoints2text.kp_to_text_real_data.save_model import Helper, Save, Mode
import datetime
import nltk
from pathlib import Path
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RunModel:

    def __init__(self):

        # read settings file, didnt use configparser, because config parser saves everything as string
        with open("../hparams.json") as json_file:
            config = json.load(json_file)

        # model settings
        self.teacher_forcing_ratio = config["model_settings"]["teacher_forcing_ratio"]
        self.embed_size = config["model_settings"]["embed_size"]  # vocab list size
        self.hidden_size_enc = config["model_settings"]["hidden_size_enc"]
        self.hidden_size_dec = config["model_settings"]["hidden_size_dec"]
        self.num_layers = config["model_settings"]["num_layers"]

        # train settings
        self.use_epochs = config["train_settings"]["use_epochs"]  # 0: time, 1: epochs
        self.num_iteration = config["train_settings"]["num_iteration"]
        self.hours = config["train_settings"]["hours"]
        self.minutes = config["train_settings"]["minutes"]
        self.show_after_epochs = config["train_settings"]["show_after_epochs"]

        # eval settings
        self.evaluate_model = config["eval_settings"]["evaluate_model"]  # 0: model is not evaluated, 1: model is evaluated
        self.num_iteration_eval = config["eval_settings"]["num_iteration_eval"]

        # test settings
        self.test_model = config["test_settings"]["test_model"]  # 0: model is not tested, 1: model is tested

        # train
        self.path_to_numpy_file_train = Path(config["train_paths"]["path_to_numpy_file_train"])
        self.path_to_csv_train = Path(config["train_paths"]["path_to_csv_train"])
        self.path_to_vocab_file_train = Path(config["train_paths"]["path_to_vocab_file_train"])

        # val
        self.path_to_numpy_file_val = config["val_paths"]["path_to_numpy_file_val"]
        self.path_to_csv_val = config["val_paths"]["path_to_csv_val"]
        self.path_to_vocab_file_val = config["val_paths"]["path_to_vocab_file_val"]

        # test
        self.path_to_numpy_file_test = config["test_paths"]["path_to_numpy_file_test"]
        self.path_to_csv_test = config["test_paths"]["path_to_csv_test"]
        self.path_to_vocab_file_test = config["test_paths"]["path_to_vocab_file_test"]

        # vocab file, containing unique words for all (train, val & test)
        self.path_to_vocab_file_all = config["vocab_file"]["path_to_vocab_file_all"]



        # set tokens
        self.PAD_token = 0
        self.UNK_token = 1
        self.SOS_token = 2
        self.EOS_token = 3

        # save / load
        self.save_model = config["save_load"]["save_model"]  # 0: model is not saved, 1: model is saved
        self.save_model_file_path = config["save_load"]["save_model_file_path"]  # if not empty use path, else create new folder, use only when documentation exists
        self.save_model_folder_path = config["save_load"]["save_model_folder_path"]
        self.save_epoch = config["save_load"]["save_epoch"]  # save each x epoch
        self.save_loss = []  # init list to save loss results
        self.save_eval = []  # init list to save evaluation results
        # if no path to a model is set -> State.new, if a path ot a model exists, just keep updating
        if self.save_model_file_path == "":
            self.save_state = Save.new
        else:
            self.save_state = Save.update

        # use documentation dictionary for saving values to a txt file while training
        self.documentation = {"epochs_total": 0,
                              "time_total_s": 0,
                              "time_total_readable": "",
                              "loss": [],
                              "loss_time_epoch": [],
                              "hypothesis": [],
                              "reference": [],
                              "BLEU": [],
                              }

        self.elapsed_time_sum = 0.0  # init time
        self.idx_save = 0  # init saving epochs
        self.time_run = time.time()  # init, start taking time to show on print, init variable
        self.time_save = time.time()  # init, start taking time to show on save, init variable

        self.load_model = config["save_load"]["load_model"]
        self.load_model_path = config["save_load"]["load_model_path"]
        # get max lengths
        # TODO skip too long data?
        # source_dim, target_dim = get_src_trgt_sizes()
        # print("source_dim: %d, target_dim: %d" % (source_dim, target_dim))
        # test set
        # source_dim_max: 291536
        # target_dim_max: 120

        # TODO
        #  - Get max length of keypoints and text for train, val & test set
        #  - Pad keypoints to max keypoint length (x,y respectively)
        #  - Set encoder hidden size to max length of keypoints
        #  - Pad text to max text length
        #  - Set decoder hidden size to max text length
        #  - Use vocab file for all three
        #  - Set decoder output dim to vocab size length
        #  - This takes a lot of time, so do the steps above only if parameters in hparams file are missing!
        # TODO: get input_dim automatically?
        # TODO: crop max input_dim?
        # self.input_dim = config["padding"]["input_dim"]  # length of source keypoints
        self.output_dim = config["padding"]["output_dim"]  # output_dim != max_length. max_length == hidden_size

        # vocab size, amount of different unique words
        if self.output_dim == 0:
            self.output_dim = DataUtils().get_file_length(self.path_to_vocab_file_all)

        # max length of source keypoints and target sentence
        if self.hidden_size_enc == 0 or self.hidden_size_dec == 0:
            print("Searching for source and target max length")
            self.hidden_size_enc, self.hidden_size_dec, lengths = DataUtils().get_kp_text_max_lengths(self.data_loader_train, self.data_loader_train, self.data_loader_train)
            with open('../../tests_graphs/lengths.txt', 'w') as f:
                for item in lengths:
                    f.write("%s\n" % item)

        # Dataloaders for train, val & test
        text2kp_train = TextKeypointsDataset(
            path_to_numpy_file=self.path_to_numpy_file_train,
            path_to_csv=self.path_to_csv_train,
            path_to_vocab_file=self.path_to_vocab_file_train,
            transform=ToTensor())
        self.data_loader_train = torch.utils.data.DataLoader(text2kp_train, batch_size=1, shuffle=True,
                                                             num_workers=0)

        # text2kp_val = TextKeypointsDataset(
        #     path_to_numpy_file=self.path_to_numpy_file_val,
        #     path_to_csv=self.path_to_csv_val,
        #     path_to_vocab_file=self.path_to_vocab_file_val,
        #     transform=ToTensor(),
        #     kp_max_len=self.hidden_size_enc,
        #     text_max_len=self.hidden_size_dec)
        # self.data_loader_val = torch.utils.data.DataLoader(text2kp_val, batch_size=1, shuffle=True, num_workers=0)
        #
        # text2kp_test = TextKeypointsDataset(
        #     path_to_numpy_file=self.path_to_numpy_file_test,
        #     path_to_csv=self.path_to_csv_test,
        #     path_to_vocab_file=self.path_to_vocab_file_test,
        #     transform=ToTensor(),
        #     kp_max_len=self.hidden_size_enc,
        #     text_max_len=self.hidden_size_dec)
        # self.data_loader_test = torch.utils.data.DataLoader(text2kp_test, batch_size=1, shuffle=True, num_workers=0)

        self.model = self.init_model(self.output_dim, self.hidden_size_enc, self.hidden_size_dec, self.embed_size,
                                     self.num_layers)

    def main(self):
        # check if model should be loaded or not. Loads model if model_file_path is set
        if self.load_model:
            if os.path.exists(self.load_model_path):
                self.model = torch.load(self.load_model_path)

        # print and train model
        print(self.model)
        self.train_run(self.data_loader_train, self.num_iteration)

        # check if model should be evaluated or not (val set)
        if self.evaluate_model:
            self.evaluate_model_own(self.data_loader_train)

        # check if model should be evaluated or not (test set)
        if self.test_model:
            self.evaluate_model_own(self.data_loader_test)

    def init_model(self, output_dim, hidden_dim_enc, hidden_dim_dec, embed_size, num_layers):
        # create encoder-decoder model
        encoder = Encoder(hidden_dim_enc, num_layers, hidden_dim_dec)
        decoder = Decoder(output_dim, hidden_dim_dec, embed_size, num_layers)
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
        total_loss_iterations_run = 0
        total_loss_iterations_save = 0

        self.time_run = time.time()  # start taking time to show on print
        self.time_save = time.time()  # start taking time to show on save
        t_end = time.time() + 60 * self.minutes + 60 * 60 * self.hours  # remaining training time

        if self.use_epochs:
            for idx in range(1, num_iteration + 1):
                self.train_helper(criterion, idx, keypoints_loader, model_optimizer, total_loss_iterations_run,
                                  total_loss_iterations_save)

        else:
            idx = 1
            while time.time() < t_end:
                self.train_helper(criterion, idx, keypoints_loader, model_optimizer, total_loss_iterations_run,
                                  total_loss_iterations_save, t_end)
                idx += 1

    def train_helper(self, criterion, idx, keypoints_loader, model_optimizer, total_loss_iterations_run,
                     total_loss_iterations_save, t_end=0.0):
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
        it = iter(keypoints_loader)
        try:
            iterator_data = next(it)
        except StopIteration:  # reinitialize data loader if num_iteration > amount of data
            it = iter(keypoints_loader)

        source_ten = torch.as_tensor(iterator_data[0], dtype=torch.float).view(-1, 1)  # TODO remove [:20]
        target_ten = torch.as_tensor(iterator_data[1], dtype=torch.long).view(-1, 1)
        print("source_ten.size: %d, target_ten.size: %d" % (source_ten.size()[0], target_ten.size()[0]))
        loss = self.train_model(source_ten, target_ten, model_optimizer, criterion)
        total_loss_iterations_run += loss
        total_loss_iterations_save += loss

        if idx % self.show_after_epochs == 0:
            elapsed_time_s = int(time.time() - self.time_save)
            self.elapsed_time_sum += elapsed_time_s

            remaining_time = int(t_end - time.time())
            average_loss = total_loss_iterations_run / self.show_after_epochs
            total_loss_iterations_run = 0

            print('Epoch %d, average loss: %.2f, elapsed time: %s'
                  % (idx, average_loss, str(datetime.timedelta(seconds=self.elapsed_time_sum))))

            if t_end != 0.0:
                print('Remaining time: %s' % str(datetime.timedelta(seconds=remaining_time)))
            self.time_run = time.time()

        if idx % self.save_epoch == 0:
            elapsed_time_s = int(time.time() - self.time_save)
            average_loss = total_loss_iterations_save / self.save_epoch
            print('Saving at epoch %d, average loss: %.2f' % (idx, average_loss))
            total_loss_iterations_save = 0
            # refresh idx_t each time saving is callled
            idx_t = idx - self.idx_save

            self.documentation["epochs_total"] = idx_t
            self.documentation["time_total_s"] = elapsed_time_s
            self.documentation["loss"] = [round(average_loss, 2)]

            self.save_helper(self.save_state, Mode.train)
            self.idx_save = idx
            self.time_save = time.time()

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

    def evaluate_model_own(self, keypoints_loader):
        # evaluate (kommt da was sinnvolles raus?)
        it = iter(keypoints_loader)

        for idx in range(1, self.num_iteration_eval + 1):
            try:
                iterator_data = next(it)
            except StopIteration:  # reinitialize data loader if num_iteration > amount of data
                it = iter(keypoints_loader)

            with torch.no_grad():
                in_ten = torch.as_tensor(iterator_data[0], dtype=torch.float).view(-1, 1)
                out_ten = torch.as_tensor(iterator_data[1], dtype=torch.long).view(-1, 1)
                print("---" * 10)

                flat_list = []  # sentence representation in int
                for sublist in out_ten.tolist():
                    for item in sublist:
                        flat_list.append(item)

                hypothesis = DataUtils().int2text(flat_list, DataUtils().vocab_int2word(self.path_to_vocab_file_train))
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

                reference = DataUtils().int2text(decoded_words,
                                                 DataUtils().vocab_int2word(self.path_to_vocab_file_train))
                ref_str = " ".join(reference)

                if len(hypothesis) >= 4 or len(reference) >= 4:
                    # there may be several references
                    bleu_score = round(nltk.translate.bleu_score.sentence_bleu([reference], hypothesis), 2)
                    print("BLEU score: %d" % bleu_score)
                    self.documentation["BLEU"].append(bleu_score)

                print("Hyp: %s" % hyp_str)
                print("Ref: %s" % ref_str)
                self.documentation["hypothesis"].append(hyp_str)
                self.documentation["reference"].append(ref_str[:20])  # cut reference down so its readable in the log

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
