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
from keypoints2text.kp_to_text_real_data.data_loader import TextKeypointsDataset, ToTensor
from keypoints2text.kp_to_text_real_data.model_seq2seq import Encoder, Decoder, Seq2Seq
from keypoints2text.kp_to_text_guru99.data_utils import DataUtils
from keypoints2text.kp_to_text_real_data.run_model_helper import Helper, Save, Mode
import datetime
import nltk
from copy import deepcopy
import configparser


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RunModel:

    def __init__(self):
        config = configparser.ConfigParser()
        config.read("hparams.ini")

        # TODO create settings file
        # model settings
        self.teacher_forcing_ratio = config["model_settings"]["teacher_forcing_ratio"]
        self.embed_size = 256  # vocab list size
        # TODO different hidden size for encoder and decoder, now: same size for both
        self.hidden_size = 512
        self.num_layers = 1

        # train settings
        self.use_epochs = 1  # 0: time, 1: epochs
        self.num_iteration = 10
        self.hours = 0
        self.minutes = 30
        self.show_after_epochs = 1

        # eval settings
        self.evaluate_model = 0  # 0: model is not evaluated, 1: model is evaluated
        self.num_iteration_eval = 20

        # test settings
        self.test_model = 0  # 0: model is not tested, 1: model is tested

        # train
        self.path_to_numpy_file_train = r"C:\Users\Asdf\Downloads\How2Sign_samples\all_files_normalized.npy"
        self.path_to_csv_train = r"C:\Users\Asdf\Downloads\How2Sign_samples\text\3_linked_to_npy\how2sign.test.id_transformed.txt_2npy.txt"
        self.path_to_vocab_file_train = r"C:\Users\Asdf\Downloads\How2Sign_samples\text_vocab\how2sign.test.id_vocab.txt"

        # val
        self.path_to_numpy_file_val = r"C:\Users\Asdf\Downloads\How2Sign_samples\all_files_normalized.npy"
        self.path_to_csv_val = r"C:\Users\Asdf\Downloads\How2Sign_samples\text\3_linked_to_npy\how2sign.test.id_transformed.txt_2npy.txt"
        self.path_to_vocab_file_val = r"C:\Users\Asdf\Downloads\How2Sign_samples\text_vocab\how2sign.test.id_vocab.txt"

        # test
        self.path_to_numpy_file_test = r"C:\Users\Asdf\Downloads\How2Sign_samples\all_files_normalized.npy"
        self.path_to_csv_test = r"C:\Users\Asdf\Downloads\How2Sign_samples\text\3_linked_to_npy\how2sign.test.id_transformed.txt_2npy.txt"
        self.path_to_vocab_file_test = r"C:\Users\Asdf\Downloads\How2Sign_samples\text_vocab\how2sign.test.id_vocab.txt"

        # vocab file, containing unique words for all (train, val & test)
        self.path_to_vocab_file_all = r"C:\Users\Asdf\Downloads\How2Sign_samples\text_vocab\how2sign.test.id_vocab.txt"

        # Dataloaders for train, val & test
        text2kp_train = TextKeypointsDataset(
            path_to_numpy_file=self.path_to_numpy_file_train,
            path_to_csv=self.path_to_csv_train,
            path_to_vocab_file=self.path_to_vocab_file_train,
            transform=ToTensor())
        self.data_loader_train = torch.utils.data.DataLoader(text2kp_train, batch_size=1, shuffle=True,
                                                             num_workers=0)

        # text2kp_val = TextKeypointsDataset(
        #     path_to_numpy_file=self.path_to_numpy_file_train,
        #     path_to_csv=self.path_to_csv_train,
        #     path_to_vocab_file=self.path_to_vocab_file_train,
        #     transform=ToTensor())
        # self.data_loader_val = torch.utils.data.DataLoader(text2kp_val, batch_size=1, shuffle=True, num_workers=0)
        #
        # text2kp_test = TextKeypointsDataset(
        #     path_to_numpy_file=self.path_to_numpy_file_train,
        #     path_to_csv=self.path_to_csv_train,
        #     path_to_vocab_file=self.path_to_vocab_file_train,
        #     transform=ToTensor())
        # self.data_loader_test = torch.utils.data.DataLoader(text2kp_test, batch_size=1, shuffle=True, num_workers=0)

        # set tokens
        self.PAD_token = 0
        self.UNK_token = 1
        self.SOS_token = 2
        self.EOS_token = 3

        # save / load
        self.save_model = 1  # 0: model is not saved, 1: model is saved
        # self.save_model_file_path = r"C:\Eigene_Programme\Git-Data\Own_Repositories\ma_2020\ma\scripts\keypoints2text\kp_to_text_real_data\saved_models\2020-04-17_15-16\model.pt"  # if not empty use path, else create new folder, use only when documentation exists
        self.save_model_file_path = r""  # if not empty use path, else create new folder, use only when documentation exists
        self.save_model_folder_path = r"C:\Eigene_Programme\Git-Data\Own_Repositories\ma_2020\ma\scripts\keypoints2text\kp_to_text_real_data\saved_models"
        self.save_loss = []  # list to save loss results
        self.save_eval = []  # list to save evaluation results
        self.save_epoch = 1  # save each x epoch

        # if no path to a model is set -> State.new, if a path ot a model exists, just keep updating
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
        self.elapsed_time_sum = 0.0
        self.idx_save = 0
        self.time_run = time.time()  # start taking time to show on print, init variable
        self.time_save = time.time()  # start taking time to show on save, init variable

        self.load_model = 0
        self.load_model_path = ""
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

        unique_words = DataUtils().get_vocab_file_length(self.path_to_vocab_file_all)
        # TODO: get input_dim automatically?
        # TODO: crop max input_dim?
        self.input_dim = 100000  # length of source keypoints
        self.output_dim = unique_words + 1  # output_dim != max_length. max_length == hidden_size
        # self.hidden = max sentence length

        # run only if hparams not available
        # print("Checking max lengths.")
        # self.hidden_dim_enc, self.hidden_dim_dec, lengths = DataUtils().get_kp_text_max_lengths(self.data_loader_train, self.data_loader_train, self.data_loader_train)
        # print("max length checking is done")
        # with open('lengths.txt', 'w') as f:
        #     for item in lengths:
        #         f.write("%s\n" % item)

        self.model = self.init_model(self.input_dim, self.output_dim, self.hidden_size, self.embed_size,
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
