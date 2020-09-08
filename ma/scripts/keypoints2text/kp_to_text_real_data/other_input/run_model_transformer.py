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
from tensorboardX import SummaryWriter
import datetime
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge
from pathlib import Path
import json

try:
    from keypoints2text.kp_to_text_real_data.data_loader import TextKeypointsDataset, ToTensor
    from keypoints2text.kp_to_text_real_data.data_loader import TextKeypointsDataset, ToTensor
    from keypoints2text.kp_to_text_real_data.model_seq2seq import Encoder, Decoder, Seq2Seq
    from keypoints2text.kp_to_text_real_data.model_seq2seq_attention import AttnEncoder, AttnDecoderRNN, AttnSeq2Seq
    from keypoints2text.kp_to_text_real_data.model_transformer import TransformerModel
    from keypoints2text.kp_to_text_real_data.data_utils import DataUtils
    from keypoints2text.kp_to_text_real_data.save_model import Helper, Save, Mode
except ImportError:  # server uses different imports than local
    from data_loader_framewise import TextKeypointsDataset, ToTensor
    from model_seq2seq import Encoder, Decoder, Seq2Seq
    from model_seq2seq_attention import AttnEncoder, AttnDecoderRNN, AttnSeq2Seq
    from model_transformer import TransformerModel
    from data_utils import DataUtils
    from run_model_helper import Helper, Save, Mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RunModel:

    def __init__(self):

        # read settings file, didnt use configparser, because config parser saves everything as string
        with open("../hparams.json") as json_file:
            config = json.load(json_file)

        # model settings
        self.teacher_forcing_ratio = config["model_settings"]["teacher_forcing_ratio"]
        self.input_size = config["model_settings"]["input_size"] * config["model_settings"]["batch_size"]
        self.hidden_size = config["model_settings"]["hidden_size"]
        self.num_layers = config["model_settings"]["num_layers"]
        self.max_length = config["model_settings"]["max_length"]
        self.dropout = config["model_settings"]["dropout"]
        self.bidir_encoder = config["model_settings"]["bidirectional_encoder"]
        self.batch_size = config["model_settings"]["batch_size"]

        # train settings
        self.use_epochs = config["train_settings"]["use_epochs"]  # 0: time, 1: epochs
        self.num_iteration = config["train_settings"]["num_iteration"]
        self.hours = config["train_settings"]["hours"]
        self.minutes = config["train_settings"]["minutes"]
        self.show_every = config["train_settings"]["show_every"]

        # eval settings
        # 0: model is not evaluated, 1: model is evaluated
        self.evaluate_model = config["eval_settings"]["evaluate_model"]
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
        # if not empty use path, else create new folder, use only when documentation exists
        self.save_model_file_path = config["save_load"]["save_model_file_path"]  # full path ".../model.pt"
        self.save_model_folder_path = config["save_load"]["save_model_folder_path"]
        self.save_every = config["save_load"]["save_every"]  # save each x epoch
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
                              "train_loss": [],
                              "val_loss": [],
                              "tloss_vloss_time_epoch": [],
                              "hypothesis": [],
                              "reference": [],
                              "Epoch_BLEU1-4_METEOR_ROUGE": [],
                              }

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

        # Create new folder if no path to a model is specified
        if self.load_model_path == "":
            self.current_folder = Helper().create_run_folder(self.save_model_folder_path)
        else:
            self.current_folder = os.path.dirname(self.load_model_path)

        # Dataloaders for train, val & test
        text2kp_train = TextKeypointsDataset(path_to_numpy_file=self.path_to_numpy_file_train,
                                             path_to_csv=self.path_to_csv_train,
                                             path_to_vocab_file=self.path_to_vocab_file_train, input_length=,
                                             transform=ToTensor())
        self.data_loader_train = torch.utils.data.DataLoader(text2kp_train, batch_size=self.batch_size, shuffle=True,
                                                             num_workers=0)

        # vocab size, amount of different unique words
        if self.output_dim == 0:
            self.output_dim = DataUtils().get_file_length(self.path_to_vocab_file_all)

        # max length of source keypoints and target sentence
        if self.hidden_size == 0:
            print("Searching for source and target max length")
            max_len_source, max_len_target, lengths = DataUtils().get_kp_text_max_lengths(
                self.data_loader_train, self.data_loader_train, self.data_loader_train)
            if max_len_source > max_len_target:
                self.hidden_size = max_len_source
            else:
                self.hidden_size = max_len_target
            with open('../../quick_test/lengths.txt', 'w') as f:
                for item in lengths:
                    f.write("%s\n" % item)

        # Dataloaders for train, val & test
        # text2kp_train = TextKeypointsDataset(
        #     path_to_numpy_file=self.path_to_numpy_file_train,
        #     path_to_csv=self.path_to_csv_train,
        #     path_to_vocab_file=self.path_to_vocab_file_train,
        #     transform=ToTensor())
        # self.data_loader_train = torch.utils.data.DataLoader(text2kp_train, batch_size=self.batch_size, shuffle=True,
        #                                                      num_workers=0)

        text2kp_val = TextKeypointsDataset(path_to_numpy_file=self.path_to_numpy_file_val,
                                           path_to_csv=self.path_to_csv_val,
                                           path_to_vocab_file=self.path_to_vocab_file_val, input_length=,
                                           transform=ToTensor())
        self.data_loader_val = torch.utils.data.DataLoader(text2kp_val, batch_size=self.batch_size, shuffle=True, num_workers=0)

        #
        # text2kp_test = TextKeypointsDataset(
        #     path_to_numpy_file=self.path_to_numpy_file_test,
        #     path_to_csv=self.path_to_csv_test,
        #     path_to_vocab_file=self.path_to_vocab_file_test,
        #     transform=ToTensor(),
        #     kp_max_len=self.hidden_size_enc,
        #     text_max_len=self.hidden_size_dec)
        # self.data_loader_test = torch.utils.data.DataLoader(text2kp_test, batch_size=1, shuffle=True, num_workers=0)

        # model options: "basic", "attn", "trans"
        self.writer = SummaryWriter(self.current_folder)
        self.plotter = {"train_loss": [], "val_loss": []}
        self.model_type = "trans"
        if self.model_type == "basic":
            self.model = self.init_model(self.input_size, self.output_dim, self.hidden_size,
                                         self.num_layers, self.SOS_token, self.EOS_token)
        elif self.model_type == "attn":
            self.model = self.init_model_attn(self.input_size, self.output_dim, self.hidden_size, self.num_layers,
                                              self.dropout, self.teacher_forcing_ratio, self.max_length,
                                              self.bidir_encoder, self.SOS_token,
                                              self.EOS_token)
        elif self.model_type == "trans":
            self.model = self.init_model_trans(self.output_dim, self.hidden_size, self.num_layers)

    def main(self):
        # check if model should be loaded or not. Loads model if model_file_path is set
        if self.load_model:
            if os.path.exists(self.load_model_path):
                self.model = torch.load(self.load_model_path)

        # print and train model
        print(self.model)
        self.train_run(self.data_loader_train, self.data_loader_val, self.num_iteration)
        self.writer.close()

        # save graph after training
        Helper().save_graph(self.current_folder, self.plotter)

        # check if model should be evaluated or not (val set)
        if self.evaluate_model:
            self.evaluate_model_own(self.data_loader_val)

        # check if model should be evaluated or not (test set)
        if self.test_model:
            self.evaluate_model_own(self.data_loader_test)

    def init_model(self, input_dim, output_dim, hidden_dim, num_layers, SOS_token, EOS_token):
        # create encoder-decoder model
        encoder = Encoder(input_dim, hidden_dim, num_layers)
        decoder = Decoder(output_dim, hidden_dim, num_layers)
        model = Seq2Seq(encoder, decoder, device, SOS_token, EOS_token).to(device)
        return model

    def init_model_attn(self, input_dim, output_dim, hidden_dim, num_layers, dropout, teacher_forcing, max_length, bi_encoder,
                        SOS_token, EOS_token):
        # create encoder-decoder model with attention
        encoder = AttnEncoder(input_dim, hidden_dim, num_layers, bi_encoder)
        decoder = AttnDecoderRNN(output_dim, hidden_dim, num_layers, dropout, max_length, bi_encoder)
        model = AttnSeq2Seq(encoder, decoder, device, teacher_forcing, max_length, SOS_token, EOS_token).to(device)
        return model

    def init_model_trans(self, output_dim, hidden_dim, num_layers):
        ntokens = output_dim  # the size of vocabulary
        emsize = 274  # embedding dimension
        nhid = 274  # the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 2  # the number of heads in the multiheadattention models
        dropout = 0.2  # the dropout value
        model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
        return model

    def train_run(self, train_loader, val_loader, num_iteration):
        """
        the outer most train method, the whole train procesdure is started here
        :param train_loader:
        :param num_iteration:
        :return:
        """
        self.model.train()
        model_optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        time_run = time.time()  # start taking time to show on print
        time_save = time.time()  # start taking time to show on save
        time_end = time.time() + 60 * self.minutes + 60 * 60 * self.hours  # remaining training time

        train_loss_show = 0
        train_loss_save = 0
        idx_epoch = 1
        idx_epoch_save = 0
        it_train = iter(train_loader)

        val_loss_show = 0
        val_loss_save = 0
        it_val = iter(val_loader)


        if self.use_epochs == 1:
            remaining = 1
            end = num_iteration
            time_end = 0
        else:
            remaining = time.time()
            end = time_end

        while remaining <= end:

            train_data = self.load_data(it_train, train_loader)
            train_loss = self.train_model(train_data, model_optimizer, criterion)
            train_loss_show += train_loss
            train_loss_save += train_loss

            val_data = self.load_data(it_val, val_loader)
            val_loss = self.val_model(val_data, criterion)
            val_loss_show += val_loss
            val_loss_save += val_loss

            # save for plotting
            self.plotter["train_loss"].append(train_loss)
            self.plotter["val_loss"].append(val_loss)

            # add to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, idx_epoch)
            self.writer.add_scalar('Loss/val', val_loss, idx_epoch)

            if idx_epoch % self.show_every == 0:
                elapsed_time_s = time.time() - time_run
                train_avg_loss = train_loss_show / self.show_every
                train_loss_show = 0

                val_avg_loss = val_loss_show / self.show_every
                val_loss_show = 0

                print('Epoch %d, avg train loss: %.2f, avg eval loss: %.2f, elapsed time: %s'
                      % (idx_epoch, train_avg_loss, val_avg_loss, str(datetime.timedelta(seconds=int(elapsed_time_s)))))

                remaining_time = int(time_end - time.time())
                if time_end != 0.0:
                    print('Remaining time: %s' % str(datetime.timedelta(seconds=remaining_time)))

            if idx_epoch % self.save_every == 0:
                elapsed_time_s = time.time() - time_save
                train_avg_loss = train_loss_save / self.save_every
                train_loss_save = 0

                val_avg_loss = val_loss_save / self.save_every
                val_loss_save = 0

                print('Saving at epoch %d, average loss: %.2f' % (idx_epoch, train_avg_loss))

                # refresh idx_epoch_save each time saving is called
                idx_epoch_save = idx_epoch - idx_epoch_save
                self.documentation["epochs_total"] = idx_epoch_save
                self.documentation["time_total_s"] = elapsed_time_s
                self.documentation["train_loss"] = [round(train_avg_loss, 2)]
                self.documentation["val_loss"] = [round(val_avg_loss, 2)]

                idx_epoch_save = idx_epoch
                time_save = time.time()
                self.save_helper(self.save_state, Mode.train)

            if self.use_epochs == 1:
                remaining += 1
            else:
                remaining = time.time()

            idx_epoch += 1

    def load_data(self, data_iterator, data_loader):
        """
        Load input data from iterator and data_loader
        Check if data has min and max length
        :param data_iterator:
        :param data_loader:
        :return: source and target data
        """
        while 1:
            try:
                data = next(data_iterator)
                # data[0].size(): (batchsize=1, frames=15, keypoints=274) => [1, 15, 274]
                # data[0].size(0): 15
                # data[1].size(): (batchsize=1, words=3) => [1, 3]
                # data[1].size(0): 3
                source_tensor = torch.as_tensor(data[0], dtype=torch.float, device=device).view(-1, self.batch_size, 274)
                target_tensor = torch.as_tensor(data[1], dtype=torch.float, device=device).view(-1, self.batch_size)

                source_tensor_size = source_tensor.size(0)
                target_tensor_size = target_tensor.size(0)

                if 0 < source_tensor_size <= self.max_length and 0 < target_tensor_size:
                    break
            except StopIteration:  # reinitialize data loader if num_iteration > amount of data
                data_iterator = iter(data_loader)
        return data

    def train_model(self, train_data, model_optimizer, criterion):
        """
        the inner most method to train the model, the actual training is implemented here
        :param source_tensor:
        :param target_tensor:
        :param model_optimizer:
        :param criterion:
        :return:
        """
        self.model.train()
        source_tensor = train_data[0]
        target_tensor = train_data[1]

        model_optimizer.zero_grad()
        epoch_loss = 0.0

        output = self.model(source_tensor)
        print("target_tensor.size() %s" % str(target_tensor.size()))
        loss = criterion(output.view(-1, 176), target_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        model_optimizer.step()

        epoch_loss += loss.item()
        return epoch_loss

    def val_model(self, data, criterion):
        """Validate model during train runtime"""
        epoch_loss = 0.0
        self.model.eval()  # Turn on the evaluation mode
        with torch.no_grad():
            source_tensor = torch.as_tensor(data[0], dtype=torch.float, device=device).view(-1, self.batch_size, 274)
            target_tensor = torch.as_tensor(data[1], dtype=torch.long, device=device).view(-1)

            output = self.model(source_tensor)
            print("target_tensor.size() %s" % str(target_tensor.size()))
            loss = criterion(output.view(-1, 176), target_tensor)
            epoch_loss += loss.item()
        return epoch_loss

    def evaluate_model_own(self, keypoints_loader):
        """
        Evaluate Model after trainign and print example sentences
        
        :param keypoints_loader: 
        :return: 
        """""
        self.model.eval()  # Turn on the evaluation mode
        it = iter(keypoints_loader)
        rouge = Rouge()
        for idx in range(1, self.num_iteration_eval + 1):
            iterator_data = self.load_data(it, keypoints_loader)

            with torch.no_grad():
                in_ten = torch.as_tensor(iterator_data[0], dtype=torch.float, device=device).view(-1, self.batch_size, 274)
                out_ten = torch.as_tensor(iterator_data[1], dtype=torch.long, device=device).view(-1)
                print("---" * 10)

                flat_list = []  # sentence representation in int
                for sublist in out_ten.tolist():
                    flat_list.append(sublist)

                hypothesis = DataUtils().int2text(flat_list, DataUtils().vocab_int2word(self.path_to_vocab_file_train))
                hypothesis = list(filter("<pad>".__ne__, hypothesis))
                hypothesis = list(filter("<eos>".__ne__, hypothesis))

                hyp_str = " ".join(hypothesis)

                print("in_ten.size: %d, out_ten.size: %d" % (in_ten.size()[0], out_ten.size()[0]))
                decoded_words = []

                output = self.model(in_ten)
                print("---" * 10)
                for ot in range(output.size(0)):
                    topv, topi = output[ot].topk(1)
                    if topi[0].item() == self.EOS_token:
                        decoded_words.append('<eos>')
                        break
                    else:
                        decoded_words.append(topi[0].item())

                reference = DataUtils().int2text(decoded_words, DataUtils().vocab_int2word(self.path_to_vocab_file_train))
                reference = list(filter("<pad>".__ne__, reference))
                reference = list(filter("<eos>".__ne__, reference))

                ref_str = " ".join(reference)

                # if len(hypothesis) >= 4 or len(reference) >= 4:
                # there may be several references
                bleu1_score = round(sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0)), 4)
                bleu2_score = round(sentence_bleu([reference], hypothesis, weights=(0.5, 0.5, 0, 0)), 4)
                bleu3_score = round(sentence_bleu([reference], hypothesis, weights=(0.33, 0.33, 0.33, 0)), 4)
                bleu4_score = round(sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25)), 4)
                meteor_score = round(single_meteor_score(ref_str, hyp_str), 4)
                # print(rouge.get_scores(hyp_str, ref_str))
                try:
                    rouge_score = round(rouge.get_scores(hyp_str, ref_str)[0]["rouge-l"]["f"], 4)
                except ValueError:
                    rouge_score = 0.0

                self.documentation["Epoch_BLEU1-4_METEOR_ROUGE"].append([bleu1_score, bleu2_score, bleu3_score,
                                                                         bleu4_score, meteor_score, rouge_score])

                # cumulative BLEU scores
                print('BLEU Cumulative 1-gram: %f' % bleu1_score)
                print('BLEU Cumulative 2-gram: %f' % bleu2_score)
                print('BLEU Cumulative 3-gram: %f' % bleu3_score)
                print('BLEU Cumulative 4-gram: %f' % bleu4_score)
                print('BLEU Cumulative 4-gram: %f' % bleu4_score)
                print('METEOR score: %f' % meteor_score)
                print('ROUGE-L F1 score score: %f' % rouge_score)

                print("Hyp: %s" % hyp_str)
                print("Ref: %s" % ref_str)
                self.documentation["hypothesis"].append(hyp_str)
                self.documentation["reference"].append(ref_str)  # cut reference down so its readable in the log

                self.save_helper(self.save_state, Mode.eval)

    def save_helper(self, save, mode):

        if self.save_model:

            # save in new file and model
            if save == Save.new:
                self.save_model_file_path = Helper().save_model(self.model, self.current_folder,
                                                                self.save_model_file_path,
                                                                self.documentation, self.save_state, mode,
                                                                self.path_to_csv_train, self.path_to_csv_val,
                                                                self.path_to_csv_test)
                self.save_state = Save.update

            # update file and model
            else:
                Helper().save_model(self.model, self.current_folder, self.save_model_file_path,
                                    self.documentation, self.save_state, mode)

            # reset variables
            self.documentation = {"epochs_total": 0,
                                  "time_total_s": 0,
                                  "time_total_readable": "",
                                  "train_loss": [],
                                  "val_loss": [],
                                  "tloss_vloss_time_epoch": [],
                                  "hypothesis": [],
                                  "reference": [],
                                  "Epoch_BLEU1-4_METEOR_ROUGE": [],
                                  }


if __name__ == '__main__':
    runny = RunModel()
    runny.main()
