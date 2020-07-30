"""run_lightning.py:

21.02.2020:

Using the transformer model with pytorch lightning. Using only the transformer model here because its suited for the usage with lightning.
I think moving the seq2seq with attention to lightning is tricky because of the AttnSeq2Seq class

"""
from __future__ import unicode_literals, division
import warnings
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
import torch.utils
import torch.utils.data
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
import sys
import json
from pathlib import Path
from tensorboardX import SummaryWriter
import time
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge
from statistics import mean
import math
import shutil
import traceback

# use try/except -> local and server import differs
try:
    from keypoints2text.kp_to_text_real_data.model_transformer import TransformerModel
    from keypoints2text.kp_to_text_real_data.data_loader import TextKeypointsDataset, ToTensor
    from keypoints2text.kp_to_text_real_data.data_utils import DataUtils
except ImportError:  # server uses different imports than local
    from data_loader import TextKeypointsDataset, ToTensor
    from data_utils import DataUtils
    from model_transformer import TransformerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
import traceback


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


class Litty(LightningModule):

    def __init__(self, hparams_path, timestr):
        super().__init__()

        # ____________________________________________________________________________________
        # SET PARAMS
        # ____________________________________________________________________________________
        with open(hparams_path) as json_file:
            config = json.load(json_file)
        # model settings
        self.teacher_forcing_ratio = config["model_settings"]["teacher_forcing_ratio"]
        self.input_size = config["model_settings"]["input_size"]
        self.output_size = config["model_settings"]["output_size"]
        self.hidden_size = config["model_settings"]["hidden_size"]
        self.num_layers = config["model_settings"]["num_layers"]
        self.max_length = config["model_settings"]["max_length"]
        self.padding = config["model_settings"]["padding"]
        self.dropout = config["model_settings"]["dropout"]
        self.bidir_encoder = config["model_settings"]["bidirectional_encoder"]
        self.batch_size = config["model_settings"]["batch_size"]
        self.fake_batch = config["model_settings"]["fake_batch"]


        # batch is multiplied by fake batch
        # e.g. batch_size = 8, fake_batch = 2 -> train with batch_size = 8, but optim.step() is calculated after 2
        if self.fake_batch < 1:
            self.fake_batch = 1

        self.model_type = config["model_settings"]["model_type"]  # model_type: basic, attn or trans
        self.num_workers = config["model_settings"]["num_workers"]
        self.reduceplt_lr_patience = config["learning_rate_settings"]["reduceplt_lr_patience"]
        self.learning_rate = config["learning_rate_settings"]["learning_rate"]
        self.auto_lr_find = config["learning_rate_settings"]["auto_lr_find"]
        self.lr = self.learning_rate

        # trans model settings
        self.nhead = config["trans_settings"]["nhead"]

        # train settings
        self.train_model_bool = config["train_settings"]["train_model_bool"]
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
        self.save_model_file_path = config["save_load"]["save_model_file_path"]  # full path ".../model.pt or model.ckpt"
        self.save_model_folder_path = config["save_load"]["save_model_folder_path"]
        self.save_every = config["save_load"]["save_every"]  # save each x epoch
        self.load_model = config["save_load"]["load_model"]
        self.load_model_path = config["save_load"]["load_model_path"]
        self.load_folder_path = config["save_load"]["load_folder_path"]

        if self.load_model:
            self.current_folder = self.load_folder_path
        else:
            self.current_folder = Path(self.save_model_folder_path) / timestr

        self.writer = SummaryWriter(self.current_folder)
        self.metrics = {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": [], "meteor": [], "rouge": []}

        self.save_params(hparams_path, self.current_folder)

        # ____________________________________________________________________________________
        # DEFINE MODEL
        # ____________________________________________________________________________________
        # self.transformer_encoder = TransformerEncoder(
        #     TransformerEncoderLayer(self.input_size, self.nhead, self.hidden_size, self.dropout), self.num_layers)
        # self.pos_encoder = PositionalEncoding(self.input_size, self.dropout)
        # # self.encoder = nn.Embedding(self.output_size, self.input_size)
        # self.decoder = nn.Linear(self.input_size, self.output_size)
        # initrange = 0.1
        # # self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.src_mask = None

        ntokens = self.output_size  # the size of vocabulary
        emsize = self.input_size  # embedding dimension
        nhid = self.hidden_size  # the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = self.num_layers  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = self.nhead  # the number of heads in the multiheadattention models
        dropout = self.dropout  # the dropout value
        self.model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)


    def save_params(self, hparams_path, current_folder):
        # save used parameter file
        shutil.copyfile(hparams_path, current_folder / "summary.json")

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        return self.model(src)

        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #     device = src.device
        #     mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #     self.src_mask = mask
        # # print(src.size())
        # # src = self.encoder(src) * math.sqrt(self.ninp)
        # # print(src.size())
        #
        # src = self.pos_encoder(src)
        #
        # output = self.transformer_encoder(src, self.src_mask)
        # output = self.decoder(output)
        # return output

    def val_dataloader(self):
        text2kp_val = TextKeypointsDataset(
            path_to_numpy_file=self.path_to_numpy_file_val,
            path_to_csv=self.path_to_csv_val,
            path_to_vocab_file=self.path_to_vocab_file_all,
            input_length=self.input_size,
            transform=ToTensor(),
            kp_max_len=self.padding,
            text_max_len=self.padding)
        data_loader_val = torch.utils.data.DataLoader(text2kp_val, batch_size=self.batch_size,
                                                      num_workers=self.num_workers)

        return data_loader_val

    def validation_step(self, batch, batch_idx):
        rouge = Rouge()
        source_tensor, target_tensor = batch
        source_tensor = source_tensor.permute(1, 0, 2)
        target_tensor = target_tensor.view(-1)
        target_tensor = target_tensor.type(torch.LongTensor).to(target_tensor.device)

        # ________
        # COMPUTE LOSS
        # ________
        output = self(source_tensor)

        ignore_index = DataUtils().text2index(["<pad>"], DataUtils().vocab_word2int(self.path_to_vocab_file_all))[0][0]
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        loss = criterion(output.view(-1, self.output_size), target_tensor)

        # ________
        # COMPUTE METRICS
        # ________

        # change from tensor to list
        # e.g. torch.Size([8000])tensor([ 397, 1339, 2807,  ...,    0,    0,    0], device='cuda:0')
        # to [ 397, 1339, 2807,  ...,    0,    0,    0]
        target_tensor_list = target_tensor.tolist()  # sentence representation in int

        # compute metrics for each batch
        for metrics_batch_index in range(self.batch_size):
            # read only from first batch
            decoded_words = []

            # cut batch in part sto compute metrics on
            # e.g. batch ~ [1,2,3,4,0,0,0,0,0,0,0,0,0,5,6,7,8,0,0,0,0,0,0.....]
            # cut into parts:
            # [1,2,3,4,0,0,0,0...] and [5,6,7,8,0,0,0,0,0....]
            # use these parts for metrics to compare
            # use mean score of all parts in one batch
            target_tensor_list_part = target_tensor_list[metrics_batch_index * self.padding:(metrics_batch_index + 1) * self.padding]
            hypothesis = DataUtils().int2text(target_tensor_list_part, DataUtils().vocab_int2word(self.path_to_vocab_file_train))
            hypothesis = list(filter("<pad>".__ne__, hypothesis))
            hypothesis = list(filter("<eos>".__ne__, hypothesis))
            hyp_str = " ".join(hypothesis)

            # with open('log_batches.txt', 'a') as f:
            #     f.write("target_tensor_list_part:\n")
            #     f.write(str(target_tensor_list_part))
            #     f.write("\n")

            for ot in range(output.size(0)):
                topv, topi = output[ot][metrics_batch_index].topk(1)
                if topi[0].item() == self.EOS_token:
                    decoded_words.append('<eos>')
                    break
                else:
                    decoded_words.append(topi[0].item())

            reference = DataUtils().int2text(decoded_words,
                                             DataUtils().vocab_int2word(self.path_to_vocab_file_all))
            reference = list(filter("<pad>".__ne__, reference))
            reference = list(filter("<eos>".__ne__, reference))
            ref_str = " ".join(reference[:len(hypothesis)])

            # print(f"\nhyp_str: {hyp_str}")
            # print(f"ref_str: {ref_str}")

            # if len(hypothesis) >= 4 or len(reference) >= 4:
            # there may be several references
            bleu1_score = round(sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0)), 4)
            bleu2_score = round(sentence_bleu([reference], hypothesis, weights=(0.5, 0.5, 0, 0)), 4)
            bleu3_score = round(sentence_bleu([reference], hypothesis, weights=(0.33, 0.33, 0.33, 0)), 4)
            bleu4_score = round(sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25)), 4)
            meteor_score = round(single_meteor_score(ref_str, hyp_str), 4)
            try:
                rouge_score = round(rouge.get_scores(hyp_str, ref_str)[0]["rouge-l"]["f"], 4)
            except ValueError:
                rouge_score = 0.0

            self.metrics["bleu1"].append(bleu1_score)
            self.metrics["bleu2"].append(bleu2_score)
            self.metrics["bleu3"].append(bleu3_score)
            self.metrics["bleu4"].append(bleu4_score)
            self.metrics["meteor"].append(meteor_score)
            self.metrics["rouge"].append(rouge_score)

        self.writer.add_scalars(f'metrics', {
            'bleu1': mean(self.metrics["bleu1"]),
            'bleu2': mean(self.metrics["bleu2"]),
            'bleu3': mean(self.metrics["bleu3"]),
            'bleu4': mean(self.metrics["bleu4"]),
            'meteor': mean(self.metrics["meteor"]),
            'rouge': mean(self.metrics["rouge"]),
        }, self.current_epoch)

        self.writer.add_scalar('lr', self.learning_rate, self.current_epoch)

        # reset
        self.metrics = {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": [], "meteor": [], "rouge": []}

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate))
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.reduceplt_lr_patience)
        # return [optimizer], [scheduler]
        return optimizer

    def train_dataloader(self):
        text2kp_train = TextKeypointsDataset(path_to_numpy_file=self.path_to_numpy_file_train,
                                             path_to_csv=self.path_to_csv_train,
                                             path_to_vocab_file=self.path_to_vocab_file_all,
                                             input_length=self.input_size,
                                             transform=ToTensor(), kp_max_len=self.padding, text_max_len=self.padding)
        data_loader_train = torch.utils.data.DataLoader(text2kp_train, batch_size=self.batch_size, shuffle=True,
                                                        num_workers=self.num_workers)
        return data_loader_train

    def training_step(self, batch, batch_idx):
        source_tensor, target_tensor = batch
        source_tensor = source_tensor.permute(1, 0, 2)
        # source_tensor = source_tensor.view(-1, self.batch_size, self.input_size).to(device)
        target_tensor = target_tensor.view(-1)
        target_tensor = target_tensor.type(torch.LongTensor).to(target_tensor.device)

        ignore_index = DataUtils().text2index(["<pad>"], DataUtils().vocab_word2int(self.path_to_vocab_file_all))[0][0]
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

        output = self(source_tensor)
        # print("target_tensor.size() %s" % str(target_tensor.size()))
        loss = criterion(output.view(-1, self.output_size), target_tensor)
        # self.logger.summary.scalar('loss', loss)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def on_epoch_end(self):
        trainer.save_checkpoint(self.current_folder / "model.ckpt")


if __name__ == '__main__':
    try:
        timestr = time.strftime("%Y-%m-%d_%H-%M")

        # set path to file containing all parameters
        if len(sys.argv) > 1:
            hparams_path = str(sys.argv[1])
        else:
            hparams_path = r"hparams.json"
        model = Litty(hparams_path, timestr)
        # trainer = Trainer(gpus=2, num_nodes=1, distributed_backend='ddp')

        if model.load_model == 1:
            trainer = Trainer(gpus=1, default_save_path=Path(model.save_model_folder_path) / timestr, resume_from_checkpoint=model.save_model_file_path, min_epochs=model.num_iteration, max_epochs=model.num_iteration)
        else:
            trainer = Trainer(gpus=1, default_save_path=Path(model.save_model_folder_path) / timestr, min_epochs=model.num_iteration, max_epochs=model.num_iteration)
        trainer.fit(model)
        # trainer.save_checkpoint(Path(model.save_model_folder_path) / timestr / "model.ckpt")
    except Exception as e:
        with open('log.txt', 'a') as f:
            f.write(str(e))
            f.write(traceback.format_exc())
