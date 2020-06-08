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

# use try/except -> local and server import differs
try:
    from keypoints2text.kp_to_text_real_data.data_loader import TextKeypointsDataset, ToTensor
    from keypoints2text.kp_to_text_real_data.data_utils import DataUtils
except ImportError:  # server uses different imports than local
    from data_loader import TextKeypointsDataset, ToTensor
    from data_utils import DataUtils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
import traceback


class Litty(LightningModule):

    def __init__(self, hparams_path):
        super().__init__()

        # ____________________________________________________________________________________
        # SET PARAMS
        # ____________________________________________________________________________________
        with open(hparams_path) as json_file:
            config = json.load(json_file)
        # model settings
        self.teacher_forcing_ratio = config["model_settings"]["teacher_forcing_ratio"]
        self.input_size = config["model_settings"]["input_size"]
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
        self.learning_rate = config["learning_rate_settings"]["learning_rate"]

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
        self.save_model_file_path = config["save_load"]["save_model_file_path"]  # full path ".../model.pt"
        self.save_model_folder_path = config["save_load"]["save_model_folder_path"]
        self.save_every = config["save_load"]["save_every"]  # save each x epoch
        self.output_dim = config["padding"]["output_dim"]

        # ____________________________________________________________________________________
        # DEFINE MODEL
        # ____________________________________________________________________________________
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(self.input_size, self.nhead, self.hidden_size, self.dropout), self.num_layers)
        self.encoder = nn.Embedding(self.output_dim, self.input_size)
        self.decoder = nn.Linear(self.input_size, self.output_dim)
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.src_mask = None

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(self.device)
            self.src_mask = mask
        # print(src.size())
        # src = self.encoder(src) * math.sqrt(self.ninp)
        # print(src.size())

        # src = self.pos_encoder(src)
        # print(src.size())

        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    # def val_dataloader(self):
    #     text2kp_val = TextKeypointsDataset(
    #         path_to_numpy_file=self.path_to_numpy_file_val,
    #         path_to_csv=self.path_to_csv_val,
    #         path_to_vocab_file=self.path_to_vocab_file_all,
    #          input_length = self.input_size,
    #         transform=ToTensor(),
    #         kp_max_len=self.padding,
    #         text_max_len=self.padding)
    #     data_loader_val = torch.utils.data.DataLoader(text2kp_val, batch_size=self.batch_size, num_workers=0)
    #
    #     return data_loader_val
    #
    # def validation_step(self, batch, batch_idx):
    #     source_tensor, target_tensor = batch
    #     source_tensor = source_tensor.permute(1, 0, 2)
    #     target_tensor = target_tensor.view(-1)
    #     target_tensor = target_tensor.type(torch.LongTensor).to(self.device)
    #     output = self(source_tensor)
    #     ignore_index = DataUtils().text2index(["<pad>"], DataUtils().vocab_word2int(self.path_to_vocab_file_all))[0][0]
    #     criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    #     # print("target_tensor.size() %s" % str(target_tensor.size()))
    #     loss = criterion(output.view(-1, self.output_dim), target_tensor)
    #     return {'loss': loss}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        text2kp_train = TextKeypointsDataset(path_to_numpy_file=self.path_to_numpy_file_train,
                                             path_to_csv=self.path_to_csv_train,
                                             path_to_vocab_file=self.path_to_vocab_file_all, input_length=self.input_size,
                                             transform=ToTensor(), kp_max_len=self.padding, text_max_len=self.padding)
        data_loader_train = torch.utils.data.DataLoader(text2kp_train, batch_size=self.batch_size, shuffle=True,
                                                        num_workers=0)
        return data_loader_train

    def training_step(self, batch, batch_idx):
        source_tensor, target_tensor = batch
        source_tensor = source_tensor.permute(1, 0, 2)
        # source_tensor = source_tensor.view(-1, self.batch_size, self.input_size).to(device)
        target_tensor = target_tensor.view(-1)
        target_tensor = target_tensor.type(torch.LongTensor).to(self.device)

        # try:
        #     source_tensor, target_tensor = batch
        #     source_tensor = source_tensor.view(-1, self.batch_size, self.input_size).to(device)
        #     target_tensor = target_tensor.view(-1)
        #     target_tensor = target_tensor.type(torch.LongTensor).to(device)
        # except RuntimeError as e:
        #     with open('log.txt', 'a+') as f:
        #         f.write(str(e))
        #         f.write(traceback.format_exc())
        #
        #     with open("runtime_error.txt", "a+") as myfile:
        #         myfile.write("\nrun:\n")
        #         myfile.write(str(source_tensor.size()))
        #         myfile.write("\n")
        #         myfile.write(str(source_tensor))
        #         myfile.write("\n")
        #         myfile.write(str(target_tensor.size()))
        #         myfile.write("\n")
        #         myfile.write(str(target_tensor))
        #         myfile.write("\n")
        #         myfile.write(
        #             str(DataUtils().int2text(target_tensor, DataUtils().vocab_int2word(self.path_to_vocab_file_all))))
        #         myfile.write("\n")
        ignore_index = DataUtils().text2index(["<pad>"], DataUtils().vocab_word2int(self.path_to_vocab_file_all))[0][0]
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

        output = self(source_tensor)
        # print("target_tensor.size() %s" % str(target_tensor.size()))
        loss = criterion(output.view(-1, self.output_dim), target_tensor)
        # self.logger.summary.scalar('loss', loss)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}


if __name__ == '__main__':
    # set path to file containing all parameters
    if len(sys.argv) > 1:
        hparams_path = str(sys.argv[1])
    else:
        hparams_path = r"hparams.json"
    model = Litty(hparams_path)
    trainer = Trainer()
    trainer.fit(model)
