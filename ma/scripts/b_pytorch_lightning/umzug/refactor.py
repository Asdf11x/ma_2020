"""refactor.py: starting to move my stuff to the lightning modules"""
from __future__ import unicode_literals, division

import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils
import torch.utils.data
import torch.utils.data
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from pytorch_lightning import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

from keypoints2text.kp_to_text_real_data.data_loader import TextKeypointsDataset, ToTensor
from keypoints2text.kp_to_text_real_data.data_utils import DataUtils


class Litty(LightningModule):

    def __init__(self):
        super().__init__()
        ntoken = 176  # the size of vocabulary
        ninp = 274  # embedding dimension
        nhid = 128  # the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 2  # the number of heads in the multiheadattention models
        dropout = 0.2  # the dropout value
        self.model_type = 'Transformer'
        self.src_mask = None
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        # print(src.size())
        # src = self.encoder(src) * math.sqrt(self.ninp)
        # print(src.size())

        # src = self.pos_encoder(src)
        # print(src.size())

        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def train_dataloader(self):
        # Dataloaders for train, val & test
        text2kp_train = TextKeypointsDataset(
            path_to_numpy_file="C:\\Users\\Asdf\\Downloads\\How2Sign_samples\\all_files_normalized.npy",
            path_to_csv="C:\\Users\\Asdf\\Downloads\\How2Sign_samples\\text_sample\\3_linked_to_npy\\how2sign.train.id_transformed.txt_2npy.txt",
            path_to_vocab_file="C:\\Users\\Asdf\\Downloads\\How2Sign_samples\\text_sample\\1_vocab_list\\vocab_merged.txt",
            transform=ToTensor(),
            kp_max_len=300,
            text_max_len=300)
        data_loader_train = torch.utils.data.DataLoader(text2kp_train, batch_size=2, shuffle=True,
                                                        num_workers=0)
        return data_loader_train

    # def val_dataloader(self):
    #     text2kp_val = TextKeypointsDataset(
    #         path_to_numpy_file=self.path_to_numpy_file_val,
    #         path_to_csv=self.path_to_csv_val,
    #         path_to_vocab_file=self.path_to_vocab_file_val,
    #         transform=ToTensor(),
    #         kp_max_len=self.padding,
    #         text_max_len=self.padding)
    #     data_loader_val = torch.utils.data.DataLoader(text2kp_val, batch_size=self.batch_size, shuffle=True, num_workers=0)
    #     return data_loader_val

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        source_tensor, target_tensor = batch
        source_tensor = source_tensor.view(-1, self.batch_size, 274)
        target_tensor = target_tensor.view(-1)
        ignore_index = DataUtils().text2index(["<pad>"], DataUtils().vocab_word2int(self.path_to_vocab_file_all))[0][0]
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

        output = self(source_tensor)
        # print("target_tensor.size() %s" % str(target_tensor.size()))
        loss = criterion(output.view(-1, self.output_dim), target_tensor)
        self.logger.summary.scalar('loss', loss)
        return loss


model = Litty()
trainer = Trainer()
trainer.fit(model)
