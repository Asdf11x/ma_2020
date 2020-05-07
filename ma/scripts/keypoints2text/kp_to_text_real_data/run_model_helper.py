"""
run_model_helper.py: contains methods not directly involved in training/evaluating the model
"""

import os
from datetime import timedelta
from pathlib import Path
import time
import torch
from enum import Enum
import json
import shutil
from matplotlib import pyplot as plt
import numpy as np
try:
    from keypoints2text.kp_to_text_real_data.data_utils import DataUtils
except ImportError:  # server uses different imports than local
    from data_utils import DataUtils


class Save(Enum):
    new = 0
    update = 1


class Mode(Enum):
    train = 0
    eval = 1


class Helper:

    def __init__(self):
        self.summary_name = "summary.txt"
        self.run_info = "train_info.txt"
        self.eval_info = "eval_info.txt"
        self.model_name = "model.pt"

    def create_run_folder(self, save_model_folder_path):
        save_model_folder_path = Path(save_model_folder_path)
        # create timestring to name folde to save current model
        timestr = time.strftime("%Y-%m-%d_%H-%M")

        # create new target directory, the files will be saved there
        if not os.path.exists(save_model_folder_path):
            os.makedirs(save_model_folder_path)

        if not os.path.exists(save_model_folder_path / timestr):
            os.makedirs(save_model_folder_path / timestr)

        current_folder = save_model_folder_path / timestr
        return current_folder

    def save_model(self, model, current_folder, save_model_file_path, doc, state, mode, train_path="", val_path="",
                   test_path=""):
        """
        Use as documentation for a training run, saved training values are time, epochs, loss. Saved evaluation values are
        Hypothesis sample sentence, Reference sample sentence and BLEU score

        This method is executed in a extra "save loop", so each  x epochs (x=save_epoch) this method is run,
        saves values of the last x epochs and adds them to already saved ones
        """

        # if new, a new document is created and all values are saved in that document
        # new is only the first save of a new model, then Save is set to update
        if state == Save.new:
            if os.path.exists(train_path):
                train_length = DataUtils().get_file_length(train_path)
            else:
                train_length = 0

            if os.path.exists(val_path):
                val_length = DataUtils().get_file_length(val_path)
            else:
                val_length = 0

            if os.path.exists(test_path):
                test_length = DataUtils().get_file_length(test_path)
            else:
                test_length = 0

            shutil.copyfile("hparams.json", current_folder / self.summary_name)

            save_model_file_path = current_folder / self.model_name
            torch.save(model, save_model_file_path)

            doc["tloss_vloss_time_epoch"].append([float(doc["train_loss"][0]), float(doc["val_loss"][0]),
                                                  str(timedelta(seconds=(int(doc["time_total_s"])))),
                                                  doc["epochs_total"]])

            with open(current_folder / self.summary_name, 'a+') as outfile:
                outfile.write("\n\n")
                outfile.write(repr(model))
                outfile.write(
                    "\n\nTrain length: %d\nVal length: %d\nTest length: %d" % (train_length, val_length, test_length))

            with open(current_folder / self.run_info, 'w') as outfile:
                json.dump(doc, outfile)

            return save_model_file_path

        # if update: load old values from a file and add new values to it
        elif state == Save.update:

            current_folder = Path(save_model_file_path).parent

            with open(current_folder / self.run_info) as json_file:
                doc_load = json.load(json_file)

            if mode == Mode.train:
                doc_load["epochs_total"] = doc_load["epochs_total"] + doc["epochs_total"]
                doc_load["time_total_s"] = doc_load["time_total_s"] + doc["time_total_s"]
                doc_load["time_total_readable"] = \
                    str(timedelta(seconds=(int(doc_load["time_total_s"]))))  # convert the time above
                doc_load["train_loss"] = doc["train_loss"]
                doc_load["val_loss"] = doc["val_loss"]
                doc_load["tloss_vloss_time_epoch"].append(
                    [float(doc["train_loss"][0]), float(doc["val_loss"][0]), doc_load["time_total_readable"],
                     doc_load["epochs_total"]])

            elif mode == Mode.eval:
                for element in doc["hypothesis"]:
                    doc_load["hypothesis"].append(element)

                for element in doc["reference"]:
                    doc_load["reference"].append(element)

                for element in doc["Epoch_BLEU1-4_METEOR_ROUGE"]:
                    element.insert(0, doc_load["epochs_total"])  # add epochs as information
                    doc_load["Epoch_BLEU1-4_METEOR_ROUGE"].append(element)

            torch.save(model, save_model_file_path)

            with open(current_folder / self.run_info, 'w') as outfile:
                json.dump(doc_load, outfile)

    def save_graph(self, current_folder, plotter):
        current_folder = Path(current_folder)
        train = plotter["train_loss"]
        val = plotter["val_loss"]

        x = np.linspace(0, len(train), len(train))
        plt.plot(x, train, label="train")
        plt.plot(x, val, label="val")
        plt.legend()
        # plt.show()
        save_graph = "train_val_loss_%d.png" % len(plotter["train_loss"])
        plt.savefig(current_folder / save_graph)

    def get_origin_json(self, save_model_file_path):
        current_folder = Path(save_model_file_path.parent)

        with open(current_folder / self.run_info) as json_file:
            return json.load(json_file)
