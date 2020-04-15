"""
run_model_helper.py: contains methods not linked directly to the model
"""

import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import time
import torch
from enum import Enum
import json


class Save(Enum):
    new = 0
    update = 1


class Mode(Enum):
    train = 0
    eval = 1


class Helper:

    def save_model(self, model, save_model_folder_path, save_model_file_path, doc, state, mode, doc_load_once={}):
        save_model_folder_path = Path(save_model_folder_path)

        if state == Save.new:
            save_model_folder_path = Path(save_model_folder_path)

            # create timestring to name folde to save current model
            timestr = time.strftime("%Y-%m-%d_%H-%M")

            # create new target directory, the files will be saved there
            if not os.path.exists(save_model_folder_path):
                os.makedirs(save_model_folder_path)

            if not os.path.exists(save_model_folder_path / timestr):
                os.makedirs(save_model_folder_path / timestr)

            current_folder = save_model_folder_path / timestr

            save_model_file_path = current_folder / "model.pt"
            # TODO uncommment to save model (testing)
            # torch.save(model, save_model_file_path)

            with open(current_folder / 'info.txt', 'w') as outfile:
                json.dump(doc, outfile)

            return save_model_file_path

        elif state == Save.update:
            current_folder = Path(save_model_file_path.parent)

            with open(current_folder / 'info.txt') as json_file:
                doc_load = json.load(json_file)

            if mode == Mode.train:
                doc_load["epochs_total"] = doc_load_once["epochs_total"] + doc["epochs_total"]
                doc_load["time_total_s"] = doc_load_once["time_total_s"] + doc["time_total_s"]
                doc_load["time_total_readable"] = \
                    str(timedelta(seconds=(int(doc_load["time_total_s"]))))  # convert the time above

                for element in doc["epochs"]:
                    doc_load["epochs"].append(element)

                for element in doc["loss"]:
                    doc_load["loss"].append(round(element, 2))

                for element in doc["elapsed_time"]:
                    doc_load["elapsed_time"].append(element)

            elif mode == Mode.eval:
                for element in doc["hypothesis"]:
                    doc_load["hypothesis"].append(element)

                for element in doc["reference"]:
                    doc_load["reference"].append(element)

                for element in doc["BLEU"]:
                    doc_load["BLEU"].append(element)

            torch.save(model, save_model_file_path)

            with open(current_folder / 'info.txt', 'w') as outfile:
                json.dump(doc_load, outfile)

    def get_origin_json(self, save_model_file_path):
        current_folder = Path(save_model_file_path.parent)

        with open(current_folder / 'info.txt') as json_file:
            return json.load(json_file)