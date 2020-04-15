"""
run_model_helper.py: contains methods not linked directly to the model
"""

import os
import re
import sys
import time
from pathlib import Path
import time
import torch


class Helper:

    def save_model(self, model, save_model_folder_path, save_model_file_path, save_loss="", save_eval=""):

        # if file path available just save model
        if save_model_file_path != "":
            save_model_file_path = Path(save_model_file_path)
            torch.save(model, save_model_file_path)

            with open(save_model_file_path.parent / "info.txt", "a") as text_file:
                timestr = time.strftime("%Y-%m-%d_%H-%M")
                text_file.writelines("%s \n" % str(timestr))
                text_file.writelines(["%s\n" % item for item in save_loss])
                text_file.write("\n")
                text_file.writelines(["%s\n" % item for item in save_eval])

        else:
            save_model_folder_path = Path(save_model_folder_path)

            # create new target directory, the files will be saved there
            if not os.path.exists(save_model_folder_path):
                os.makedirs(save_model_folder_path)

            # create timestring to name folde to save current model
            timestr = time.strftime("%Y-%m-%d_%H-%M")

            if not os.path.exists(save_model_folder_path / timestr):
                os.makedirs(save_model_folder_path / timestr)

            save_model_folder_path = save_model_folder_path / timestr

            save_model_file_path = save_model_folder_path / "model.pt"
            # TODO uncommment to save model (testing)
            # torch.save(model, save_model_file_path)

            with open(save_model_folder_path / "info.txt", "w+") as text_file:
                text_file.write(repr(model))
                text_file.write("\n")
                text_file.writelines(["%s\n" % item for item in save_loss])
                text_file.write("\n")
                text_file.writelines(["%s\n" % item for item in save_eval])
            return save_model_file_path


# Helper().save_model("",
#                     r"C:\Eigene_Programme\Git-Data\Own_Repositories\ma_2020\ma\scripts\keypoints2text\kp_to_text_real_data\saved_models", "",
#                     ["hi", "0.5"], ["eval", "BLEU Score"])
