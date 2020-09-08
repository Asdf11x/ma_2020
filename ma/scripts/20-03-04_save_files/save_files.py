"""save_files.py: save json files in a single numpy file

use batches to save the files into numpy file.
"batches": Create a txt file on the side which contains all folder names and this file is processed in batches
"""

import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import psutil


class SaveFiles:

    def __init__(self, path_to_json_dir, path_to_target_dir):
        self.path_to_json = Path(path_to_json_dir)
        self.path_to_target_dir = path_to_target_dir
        self.keys = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
        self.remaining_folders_name = "remaining_folders.txt"
        # set global? np.load settings. If not, np.load throws error
        old = np.load
        np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)

    def main(self):
        # create folders and paths
        data_dir_target, subdirectories = self.create_folders()

        # read files to dictionary and save dictionary in target directory
        self.copy_dictionary_to_file(data_dir_target, subdirectories)

    def create_folders(self):
        if self.path_to_target_dir == "":
            data_dir_target = self.path_to_json.parent / str(self.path_to_json.name + "_saved_numpy")
        else:
            data_dir_target = Path(self.path_to_target_dir)

        # create new target directory, the files will be saved there
        if not os.path.exists(data_dir_target):
            os.makedirs(data_dir_target)

        if os.path.isfile(data_dir_target / self.remaining_folders_name):
            subdirectories = np.loadtxt(data_dir_target / self.remaining_folders_name, delimiter="\n",
                                        dtype="str").tolist()
        else:
            # get subdirectories of the path
            subdirectories = [x[1] for x in os.walk(self.path_to_json)][0]
            np.savetxt((data_dir_target / self.remaining_folders_name), subdirectories, delimiter="\n", fmt="%s")

        print("%d folders left." % len(subdirectories))
        if len(subdirectories) == 0:
            print("No subdirectories left. Exit.")
            sys.exit()

        return data_dir_target, subdirectories

    def copy_dictionary_to_file(self, data_dir_target, subdirectories):
        dictionary_file_path = data_dir_target / 'raw_data.npy'
        subdirectories_file = subdirectories.copy()

        # proceed in batches
        if len(subdirectories) > 2500:
            subdirectories = subdirectories[:2000]
        self.print_memory_usage()

        print("Saving files to %s " % dictionary_file_path)

        all_files = {}
        index = 0

        for subdir in subdirectories:
            index += 1
            print("%d of %d" % (index, len(subdirectories)))
            print("Reading files from %s" % subdir)
            self.print_memory_usage()

            json_files = [pos_json for pos_json in os.listdir(self.path_to_json / subdir)
                          if pos_json.endswith('.json')]
            all_files[subdir] = {}
            # load files from one folder into dictionary
            for file in json_files:
                temp_df = json.load(open(self.path_to_json / subdir / file))
                all_files[subdir][file] = temp_df

            subdirectories_file.remove(subdir)

        self.print_memory_usage()
        np.savetxt((data_dir_target / self.remaining_folders_name), subdirectories_file, delimiter="\n", fmt="%s")

        # update .npy file
        if os.path.isfile(dictionary_file_path):
            dictionary_from_file = np.load(dictionary_file_path).item()
            dictionary_from_file.update(all_files)
            np.save(dictionary_file_path, dictionary_from_file)
        else:
            np.save(dictionary_file_path, all_files)

        return Path(dictionary_file_path)

    def print_memory_usage(self):
        process = psutil.Process(os.getpid())
        print("Current memory usage: %.1f MB" % float(process.memory_info().rss / 1000000))  # divided to get mb


if __name__ == '__main__':
    # origin json files directory
    if len(sys.argv) > 1:
        path_to_json_dir = sys.argv[1]
    else:
        print("Set json file directory")
        sys.exit()

    # target directory
    path_to_target_dir = ""
    if len(sys.argv) > 2:
        path_to_target_dir = sys.argv[2]
    try:
        norm = SaveFiles(path_to_json_dir, path_to_target_dir)
        start_time = time.time()
        norm.main()
        print("--- %.4s seconds ---" % (time.time() - start_time))
    except NameError:
        print("Set paths")
