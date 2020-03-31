"""save_files.py: save json files in a single numpy file"""

import json
import numpy as np
import os
from pathlib import Path
import sys
import time
import psutil

class SaveFiles:

    def __init__(self, path_to_json_dir, path_to_target_dir):
        self.path_to_json = Path(path_to_json_dir)
        self.path_to_target_dir = path_to_target_dir
        self.keys = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']

    def main(self):
        # create folders and paths
        data_dir_target, subdirectories = self.create_folders()

        # read files to dictionary and save dictionary in target directory
        self.copy_dictionary_to_file(data_dir_target, subdirectories)

    def create_folders(self):
        # get subdirectories of the path
        subdirectories = [x[1] for x in os.walk(self.path_to_json)]
        subdirectories = subdirectories[0]

        if self.path_to_target_dir == "":
            data_dir_target = self.path_to_json.parent / str(self.path_to_json.name + "_saved_numpy")
        else:
            data_dir_target = Path(self.path_to_target_dir)

        # create new target directory, the files will be saved there
        if not os.path.exists(data_dir_target):
            os.makedirs(data_dir_target)

        return data_dir_target, subdirectories

    def copy_dictionary_to_file(self, data_dir_target, subdirectories):
        dictionary_file_path = data_dir_target / 'all_files_np.npy'
        last_folder = os.path.basename(os.path.normpath(dictionary_file_path.parent)) + "/" + str(
            dictionary_file_path.name)
        self.print_memory_usage()
        print("Saving files to %s " % dictionary_file_path)

        index = 0
        dirs_list = []
        for subdir in subdirectories:
            self.print_memory_usage()
            print("Reading files from %s" % subdir)
            json_files = [pos_json for pos_json in os.listdir(self.path_to_json / subdir)
                          if pos_json.endswith('.json')]
            files_list = []

            # load files from one folder into dictionary
            for file in json_files:
                temp_df = json.load(open(self.path_to_json / subdir / file))
                keypoints = []

                for k in self.keys:
                    # drop confidence
                    xy = np.delete(temp_df['people'][0][k], np.arange(2, len(temp_df['people'][0][k]), 3, dtype=np.float32))
                    keypoints.append(xy)
                files_list.append(keypoints)
            dirs_list.append(files_list)

            index += 1
            print("%d of %d" % (index, len(subdirectories)))

        self.print_memory_usage()
        np.save(dictionary_file_path, dirs_list)
        return Path(dictionary_file_path)

    def print_memory_usage(self):
        process = psutil.Process(os.getpid())
        print("Current memory usage: %s MB" % str(process.memory_info().rss / 1000000))  # divided to get mb


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
