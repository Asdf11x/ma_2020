"""centralize_normalize_np.py: normalize over all files from each folder of an directory
- read keypoints from each folder from an directory and write them into a dictionary
- compute the mean and stdev of each value
- use the mean and stdev to normalize the data and write them into new json, repeat for all folders

Version description:
- data read in row per row and transpose
- Using the reduced all_files_np.npy file
"""

import json
import random

import numpy as np
import os
import statistics
from pathlib import Path
import sys
import time
import warnings
import os
import psutil
import copy

class Normalize:

    def __init__(self, path_to_numpy_file, path_to_target_dir="", path_to_json_dir=""):
        self.path_to_numpy_file = Path(path_to_numpy_file)
        self.path_to_target_dir = path_to_target_dir
        self.path_to_json = Path(path_to_json_dir)
        self.keys = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']

        # set global? np.load settings. If not, np.load throws error
        old = np.load
        np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)

    def main(self):

        # create target directory
        self.create_folders()
        self.print_memory_usage()

        # centralize values
        all_files_dictionary_centralized = None
        # dont use centralization, in current version
        # all_files_dictionary_centralized = self.centralize()

        # normalize values
        # Either by '_transposed', '_np' or '_column'
        all_mean_stdev = self.compute_mean_stdev_transposed(all_files_dictionary_centralized)
        self.normalize(all_mean_stdev, all_files_dictionary_centralized)
        # self.print_memory_usage()

    def create_folders(self):
        if self.path_to_target_dir == "":
            data_dir_target = self.path_to_numpy_file.parent
        else:
            data_dir_target = self.path_to_target_dir

        # create new target directory, the files will be saved there
        if not os.path.exists(data_dir_target):
            os.makedirs(data_dir_target)
        self.path_to_target_dir = Path(data_dir_target)

    def centralize(self):

        # load from .npy file
        print("To centralize load file from \n %s" % self.path_to_numpy_file)
        all_files_dictionary = np.load(self.path_to_numpy_file).item()

        # used keys of openpose here
        for subdir in all_files_dictionary.keys():

            all_files = {}
            round_precision = 5

            # load files from one folder into dictionary
            for file in all_files_dictionary[subdir]:
                temp_df = all_files_dictionary[subdir][file]
                # print(temp_df)
                all_files[file] = {}
                once = 1
                # init dictionaries & write x, y values into dictionary
                for k in self.keys:
                    all_files[file][k] = {'x': [], 'y': []}
                    all_files[file][k]['x'].append(temp_df['people'][0][k][0::3])
                    all_files[file][k]['y'].append(temp_df['people'][0][k][1::3])
                    temp_c = temp_df['people'][0][k][2::3]
                    results_x = []
                    results_y = []
                    x_in_key = all_files[file][k]['x'][0]
                    y_in_key = all_files[file][k]['y'][0]

                    # set neck once
                    if once == 1:
                        neck_zero_x = all_files[file][k]['x'][0][0]
                        neck_zero_y = all_files[file][k]['y'][0][0]
                        once = 0

                    # compute for pose
                    if k == "pose_keypoints_2d":
                        results_x.append(0)
                        results_y.append(0)
                        # start with 1 -> element 0 is neck
                        # get upper body
                        for idx in range(1, len(x_in_key[:9])):
                            if x_in_key[idx] == 0:
                                results_x.append('Null')
                            else:
                                results_x.append(round(neck_zero_x - x_in_key[idx], round_precision))

                            if y_in_key[idx] == 0:
                                results_y.append("Null")
                            else:
                                results_y.append(round(neck_zero_y - y_in_key[idx], round_precision))

                        # add Null as legs
                        results_x += (['Null'] * 6)
                        results_y += (['Null'] * 6)

                        for idx in range(15, len(x_in_key[:19])):
                            if x_in_key[idx] == 0:
                                results_x.append('Null')
                            else:
                                results_x.append(round(neck_zero_x - x_in_key[idx], round_precision))

                            if y_in_key[idx] == 0:
                                results_y.append("Null")
                            else:
                                results_y.append(round(neck_zero_y - y_in_key[idx], round_precision))

                        # add more legs
                        results_x += (['Null'] * 6)
                        results_y += (['Null'] * 6)

                        values = []
                        for index in range(len(temp_c)):
                            values.append(results_x[index])
                            values.append(results_y[index])
                            values.append(temp_c[index])
                        temp_df['people'][0][k] = values
                    else:
                        # start with 1 -> element 0 is neck
                        # get upper body
                        for idx in range(0, len(x_in_key)):
                            if x_in_key[idx] == 0:
                                results_x.append('Null')
                            else:
                                results_x.append(round(neck_zero_x - x_in_key[idx], round_precision))

                            if y_in_key[idx] == 0:
                                results_y.append("Null")
                            else:
                                results_y.append(round(neck_zero_y - y_in_key[idx], round_precision))

                        values = []
                        for index in range(len(temp_c)):
                            values.append(results_x[index])
                            values.append(results_y[index])
                            values.append(temp_c[index])
                        temp_df['people'][0][k] = values

                all_files_dictionary[subdir][file] = temp_df

                # ## Save our changes to JSON file
                # jsonFile = open(data_dir_target / subdir / file, "w+")
                # jsonFile.write(json.dumps(temp_df))
                # jsonFile.close()

        print("centralization done")
        self.save_to_numpy(all_files_dictionary)
        return all_files_dictionary

    def save_to_numpy(self, all_files_dictionary):
        dictionary_file_path = self.path_to_target_dir / 'all_files_centralized.npy'
        last_folder = os.path.basename(os.path.normpath(dictionary_file_path.parent)) + "/" + str(
            dictionary_file_path.name)
        print("Saving centralized results to %s " % last_folder)
        np.save(dictionary_file_path, all_files_dictionary)

    def compute_mean_stdev_transposed(self, all_files_dictionary_centralized=None):
        """
        Read data row per row and transpose to computed mean and stdev
        :param all_files_dictionary_centralized:
        :return:
        """

        all_files = self.dictionary_check(all_files_dictionary_centralized)
        self.print_memory_usage()
        # use keys of openpose here
        # all_mean_stdev = {}  # holds means and stdev of each directory, one json file per directory

        # keys
        # | - key (pose, face hands)
        # | - - x_coord
        # | - - - x_values of file_0
        # | - - - x_values of file_n
        # | - - y_coord

        keys = [[[], []], [[], []], [[], []], [[], []]]
        all_mean_stdev = [[[], []], [[], []], [[], []], [[], []]]

        for subdir in all_files:
            # load files from one folder into dictionary
            for file in subdir:
                # for key in file:
                for i in range(len(file)):
                    # print(file[i][0::2])
                    keys[i][0].append(file[i][0::2].tolist())
                    keys[i][1].append(file[i][1::2].tolist())

        print("Files read, computing mean and stdev")

        # for k in keys:
        for i in range(len(keys)):
            mean_stdev_x = []
            mean_stdev_y = []
            for list in np.array(keys[i][0]).T:
                if "Null" in list:
                    list = [i for i in list if i != "Null"]
                    if list == []:
                        mean_stdev_x.append(["Null", "Null"])
                    else:
                        list = [float(item) for item in list]
                        mean_stdev_x.append([np.mean(list), statistics.pstdev(list)])
                else:
                    list = [float(item) for item in list]
                    mean_stdev_x.append([np.mean(list), statistics.pstdev(list)])
                    # print(mean_stdev_x)

            for list in np.array(keys[i][1]).T:
                if "Null" in list:
                    list = [i for i in list if i != "Null"]
                    if list == []:
                        mean_stdev_y.append(["Null", "Null"])
                    else:
                        list = [float(item) for item in list]
                        mean_stdev_y.append([np.mean(list), statistics.pstdev(list)])
                else:
                    list = [float(item) for item in list]
                    mean_stdev_y.append([np.mean(list), statistics.pstdev(list)])

            all_mean_stdev[i] = [np.array(mean_stdev_x).T.tolist(), np.array(mean_stdev_y).T.tolist()]

        # write the computed means and std_dev into json file
        f = open(self.path_to_target_dir / "all_mean_stdev.json", "w")
        f.write(json.dumps(all_mean_stdev))
        f.close()

        return all_mean_stdev

    def compute_mean_stdev_np(self, all_files_dictionary_centralized=None):
        """
        Read data column per column with np arrays and np.c_ to computed mean and stdev
        :param all_files_dictionary_centralized:
        :return:
        """

        all_files = self.dictionary_check(all_files_dictionary_centralized)
        self.print_memory_usage()
        # use keys of openpose here
        all_mean_stdev = {}  # holds means and stdev of each directory, one json file per directory
        once = 1
        all_files_xy = {'all': {}}

        for subdir in all_files[:1]:
            # load files from one folder into dictionary
            for file in subdir[:1]:
                temp_df = file
                print(temp_df)
                if once == 1:
                    for k in self.keys:
                        all_files_xy['all'][k] = {'x': np.empty((len(temp_df['people'][0][k][0::3]), 0), dtype=np.float),
                                                  'y': np.empty((len(temp_df['people'][0][k][1::3]), 0), dtype=np.float)}
                    once = 0

                for k in self.keys:
                    all_files_xy['all'][k]['x'] = np.c_[
                        all_files_xy['all'][k]['x'], np.array(temp_df['people'][0][k][0::3])]
                    all_files_xy['all'][k]['y'] = np.c_[
                        all_files_xy['all'][k]['y'], np.array(temp_df['people'][0][k][1::3])]
        print("Files read, computing mean and stdev")

        for k in self.keys:
            mean_stdev_x = []
            mean_stdev_y = []
            for list in np.array(all_files_xy['all'][k]['x']):
                warnings.simplefilter(action='ignore', category=FutureWarning)
                if "Null" in list:
                    mean_stdev_x.append(["Null", "Null"])
                else:
                    list = [float(item) for item in list]
                    mean_stdev_x.append([np.mean(list), statistics.pstdev(list)])

            for list in np.array(all_files_xy['all'][k]['y']):
                if "Null" in list:
                    mean_stdev_y.append(["Null", "Null"])
                else:
                    list = [float(item) for item in list]
                    mean_stdev_y.append([np.mean(list), statistics.pstdev(list)])

            all_mean_stdev[k] = [np.array(mean_stdev_x).T.tolist(), np.array(mean_stdev_y).T.tolist()]

        # write the computed means and std_dev into json file
        f = open(self.path_to_target_dir / "all_mean_stdev.json", "w")
        f.write(json.dumps(all_mean_stdev))
        f.close()

        return all_mean_stdev

    def compute_mean_stdev_column(self, all_files_dictionary_centralized=None):
        """
        Read data column per column with lists and for each loop  to computed mean and stdev
        :param all_files_dictionary_centralized:
        :return:
        """
        all_files = self.dictionary_check(all_files_dictionary_centralized)
        self.print_memory_usage()

        # use keys of openpose here
        all_mean_stdev = {}  # holds means and stdev of each directory, one json file per directory
        once = 1
        all_files_xy = {'all': {}}
        self.print_memory_usage()
        print("load data into dictionary")

        for subdir in all_files.keys():
            # load files from one folder into dictionary
            for file in all_files[subdir]:
                temp_df = all_files[subdir][file]
                if once == 1:
                    for k in self.keys:
                        all_files_xy['all'][k] = {'x': [[] for x in range(len(temp_df['people'][0][k][0::3]))],
                                                  'y': [[] for x in range(len(temp_df['people'][0][k][1::3]))]}

                    once = 0

                for k in self.keys:
                    for i in range(len(temp_df['people'][0][k][0::3])):
                        all_files_xy['all'][k]['x'][i].append(temp_df['people'][0][k][0::3][i])
                        all_files_xy['all'][k]['y'][i].append(temp_df['people'][0][k][1::3][i])

        self.print_memory_usage()
        print("Files read, computing mean and stdev")

        for k in self.keys:
            mean_stdev_x = []
            mean_stdev_y = []
            self.print_memory_usage()
            for list in np.array(all_files_xy['all'][k]['x']):

                warnings.simplefilter(action='ignore', category=FutureWarning)
                if 'Null' in list:
                    mean_stdev_x.append(["Null", "Null"])
                else:
                    list = [float(item) for item in list]
                    mean_stdev_x.append([np.mean(list), statistics.pstdev(list)])

            for list in np.array(all_files_xy['all'][k]['y']):
                if 'Null' in list:
                    mean_stdev_y.append(["Null", "Null"])
                else:
                    list = [float(item) for item in list]
                    mean_stdev_y.append([np.mean(list), statistics.pstdev(list)])

            all_mean_stdev[k] = [np.array(mean_stdev_x).T.tolist(), np.array(mean_stdev_y).T.tolist()]

        # write the computed means and std_dev into json file
        f = open(self.path_to_target_dir / "all_mean_stdev.json", "w")
        f.write(json.dumps(all_mean_stdev))
        f.close()

        return all_mean_stdev

    def normalize(self, all_mean_stdev, all_files_dictionary_centralized=None):

        all_files = self.dictionary_check(all_files_dictionary_centralized)
        self.print_memory_usage()
        dirs_list = []
        # use mean and stdev to compute values for the json files
        for subdir in all_files:
            files_list = []
            for file in subdir:
                keypoints = []

                # x -> [0::3]
                # y -> [1:.3]
                # c -> [2::3] (confidence)
                for k in range(len(file)):
                    # x values
                    temp_x = file[k][0::2]
                    temp_y = file[k][1::2]

                    # get x, y values and normalize it
                    for index in range(len(temp_x)):
                        mean_x = all_mean_stdev[k][0][0][index]
                        stdev_x = all_mean_stdev[k][0][1][index]

                        mean_y = all_mean_stdev[k][1][0][index]
                        stdev_y = all_mean_stdev[k][1][1][index]

                        if temp_x[index] == "Null":
                            temp_x[index] = temp_x[index]
                        elif str(stdev_x) == "Null":
                            temp_x[index] = temp_x[index]
                        elif float(stdev_x) == 0:
                            temp_x[index] = temp_x[index]
                        else:
                            temp_x[index] = (temp_x[index] - float(mean_x)) / float(stdev_x)

                        if temp_y[index] == "Null":
                            temp_y[index] = temp_y[index]
                        elif str(stdev_y) == "Null":
                            temp_y[index] = temp_y[index]
                        elif float(stdev_y) == 0:
                            temp_y[index] = temp_y[index]
                        else:
                            temp_y[index] = (temp_y[index] - float(mean_y)) / float(stdev_y)

                    # build new array of normalized values
                    values = [j for i in zip(temp_x, temp_y) for j in i]
                    # copy the array of normalized values where it came from
                    keypoints.append(values)

                files_list.append(keypoints)
            # print(files_list)
            dirs_list.append(files_list)
        print(dirs_list)
        self.print_memory_usage()

        dictionary_file_path = self.path_to_target_dir / 'all_files_normalized.npy'
        last_folder = os.path.basename(os.path.normpath(dictionary_file_path.parent)) + "/" + str(
            dictionary_file_path.name)
        print("Saving normalized results to %s " % last_folder)
        np.save(dictionary_file_path, files_list)
        self.print_memory_usage()

    def dictionary_check(self, all_files_dictionary_centralized):
        # load from .npy file
        if all_files_dictionary_centralized is None:
            print("Loading from %s file" % self.path_to_numpy_file)
            all_files = np.load(self.path_to_numpy_file)
        else:
            print("Using internal centralized dictionary")
            all_files = all_files_dictionary_centralized
        return all_files

    def save_dictionary_to_file(self, subdirectories):
        dictionary_file_path = self.path_to_target_dir / 'all_files.npy'
        last_folder = os.path.basename(os.path.normpath(dictionary_file_path.parent)) + "/" + str(
            dictionary_file_path.name)

        if dictionary_file_path.is_file():
            print(".../%s file already exists. Not copying files " % last_folder)
            return dictionary_file_path
        else:
            print("Saving files to %s " % dictionary_file_path)

        # use keys of openpose here
        all_files = {}

        for subdir in subdirectories:
            print("Reading files from %s" % subdir)
            json_files = [pos_json for pos_json in os.listdir(Path(self.path_to_json) / subdir)
                          if pos_json.endswith('.json')]
            all_files[subdir] = {}
            # load files from one folder into dictionary
            for file in json_files:
                temp_df = json.load(open(Path(self.path_to_json) / subdir / file))
                all_files[subdir][file] = temp_df

        np.save(dictionary_file_path, all_files)
        return Path(dictionary_file_path)

    def print_memory_usage(self):
        process = psutil.Process(os.getpid())
        print("Current memory usage: %s MB" % str(process.memory_info().rss / 1000000))  # divided to get mb


if __name__ == '__main__':
    # path to numpy file, necessary to run the script
    if len(sys.argv) > 1:
        path_to_numpy_file = sys.argv[1]
    else:
        print("Set numpy file")
        sys.exit()

    # target directory
    path_to_target_dir = ""
    if len(sys.argv) > 2:
        path_to_target_dir = sys.argv[2]
    else:
        print("Target directory not set, using default")

    path_to_json_dir = ""
    # origin json files directory
    if len(sys.argv) > 3:
        path_to_json_dir = sys.argv[3]

    norm = Normalize(path_to_numpy_file, path_to_target_dir, path_to_json_dir)
    start_time = time.time()
    norm.main()
    print("--- %.4s seconds ---" % (time.time() - start_time))
