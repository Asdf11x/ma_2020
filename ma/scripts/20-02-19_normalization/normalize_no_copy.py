"""normalize.py: normalize json files in a folder
Copy json data from folder_name to folder_name_normalized
Work in the fodler folder_name_normalized
Get x/y_mean and x/y_stddev for whole folder and write it into a dictionary
use the dictionary to normalize the values
write normalized values into json file in folder_name_normalized

same functions as normalize.py, but copying files on-the-fly without a save copy beforehand
"""

import json
import time

import numpy as np
import os
import statistics
from distutils.dir_util import copy_tree
from pathlib import Path
import sys


class Normalize:

    def __init__(self, path_to_json_dir):
        self.path_to_json = path_to_json_dir

    def main_normalize(self):
        print("Start to copy files...")
        # self.copy_files()
        self.normalize()

    def copy_files(self):
        # copy
        os.walk(self.path_to_json)
        subdirectories = [x[1] for x in os.walk(self.path_to_json)]
        data_folder = Path(self.path_to_json)
        # if there are folders with "_normalized" dont copy them again
        subdirectories_copy = [s for s in subdirectories[0] if "_normalized" not in s]
        for subdir in subdirectories_copy:
            if not os.path.exists(data_folder / str(subdir + "_normalized")):
                os.makedirs(data_folder / str(subdir + "_normalized"))
            copy_tree(str(data_folder / subdir), str(data_folder / str(subdir + "_normalized")))

            print("Copied files from %s to %s" % (
            str(data_folder / subdir), str(data_folder / str(subdir + "_normalized"))))

    def normalize(self):
        # copy
        os.walk(self.path_to_json)
        subdirectories = [x[1] for x in os.walk(self.path_to_json)]
        data_folder = Path(self.path_to_json)
        # if there are folders with "_normalized" dont copy them again
        subdirectories_copy = [s for s in subdirectories[0] if "_normalized" not in s]
        for subdir in subdirectories_copy:
            if not os.path.exists(data_folder / str(subdir + "_normalized")):
                os.makedirs(data_folder / str(subdir + "_normalized"))
            # copy_tree(str(data_folder / subdir), str(data_folder / str(subdir + "_normalized")))

            print("Copied files from %s to %s" % (
            str(data_folder / subdir), str(data_folder / str(subdir + "_normalized"))))

        # used keys of openpose here
        keys = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
        folder_mean_stddev = {'pose_keypoints_2d': [], 'face_keypoints_2d': [], 'hand_left_keypoints_2d': [],
                              'hand_right_keypoints_2d': []}
        all_mean_stddev = {}

        # work
        os.walk(self.path_to_json)
        subdirectories = [x[1] for x in os.walk(self.path_to_json)]
        data_folder = Path(self.path_to_json)
        # if there are folders with "_normalized" use em for working directories
        subdirectories_work = [s for s in subdirectories[0] if "_normalized" in s]

        # get mean and stddev of whole folder and write it into a dictionary
        # the dictionary contains for each key the mean and stddev for x and y of the whole folder:
        # folder_name - key - 0 - 0: array of x_mean
        # folder_name - key - 0 - 1: array of x_stddev
        # folder_name - key - 1 - 0: array of y_mean
        # folder_name - key - 1 - 1: array of y_stddev
        for subdir in subdirectories_copy:
            print("Computing mean and stddev for %s" % (subdir))
            json_files = [pos_json for pos_json in os.listdir(data_folder / subdir)
                          if pos_json.endswith('.json')]
            idx = 0
            for k in keys:
                x_all = []
                y_all = []
                x_all_T = []
                y_all_T = []
                for file in json_files:
                    # print(file)
                    # set file for class
                    x, y = self.get_points(data_folder / subdir, file, k)

                    x_all.append(x)
                    y_all.append(y)

                    x_all_T = np.array(x_all).T.tolist()
                    y_all_T = np.array(y_all).T.tolist()
                    # print(x_all_T)
                # if idx % 500 == 0:
                #     print("%s file : %d of %d" % (file, idx, len(json_files)))
                # idx += 1

                # fill dictionary for each folder with y/x_mean, y/x_stddev
                folder_mean_stddev[k] = [self.get_mean_stddev(x_all_T), self.get_mean_stddev(y_all_T)]
            print(folder_mean_stddev)
            # print(folder_mean_stddev)
            all_mean_stddev[subdir] = folder_mean_stddev.copy()

        print("Computed all mean and stddev. Normalizing...")

        # use mean and stddev from above to compute values for the json files
        for subdir in subdirectories_copy:
            folder_mean_stddev = all_mean_stddev[subdir]
            json_files = [pos_json for pos_json in os.listdir(data_folder / subdir)
                          if pos_json.endswith('.json')]

            for file in json_files:
                jsonFile = open(data_folder / subdir / file, "r")  # Open the JSON file for reading
                data = json.load(jsonFile)  # Read the JSON into the buffer
                jsonFile.close()  # Close the JSON file

                # x -> [0::3]
                # y -> [1:.3]
                # c -> [2::3] (confidence)
                for k in keys:
                    # x values
                    temp_x = data['people'][0][k][0::3]
                    temp_y = data['people'][0][k][1::3]
                    temp_c = data['people'][0][k][2::3]

                    # get x values and normalize it
                    for index in range(len(temp_x)):
                        mean = folder_mean_stddev[k][0][0][index]
                        stddev = folder_mean_stddev[k][0][1][index]
                        if stddev != 0:
                            temp_x[index] = (temp_x[index] - mean) / stddev
                        else:
                            temp_x[index] = temp_x[index]

                    # get y values and normalize it
                    for index in range(len(temp_y)):
                        mean = folder_mean_stddev[k][1][0][index]
                        stddev = folder_mean_stddev[k][1][1][index]
                        if stddev != 0:
                            temp_y[index] = (temp_y[index] - mean) / stddev
                        else:
                            temp_y[index] = temp_y[index]

                    # build new array of normalized values
                    values = []
                    for index in range(len(temp_x)):
                        values.append(temp_x[index])
                        values.append(temp_y[index])
                        values.append(temp_c[index])

                    # copy the array of normalized values where it came from
                    data['people'][0][k] = values

                # ## Save our changes to JSON file
                jsonFile = open(data_folder / str(subdir + "_normalized") / file, "w+")
                jsonFile.write(json.dumps(data))
                jsonFile.close()

    def get_points(self, path, file, key):
        temp_df = json.load(open(path / file))
        temp_x_pose = temp_df['people'][0][key][0::3]
        temp_y_pose = temp_df['people'][0][key][1::3]
        return [temp_x_pose, temp_y_pose]

    def get_mean_stddev(self, values):
        means = []
        std_devs = []
        for array in values:
            # print(array)
            means.append(np.mean(array))
            std_devs.append(statistics.stdev(array))
        return [means, std_devs]

    def compute_normalization(self, values):
        result = []
        for array in values:
            # print(array)
            mean = np.mean(array)
            std_dev = statistics.stdev(array)
            helper_array = []
            for element in array:
                if std_dev != 0:
                    helper_array.append((element - mean) / std_dev)
                else:
                    helper_array.append(element)
            result.append(helper_array)
        return result


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path_to_json_dir = sys.argv[1]
    else:
        path_to_json_dir = r"C:\Users\Asdf\Downloads\How2Sign_samples\openpose_output\json"
    norm = Normalize(path_to_json_dir)
    start_time = time.time()
    norm.main_normalize()
    print("--- %s seconds ---" % (time.time() - start_time))
