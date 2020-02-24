"""normalize_slow.py: normalize json files in a folder
Copy json data from folder_name to folder_name_normalized
Work in the fodler folder_name_normalized
Get x/y_mean and x/y_stddev for whole folder and write it into a dictionary
use the dictionary to normalize the values
write normalized values into json file in folder_name_normalized
"""

import json
import numpy as np
import os
import statistics
from pathlib import Path
import sys
import time


class Normalize:

    def __init__(self, path_to_json_dir):
        self.path_to_json = path_to_json_dir

    def main_normalize(self):
        self.normalize()

    def normalize(self):
        # get subdirectories of the path
        os.walk(self.path_to_json)
        subdirectories = [x[1] for x in os.walk(self.path_to_json)]
        data_dir_origin = Path(self.path_to_json)
        subdirectories = subdirectories[0]

        # create new target directory, the centralized fiels will be saved there
        if not os.path.exists(data_dir_origin.parent / str(data_dir_origin.name + "_normalized")):
            os.makedirs(data_dir_origin.parent / str(data_dir_origin.name + "_normalized"))

        data_dir_target = data_dir_origin.parent / str(data_dir_origin.name + "_normalized")

        for subdir in subdirectories:
            if not os.path.exists(data_dir_target / subdir):
                os.makedirs(data_dir_target / subdir)

        # used keys of openpose here
        keys = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
        folder_mean_stddev = {'pose_keypoints_2d': [], 'face_keypoints_2d': [], 'hand_left_keypoints_2d': [],
                              'hand_right_keypoints_2d': []}
        all_mean_stddev = {}  # holds means and stddev of each directory, one json file per directory
        total_computed_mean_stddev = {}  # holds the computed means and stddevs from all_mean_stddev, so its one json file in total
        files_amount = {}
        once = 1
        # get mean and stddev of whole folder and write it into a dictionary
        # the dictionary contains for each key the mean and stddev for x and y of the whole folder:
        # folder_name - key - 0 - 0: array of x_mean
        # folder_name - key - 0 - 1: array of x_stddev
        # folder_name - key - 1 - 0: array of y_mean
        # folder_name - key - 1 - 1: array of y_stddev
        all_files = {}
        all_files['all'] = {}
        for subdir in subdirectories:
            print("Computing mean and stddev for %s" % (subdir))
            json_files = [pos_json for pos_json in os.listdir(data_dir_origin / subdir)
                          if pos_json.endswith('.json')]

            folder_mean_stddev = {}
            files_amount[subdir] = len([name for name in Path(data_dir_origin / subdir).iterdir()
                                        if os.path.isfile(name)])

            # load files from one folder into dictionary
            for file in json_files:
                temp_df = json.load(open(data_dir_origin / subdir / file))
                if once == 1:
                    for k in keys:
                        all_files['all'][k] = {'x': [], 'y': []}
                    once = 0
                for k in keys:
                    all_files['all'][k]['x'].append(temp_df['people'][0][k][0::3])
                    all_files['all'][k]['y'].append(temp_df['people'][0][k][1::3])

        mean_stddev_x = []
        mean_stddev_y = []
        for k in keys:
            for list in np.array(all_files['all'][k]['x']).T.tolist():
                mean_stddev_x.append([np.mean(list), statistics.pstdev(list)])

            for list in np.array(all_files['all'][k]['y']).T.tolist():
                mean_stddev_y.append([np.mean(list), statistics.pstdev(list)])

            all_mean_stddev[k] = [np.array(mean_stddev_x).T.tolist(), np.array(mean_stddev_y).T.tolist()]

        print(all_mean_stddev)

        print("Computed all mean and stddev. Normalizing...")

        # use mean and stddev from above to compute values for the json files

        for subdir in subdirectories:
            json_files = [pos_json for pos_json in os.listdir(data_dir_origin / subdir)
                          if pos_json.endswith('.json')]

            for file in json_files:
                jsonFile = open(data_dir_origin / subdir / file, "r")  # Open the JSON file for reading
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
                        mean = all_mean_stddev[k][0][0][index]
                        stddev = all_mean_stddev[k][0][1][index]
                        if stddev != 0:
                            temp_x[index] = (temp_x[index] - mean) / stddev
                        else:
                            temp_x[index] = temp_x[index]

                    # get y values and normalize it
                    for index in range(len(temp_y)):
                        mean = all_mean_stddev[k][1][0][index]
                        stddev = all_mean_stddev[k][1][1][index]
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
                jsonFile = open(data_dir_target / subdir / file, "w+")
                jsonFile.write(json.dumps(data))
                jsonFile.close()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path_to_json_dir = sys.argv[1]
    else:
        path_to_json_dir = r"C:\Users\Asdf\Downloads\How2Sign_samples\openpose_output\json_testy_own_values"
    norm = Normalize(path_to_json_dir)
    start_time = time.time()
    norm.main_normalize()
    print("--- %s seconds ---" % (time.time() - start_time))
