"""norm_cent.py: normalize over all files from each folder of an directory
- read keypoints from each folder from an directory and write them into a dictionary
- compute the mean and stdev of each value
- use the mean and stdev to normalize the data and write them into new json, repeat for all folders
"""

import json
import numpy as np
import os
import statistics
from pathlib import Path
import sys
import time


class Normalize:

    def __init__(self, path_to_json_dir, path_to_target_dir):
        self.path_to_json = path_to_json_dir
        self.path_to_target_dir = path_to_target_dir

    def normalize(self):
        data_dir_origin, data_dir_target, subdirectories = self.create_folders()
        all_mean_stdev, keys = self.compute_mean_stdev(data_dir_origin, data_dir_target, subdirectories)
        self.normalize_write(all_mean_stdev, data_dir_origin, data_dir_target, keys, subdirectories)

    def create_folders(self):
        # get subdirectories of the path
        subdirectories = [x[1] for x in os.walk(self.path_to_json)]
        data_dir_origin = Path(self.path_to_json)
        subdirectories = subdirectories[0]

        if self.path_to_target_dir == "":
            data_dir_target = data_dir_origin.parent / str(data_dir_origin.name + "_normalized")
        else:
            data_dir_target = Path(self.path_to_target_dir)

        # create new target directory, the fils will be saved there
        if not os.path.exists(data_dir_target):
            os.makedirs(data_dir_target)

        for subdir in subdirectories:
            if not os.path.exists(data_dir_target / subdir):
                os.makedirs(data_dir_target / subdir)

        return data_dir_origin, data_dir_target, subdirectories

    def compute_mean_stdev(self, data_dir_origin, data_dir_target, subdirectories):
        # use keys of openpose here
        keys = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
        all_mean_stdev = {}  # holds means and stdev of each directory, one json file per directory
        once = 1
        all_files = {'all': {}}
        mean_stdev_x = []
        mean_stdev_y = []

        for subdir in subdirectories:
            print("Reading files from %s" % subdir)
            json_files = [pos_json for pos_json in os.listdir(data_dir_origin / subdir)
                          if pos_json.endswith('.json')]

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
        print("Files read, computing mean and stdev")

        for k in keys:
            print(np.array(all_files['all'][k]['x']).T.tolist())
            for list in np.array(all_files['all'][k]['x']).T.tolist():
                if "Null" in list:
                    mean_stdev_x.append(["Null", "Null"])
                else:
                    list = [float(item) for item in list]
                    mean_stdev_x.append([np.mean(list), statistics.pstdev(list)])

            for list in np.array(all_files['all'][k]['y']).T.tolist():
                if "Null" in list:
                    mean_stdev_y.append(["Null", "Null"])
                else:
                    list = [float(item) for item in list]
                    mean_stdev_y.append([np.mean(list), statistics.pstdev(list)])

            all_mean_stdev[k] = [np.array(mean_stdev_x).T.tolist(), np.array(mean_stdev_y).T.tolist()]

        f = open(data_dir_target / "dir_mean_stdev.json", "w")
        f.write(json.dumps(all_mean_stdev))
        f.close()
        print("Normalizing...")
        return all_mean_stdev, keys

    def normalize_write(self, all_mean_stdev, data_dir_origin, data_dir_target, keys, subdirectories):
        # use mean and stdev to compute values for the json files
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
                        mean_x = all_mean_stdev[k][0][0][index]
                        stdev_x = all_mean_stdev[k][0][1][index]

                        mean_y = all_mean_stdev[k][1][0][index]
                        stdev_y = all_mean_stdev[k][1][1][index]

                        if str(stdev_x) == "Null":
                            temp_x[index] = temp_x[index]
                        elif float(stdev_x) == 0:
                            temp_x[index] = temp_x[index]
                        else:
                            temp_x[index] = (temp_x[index] - float(mean_x)) / float(stdev_x)

                        if str(stdev_y) == "Null":
                            temp_y[index] = temp_y[index]
                        elif float(stdev_y) == 0:
                            temp_y[index] = temp_y[index]
                        else:
                            temp_y[index] = (temp_y[index] - float(mean_y)) / float(stdev_y)

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
    # origin json files directory
    if len(sys.argv) > 1:
        path_to_json_dir = sys.argv[1]
    else:
        print("set json file directory")

    # target directory
    path_to_target_dir = ""
    if len(sys.argv) > 2:
        path_to_target_dir = sys.argv[2]

    norm = Normalize(path_to_json_dir, path_to_target_dir)
    start_time = time.time()
    norm.normalize()
    print("--- %s seconds ---" % (time.time() - start_time))
