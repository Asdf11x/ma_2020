"""rescale.py: rescale skeletons towards a mean and draw new

"""

import json
import os
import sys
import time
from pathlib import Path


class Centralize:

    def __init__(self, path_to_json_dir, path_to_target_dir):
        self.path_to_json = path_to_json_dir
        self.path_to_target_dir = path_to_target_dir

    def centralize(self):
        # get subdirectories of the path
        os.walk(self.path_to_json)
        subdirectories = [x[1] for x in os.walk(self.path_to_json)]
        data_dir_origin = Path(self.path_to_json)
        subdirectories = subdirectories[0]

        # create new target directory, the files will be saved there
        if self.path_to_target_dir == "":
            data_dir_target = data_dir_origin.parent / str(data_dir_origin.name + "_centralized")
        else:
            data_dir_target = Path(self.path_to_target_dir)

        # create new target directory, the fils will be saved there
        if not os.path.exists(data_dir_target):
            os.makedirs(data_dir_target)

        for subdir in subdirectories:
            if not os.path.exists(data_dir_target / subdir):
                os.makedirs(data_dir_target / subdir)

        # used keys of openpose here
        keys = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
        folder_keys = {'pose_keypoints_2d': [], 'face_keypoints_2d': [], 'hand_left_keypoints_2d': [],
                       'hand_right_keypoints_2d': []}

        for subdir in subdirectories:
            json_files = [pos_json for pos_json in os.listdir(data_dir_origin / subdir)
                          if pos_json.endswith('.json')]

            all_files = {}
            once = 1
            # load files from one folder into dictionary
            for file in json_files:
                temp_df = json.load(open(data_dir_origin / subdir / file))
                # print(temp_df)
                all_files[file] = {}

                # init dictionaries & write x, y values into dictionary
                for k in keys:
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
                        neck_zero_y = all_files[file][k]['x'][0][0]
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
                                results_x.append(neck_zero_x - x_in_key[idx])

                            if y_in_key[idx] == 0:
                                results_y.append("Null")
                            else:
                                results_y.append(neck_zero_y - y_in_key[idx])

                        # add Null as legs
                        results_x += (['Null'] * 6)
                        results_y += (['Null'] * 6)

                        for idx in range(15, len(x_in_key[:19])):
                            if x_in_key[idx] == 0:
                                results_x.append('Null')
                            else:
                                results_x.append(neck_zero_x - x_in_key[idx])

                            if y_in_key[idx] == 0:
                                results_y.append("Null")
                            else:
                                results_y.append(neck_zero_y - y_in_key[idx])

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
                                results_x.append(neck_zero_x - x_in_key[idx])

                            if y_in_key[idx] == 0:
                                results_y.append("Null")
                            else:
                                results_y.append(neck_zero_y - y_in_key[idx])

                        values = []
                        for index in range(len(temp_c)):
                            values.append(results_x[index])
                            values.append(results_y[index])
                            values.append(temp_c[index])
                        temp_df['people'][0][k] = values

                # ## Save our changes to JSON file
                jsonFile = open(data_dir_target / subdir / file, "w+")
                jsonFile.write(json.dumps(temp_df))
                jsonFile.close()
            print("%s done" % subdir)


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

    norm = Centralize(path_to_json_dir, path_to_target_dir)
    start_time = time.time()
    norm.centralize()
    print("--- %s seconds ---" % (time.time() - start_time))
