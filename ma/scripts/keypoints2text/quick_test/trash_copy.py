"copy files from server to a structure for 2d_vis.yp"

from pathlib import Path
from shutil import copyfile
import os
from shutil import copyfile



path_to_all =r"C:\Users\Asdf\Downloads\Master\ISLR\train_mod_rendered"


pathl_to_all = Path(path_to_all)

index = 0

for item in os.listdir(pathl_to_all):
    if os.path.isdir(os.path.join(pathl_to_all, item)):
        json_file = path_to_all +"\\" +  item + "\\0_keypoints.json"
        copyfile(json_file, r"C:\Users\Asdf\Downloads\How2Sign_samples\openpose_output\json\00islr_a" + "\\" + str(index) + "_keypoints.json")
        index += 1
        print(json_file)

# for directory in next(os.walk(path_to_all))[1]:
#     for file in os.walk(str(pathl_to_all / directory)):
#         print(file)
        # json_file = file[0] + "\\0_keypoints.json"
        # print(json_file)
        # if os.path.exists(json_file):
        #     print(json_file)