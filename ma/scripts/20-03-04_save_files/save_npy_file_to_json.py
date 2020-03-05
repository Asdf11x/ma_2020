import numpy as np
import sys
import json
import os
from pathlib import Path

old = np.load
np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)

# path_to_numpy_file = sys.argv[1]
path_to_numpy_files = Path(r"C:\Users\Asdf\Downloads\How2Sign_samples\openpose_output\json_testy_1_saved_numpy\\")
numpy_files = [pos_json for pos_json in os.listdir(path_to_numpy_files) if pos_json.endswith('.npy')]

def load(path_to_numpy_file, file_name):
    # load from .npy file
    # print("load from %s" % path_to_numpy_file)

    all_files_dictionary = np.load(path_to_numpy_file).item()
    print("Save to %s" % file_name)

    file_name = os.path.splitext(file_name)[0] + ".json"

    jsonFile = open(file_name, "w+")
    jsonFile.write(json.dumps(all_files_dictionary))
    jsonFile.close()


for file_name in numpy_files:
    print("Loading %s" % file_name)
    load(path_to_numpy_files / file_name, file_name)
