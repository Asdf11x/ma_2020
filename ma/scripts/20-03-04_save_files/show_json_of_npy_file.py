import numpy as np
import sys
import json
import os
from pathlib import Path
import psutil
old = np.load
np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)

path_to_numpy_files = Path(sys.argv[1])
# path_to_numpy_files = Path(r"/home/...")
numpy_files = [pos_json for pos_json in os.listdir(path_to_numpy_files) if pos_json.endswith('.npy')]

process = psutil.Process(os.getpid())
print("Current memory usage: %s megabytes" % str(process.memory_info().rss / 1000000))  # divided to get mb

def load(path_to_numpy_file, file_name):
    # load from .npy file
    # print("load from %s" % path_to_numpy_file)
    process = psutil.Process(os.getpid())
    print("Current memory usage: %s megabytes" % str(process.memory_info().rss / 1000000))  # divided to get mb
    all_files_dictionary = np.load(path_to_numpy_file).item()
    print("Save to %s" % file_name)
    process = psutil.Process(os.getpid())
    print("Current memory usage: %s megabytes" % str(process.memory_info().rss / 1000000))  # divided to get mb
    file_name = os.path.splitext(file_name)[0] + ".json"

    jsonFile = open(file_name, "w+")
    jsonFile.write(json.dumps(all_files_dictionary))
    jsonFile.close()


for file_name in numpy_files:
    print("Loading %s" % file_name)
    load(path_to_numpy_files / file_name, file_name)
