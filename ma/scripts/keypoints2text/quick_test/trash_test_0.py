"""

"""

from pathlib import Path
from shutil import copyfile
import os

path_to_all =r"C:\Users\Asdf\Downloads\Master\ISLR\train"
path_to_new = r"C:\Users\Asdf\Downloads\Master\ISLR\train_mod"

pathl_to_all = Path(path_to_all)
pathl_to_new = Path(path_to_new)

# print(next(os.walk(path_to_all))[1])

for directory in next(os.walk(path_to_all))[1]:
    for file in os.walk(str(pathl_to_all / directory)):
        for file_entry in file[2]:
            print(str(str(directory+file_entry)[:-4] + "," + str(directory+file_entry).lower()[:-7]) )
            os.makedirs(os.path.dirname(pathl_to_new / str(str(directory+file_entry)[:-4] + "\\" + str(file_entry))), exist_ok=True)
            copyfile(str(pathl_to_all / directory / file_entry), pathl_to_new / str(str(directory+file_entry)[:-4] + "\\" + str(file_entry)))
        # print(file)
    # print(pathl_to_all / directory)



# print(os.walk(path_to_all))
# copyfile(src, dst)
