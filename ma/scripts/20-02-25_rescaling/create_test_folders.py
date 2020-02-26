import os

f = open("folders.txt", "r")
alist = [line.rstrip() for line in f]
test_folders = r"C:\Users\Asdf\Downloads\How2Sign_samples\openpose_output\test_folders"
for lines in alist:
    if not os.path.exists(test_folders + "\\" + lines):
        os.makedirs(test_folders + "\\" + lines)
