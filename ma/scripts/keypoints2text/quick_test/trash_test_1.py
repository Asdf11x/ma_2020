"""

"""

from pathlib import Path
from shutil import copyfile
import os
import cv2
import numpy as np

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

path_to_all =r"C:\Users\Asdf\Downloads\Master\ISLR\train"
path_to_new = r"C:\Users\Asdf\Downloads\Master\ISLR\train_mod"

pathl_to_all = Path(path_to_all)
pathl_to_new = Path(path_to_new)

# print(next(os.walk(path_to_all))[1])



for directory in next(os.walk(pathl_to_new))[1]:
    paaath = pathl_to_new / str(directory +"\\"+ directory[1:]+".jpg")

    # paaath = str(paaath).replace("\\", "\\\\")
    print(paaath)

    # print(os.path.join(str(pathl_to_new / directory), str(directory[1:]+".jpg")))
    # repl = "C:\Users\Asdf\Downloads\Master\I
    # SLR\train_mod\A001\001.jpg"
    # repl = repl.replace("\\", "\\\\")
    # print(os.path.join("", paaath))
    # for file in os.walk(str(pathl_to_all)):
    #     print(file)
        # for file_entry in file[2]:
            # print(str(str(directory+file_entry)[:-4] + "," + str(directory+file_entry).lower()[:-7]) )
            # os.makedirs(os.path.dirname(pathl_to_new / str(str(directory+file_entry)[:-4] + "\\" + str(file_entry))), exist_ok=True)
            # copyfile(str(pathl_to_all / directory / file_entry), pathl_to_new / str(str(directory+file_entry)[:-4] + "\\" + str(file_entry)))
    # paaath = str(paaath).replace("a", "b")
    # paaath = str(paaath)
    # img = cv2.imread("C:\Users\Asdf\Downloads\Master\ISLR\train_mod\A001\001.jgp")
    blank_image = np.zeros((640, 640, 3), np.uint8)
    img = cv2.imread(str(paaath))
    height, width, channels = img.shape
    print(width)
    if width == 1920:
        img = image_resize(img, width=640)
        blank_image[0:0 + 362, 0:0 + 640] = img
    elif height == 1920:
        img = image_resize(img, height=640)
        blank_image[0:0 + 640, 0:0 + 362] = img
    else:
        # img = image_resize(img, width=640)
        blank_image[0:0 + 480, 0:0 + 640] = img

    # cv2.imshow("as", blank_image)
    # print(img)
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    # img = img[0:480, 80:560]
    cv2.imwrite(str(paaath), blank_image)