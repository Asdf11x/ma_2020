"""
trash_test.py:

die idee war die pixel eines bildes prozentual schwarz zu färben, um informationen zu verschleiern.
Damit sollte man einen Eindruck von der Qualitaet von OpenPose, welches die Videos auch nur zu einem gewissen
Prozentsatz erkennt.

Ergebnis: Die Bilder von How2Sign sind qualitativ ganz gut und selbst bei raten von 99% schwarzer Pixel,
erkennt man das Motiv noch

Falls man Bilder von Texten nimmt, erkentn man teils bei 90% schon nichts mehr und sobald die Qualität oder Auflösung
geringer sind, dan erkentn mkan auch schond eutlich früher nichts mehr zB bei 20%

--> Also man kann durch die Bilduqalität das Ergebnis massgeblich beeinflussen


06.08.20
"""
import cv2
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt 
import random

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

img = cv2.imread(r"C:\Users\Asdf\Downloads\Master\ISLR\train_mod\A001\001 - Kopie.jpg")
# img = image_resize(img, width=480, height=480)

# y, x
img = img[0:480, 80:560]

# height, width, channels = img.shape

# for h in range(height):
#     for w in range(width):
#         percentage = random.randint(0, 100)
#         if percentage < 50:
#             img[h,w] = 0
cv2.imwrite(r"C:\Users\Asdf\Downloads\Master\ISLR\train_mod\A001\001 - Kopie.jpg",img)
cv2.imshow("image", img)
cv2.waitKey(0)