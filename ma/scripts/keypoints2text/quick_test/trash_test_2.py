"""

"""

from pathlib import Path
from shutil import copyfile
import os
import cv2
import numpy as np
from random import randrange

content = []

with open(r"C:\Users\Asdf\Downloads\Master\ISLR\islr\3_linked_to_npy\islr_full_linked.txt", 'r') as file:
    for line in file:
        content.append(line.strip())

content = content[1:]
print(content)

content_val = content[::3]
del content[::3]
print(content)
print(content_val)


"""unnoetig
check_c = []

alphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y"]

dicc = {"a": [],"b": [],"c": [],"d": [],"e": [],"f": [],"g": [],"h": [],"i": [],"j": [],"k": [],"l": [],"m": [],"n": [],"o": [],"p": [],"q": [],"r": [],"s": [],"t": [],"u": [],"v": [],"w": [],"x": [],"y":[]}

for element in content:
    for char in alphabet:
        if char in element:
            dicc[char].append(element)
"""