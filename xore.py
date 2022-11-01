# create masked images from inDir files to outDir folder
# operation - pixel(Y,X) = pixel(Y,X) XOR pixel(Y+shift_y,X+shift_x)

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

inDir   = "C:/WORK/DL0/PetImages/"
outDir  = "C:/WORK/DL0/mask20/"

def get_masked (orig, leva = 1, shift_y = 1, shift_x = 1):
    orig = orig.copy()                      # make a copy because original is read-only
    hei = len(orig)                         # image height, pixels
    wid = len(orig[0])                      # width
    orig[:hei-shift_y, :wid-shift_x] = \
        (orig[: hei-shift_y, : wid-shift_x] ^ \
         orig[shift_y : hei, shift_x : wid] // leva \
         ) & 0xFF                           # divide mask pixel RGB values by leva
    return orig

for petDir in ["Cat", "Dog"]:
    petFiles = os.listdir(inDir + petDir)
    for ind in range(len(petFiles)):
        try:
            ori = mpimg.imread(inDir + petDir + '/' + petFiles[ind]) # read pet image
            shift = get_masked(ori, 2)                      # create masked image, mask denominator here - 2
            im = Image.fromarray(shift)
            im.save(outDir + petDir + '/' + petFiles[ind])    # save
            print(petDir, "\t ", petFiles[ind], "\t ", ind)
        except:
            print(petDir, "\t ", petFiles[ind], "\t ", ind, " broken file")

exit(0)
