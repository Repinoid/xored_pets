import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

dl_folder = "C:/WORK/DL0/"                  # 
cats_dir = dl_folder + "vmesto/Cat/"        # cat's images
dogs_dir = dl_folder + "vmesto/Dog/"        # dogs

cats = os.listdir(cats_dir)
dogs = os.listdir(dogs_dir)


def get_masked (orig, leva = 1, shift_y = 1, shift_x = 1):
    orig = orig.copy()                      # make a copy because original is read-only
    hei = len(orig)                         # image height, pixels
    wid = len(orig[0])                      # width
    for y in range(hei - shift_y):          # shift_y - vertical shift of masking image
        for x in range(wid - shift_x):      # shift_x - horizontal shift of masking image
            orig[y, x] = ( orig[y, x] ^ (orig[y + shift_y, x + shift_x] // leva) ) & 0xFF   # XORing
                                            # leva - denominator of masking pixel RGB values
    return orig


for ind in range(len(cats)):
    print("Cat ", ind)
    ori = mpimg.imread(cats_dir + cats[ind])        # read pet image
    shift = get_masked(ori, 2)                      # create masked image, mask denominator here - 2
    im = Image.fromarray(shift)
    im.save(dl_folder + "mask20/Cat/" +cats[ind])    # save

for ind in range(len(dogs)):
    print("Dog ", ind)
    ori = mpimg.imread(dogs_dir + dogs[ind])
    shit = get_masked(ori, 2)
    im = Image.fromarray(shit)
    im.save(dl_folder + "mask20/Dog/" + dogs[ind])
