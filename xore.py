import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

cats = os.listdir("C:/WORK/DL0/vmesto/Cat")
dogs = os.listdir("C:/WORK/DL0/vmesto/Dog")


def get_masked (orig, leva = 1, shift_y = 1, shift_x = 1):
    orig = orig.copy()
    hei = len(orig)
    wid = len(orig[0])
    for y in range(hei - shift_y - 1):
        for x in range(wid - shift_x - 1):
            orig[y, x] = (orig[y, x] ^ orig[y + shift_y, x + shift_x] // leva) & 0xFF
    return orig


for ind in range(len(cats)):
    print("Cat ", ind)
    ori = mpimg.imread("C:/WORK/DL0/vmesto/Cat/" + cats[ind])
    shit = get_masked(ori, 2)
    im = Image.fromarray(shit)
    im.save("C:/WORK/DL0/mask2/Cat/"+cats[ind])

for ind in range(len(dogs)):
    print("Dog ", ind)
    ori = mpimg.imread("C:/WORK/DL0/vmesto/Dog/" + dogs[ind])
    shit = get_masked(ori, 2)
    im = Image.fromarray(shit)
    im.save("C:/WORK/DL0/mask2/Dog/"+dogs[ind])

#imgplot = plt.imshow(shit)
#plt.show()
