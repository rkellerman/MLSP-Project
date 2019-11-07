import numpy as np
from skimage import color
import cv2
from scipy.stats import itemfreq
from skimage.feature import local_binary_pattern
import copy as cp
import matplotlib.pyplot as plt


'''
Get the histogram of each texture of each superpixel
'''
test_image_file = 'HighwayDriving/Train/TrainSeq10/image/TrainSeq10_RGB_Image_0010.png'
test_image_texture = 'grad_lbp_2.png'

image = plt.imread(test_image_file)
plt.imshow(image)
plt.show()
seg = np.loadtxt(open("segments.csv", "rb")) + 1

histo = np.zeros(10)
print(histo)
for i in range(seg.max().astype(int)):
    i += 1
    print(i)
    single_seg = cp.copy(seg)
    single_seg[single_seg != i] = -1
    single_seg[single_seg > 0] = 1
    img = cp.copy(image)

    img = (img + 1) * np.stack([single_seg, single_seg, single_seg], axis=2)
    img -= 1
    hist, _ = np.histogram(img[img >= 0], range=(0, 1))

    histo = np.row_stack([histo, hist])
    #print(histo)
    img[img < 0] = 0
    #plt.imshow(img)
    #plt.show()
histo = histo[1:, :]
print(histo)
