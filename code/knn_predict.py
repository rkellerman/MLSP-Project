import numpy as np
import joblib
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import skimage.segmentation as segmentation
import copy as cp
import skimage.color as color


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


knn = joblib.load("knn_model.m")
test_image_file = 'HighwayDriving/Test/TestSeq00/image/TestSeq00_RGB_Image_0020.png'

image = plt.imread(test_image_file)
image_gray = rgb2gray(image)
radius = 5
no_points = 8 * radius
image_lbp = local_binary_pattern(image_gray, no_points, radius, method='uniform')
image_lbp = np.abs(image_lbp)
image_lbp /= image_lbp.max()

image = image.astype(np.float)
seg = segmentation.slic(image, n_segments=300, sigma=5)
#print(seg)
histo = np.zeros(10)
seg += 1
for k in range(seg.max().astype(int)):
    k += 1
    single_seg = cp.copy(seg)
    single_seg[single_seg != k] = -1
    single_seg[single_seg > 0] = 1
    img = cp.copy(image_lbp)

    img = (img + 1) * single_seg#np.stack([single_seg, single_seg, single_seg], axis=2)

    img -= 1
    #print(img)
    hist, _ = np.histogram(img[img >= 0], range=(0, 1))

    histo = np.row_stack([histo, hist])
    img[img < 0] = 0
    #plt.imshow(img, cmap='gray')
    #plt.show()
histo = histo[1:, :]
knn_predict = knn.predict(histo)

seg_label2 = cp.copy(seg)+1
print(seg_label2)
print(knn_predict)

for i in range(seg.max().astype(int)):
    knn_label = knn_predict[i].astype(int)#kmeans.labels_[i]
    i += 2
    #seg_label = cp.copy(seg)
    seg_label2[seg_label2 == i] = knn_label#[i - 3]
    #single_seg[single_seg > 0] = 1

print(seg_label2)
fig = plt.figure("Superpixels")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(segmentation.mark_boundaries(color.label2rgb(seg_label2, image, kind='overlay'), seg_label2))
plt.show()