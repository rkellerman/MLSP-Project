import numpy as np
from skimage import color
import sklearn.cluster as cluster
import skimage.segmentation as segmentation
import cv2
from scipy.stats import itemfreq
from skimage.feature import local_binary_pattern
import copy as cp
import matplotlib.pyplot as plt
from sklearn import neighbors
import joblib


def binary_mask(labeled_image, include_lane=False):
    mask = np.all(labeled_image == 128 / 255.0, axis=2)

    if include_lane:
        mask1 = labeled_image[:, :, 0] == 1.0
        mask2 = labeled_image[:, :, 1] == 1.0
        mask3 = labeled_image[:, :, 2] == 0.0

        lane_mask = np.all(np.stack([mask1, mask2, mask3], axis=2), axis=2)
        mask = np.any(np.stack([mask, lane_mask], axis=2), axis=2)

    # plt.imshow(mask)
    # plt.show()

    return mask


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


points = np.zeros(10)
labels = np.array([0])
#print(labels.shape)

for i in range(15):
    for j in range(60):

        print(i)
        #image_file = 'HighwayDriving/Train/TrainSeq' + str(i).zfill(2) + '/image/TrainSeq' + str(i).zfill(
        #    2) + '_RGB_Image_' + str(j).zfill(4) + '.png'
        #labeled_image_file = 'HighwayDriving/Train/TrainSeq' + str(i).zfill(2) + '/label/TrainSeq' + str(
        #    i).zfill(2) + '_ColorLabel_' + str(j).zfill(4) + '.png'
        histo_file = 'HighwayDriving/Train/TrainSeq' + str(i).zfill(2) + '/histo/TrainSeq' + str(
            i).zfill(2) + '_Gray_histo_' + str(j).zfill(4) + '.csv'

        #histo_file = 'HighwayDriving/Train/TrainSeq00/histo/TrainSeq00_Gray_histo_0000.csv'

        histo_label = np.loadtxt(open(histo_file, "rb"))
        points = np.row_stack([points, histo_label[:, 0:10]])
        labels = np.hstack([labels, histo_label[:, 10]])
        #print(points)
        #print(labels)
        #image = plt.imread(image_file)
        #image_gray = rgb2gray(image)
        #labeled_image = plt.imread(labeled_image_file)

        #true_mask = binary_mask(labeled_image, include_lane=True)
        #unique_segs = np.unique(seg)
        #segment_labels = np.zeros(len(unique_segs))
        #cluster_mask = np.zeros(image_gray.shape[:2], dtype="uint8")

        #for (i, segVal) in enumerate(unique_segs):
            # construct a mask for the segment
        #    seg_mask = np.zeros(image_gray.shape[:2], dtype="uint8")
        #    seg_mask[seg == segVal] = 1.0

        #    if np.sum(np.bitwise_and(seg_mask, true_mask)) / np.sum(seg_mask) >= 0.5:
        #       segment_labels[i] = 1
        #        cluster_mask[seg == segVal] = 1


print(labels.astype(int).shape)
knn = neighbors.KNeighborsClassifier().fit(points[1:, :], labels[1:].astype(int))
joblib.dump(knn, 'knn_model.m')