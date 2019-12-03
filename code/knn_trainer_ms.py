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
from sklearn.preprocessing import MinMaxScaler
import joblib


def binary_mask(labeled_image, include_lane=False):
    mask = np.all(labeled_image == 128 / 255.0, axis=2)

    if include_lane:
        mask1 = labeled_image[:, :, 0] == 1.0
        mask2 = labeled_image[:, :, 1] == 1.0
        mask3 = labeled_image[:, :, 2] == 0.0

        lane_mask = np.all(np.stack([mask1, mask2, mask3], axis=2), axis=2)
        mask = np.any(np.stack([mask, lane_mask], axis=2), axis=2)


    return mask


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


points = np.zeros(6)
labels = np.array([0])
#print(labels.shape)

for i in range(15):
    for j in range(60):

        print(i)

        histo_file = 'HighwayDriving/Train/TrainSeq' + str(i).zfill(2) + '/histo/TrainSeq' + str(
            i).zfill(2) + '_mean_std_' + str(j).zfill(4) + '.csv'

        histo_label = np.loadtxt(open(histo_file, "rb"))
        points = np.row_stack([points, histo_label[:, 0:6]])
        labels = np.hstack([labels, histo_label[:, 6]])

points = points[1:, :]
labels = labels[1:]
scaler = MinMaxScaler()
scaler.fit(points)
data_train = scaler.transform(points)

print(labels.astype(int).shape)
knn = neighbors.KNeighborsClassifier().fit(data_train, labels.astype(int))
joblib.dump(knn,"knn_model_ms.m")
