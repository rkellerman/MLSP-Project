import numpy as np
import skimage.segmentation as segmentation
from skimage.feature import local_binary_pattern
import copy as cp
import matplotlib.pyplot as plt

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

for i in range(15):
    for j in range(60):

        print(i)
        image_file = 'HighwayDriving/Train/TrainSeq' + str(i).zfill(2) + '/image/TrainSeq' + str(i).zfill(
            2) + '_RGB_Image_' + str(j).zfill(4) + '.png'
        labeled_image_file = 'HighwayDriving/Train/TrainSeq' + str(i).zfill(2) + '/label/TrainSeq' + str(
            i).zfill(2) + '_ColorLabel_' + str(j).zfill(4) + '.png'

        image = plt.imread(image_file)
        image_gray = rgb2gray(image)
        labeled_image = plt.imread(labeled_image_file)
        radius = 5
        no_points = 8 * radius
        image_lbp = local_binary_pattern(image_gray, no_points, radius, method='uniform')
        image_lbp = np.abs(image_lbp)
        image_lbp /= image_lbp.max()

        image = image.astype(np.float)
        seg = segmentation.slic(image, n_segments=300, sigma=5)

        histo = np.zeros(10)
        seg += 1
        for k in range(seg.max().astype(int)):
            k += 1
            single_seg = cp.copy(seg)
            single_seg[single_seg != k] = -1
            single_seg[single_seg > 0] = 1
            img = cp.copy(image_lbp)

            img = (img + 1) * single_seg

            img -= 1
            hist, _ = np.histogram(img[img >= 0], range=(0, 1))

            histo = np.row_stack([histo, hist])
            img[img < 0] = 0
        histo = histo[1:, :]

        true_mask = binary_mask(labeled_image, include_lane=True)
        unique_segs = np.unique(seg)
        segment_labels = np.zeros(len(unique_segs))
        cluster_mask = np.zeros(image_gray.shape[:2], dtype="uint8")

        for (l, segVal) in enumerate(unique_segs):
            # construct a mask for the segment
            seg_mask = np.zeros(image_gray.shape[:2], dtype="uint8")
            seg_mask[seg == segVal] = 1.0

            if np.sum(np.bitwise_and(seg_mask, true_mask)) / np.sum(seg_mask) >= 0.5:
                segment_labels[l] = 1
                cluster_mask[seg == segVal] = 1
        histo = np.column_stack([histo, segment_labels])#Set segment_labels with histo of each segment

        np.savetxt('HighwayDriving/Train/TrainSeq' + str(i).zfill(2) + '/histo/TrainSeq' + str(
            i).zfill(2) + '_Gray_histo_' + str(j).zfill(4) + '.csv', histo)
