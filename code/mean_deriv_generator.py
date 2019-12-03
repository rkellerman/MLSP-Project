import numpy as np
import cv2
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

        #if j > 0:
        #    break
        #if i != 10:
        #    break

        print(i)
        image_file = 'HighwayDriving/Train/TrainSeq' + str(i).zfill(2) + '/image/TrainSeq' + str(i).zfill(
            2) + '_RGB_Image_' + str(j).zfill(4) + '.png'
        labeled_image_file = 'HighwayDriving/Train/TrainSeq' + str(i).zfill(2) + '/label/TrainSeq' + str(
            i).zfill(2) + '_ColorLabel_' + str(j).zfill(4) + '.png'

        image = plt.imread(image_file)
        image_gray = rgb2gray(image)
        labeled_image = plt.imread(labeled_image_file)
        radius = 20
        no_points = 1 * radius
        image_lbp = local_binary_pattern(image_gray, no_points, radius, method='uniform')
        image_lbp = np.abs(image_lbp)
        image_lbp /= image_lbp.max()

        image_1 = image[:, :, 0] * image_lbp
        image_2 = image[:, :, 1] * image_lbp
        image_3 = image[:, :, 2] * image_lbp
        # image_1 = edge
        # image_2 = edge
        # image_3 = edge
        #image_lbp = np.stack([image_1, image_2, image_3], axis=2)
        #hist = cv2.calcHist([image_lbp], [0, 1, 2], None, [0, 255], [0, 255])
        #print(image_lbp)
        #plt.imshow(image_lbp)
        #plt.show()

        image = image.astype(np.float)
        seg = segmentation.slic(image, n_segments=300, sigma=5)

        histo = np.zeros(6)
        seg += 1
        for k in range(seg.max().astype(int)):
            k += 1
            single_seg = cp.copy(seg)
            single_seg[single_seg != k] = -1
            single_seg[single_seg > 0] = 1
            img_1 = cp.copy(image_1)
            img_2 = cp.copy(image_2)
            img_3 = cp.copy(image_3)

            img_1 = (img_1 + 1) * single_seg
            img_2 = (img_2 + 1) * single_seg
            img_3 = (img_3 + 1) * single_seg

            img_1 -= 1
            img_2 -= 1
            img_3 -= 1
            #print(img_1[img_1 >= 0])
            mean_1 = np.mean(img_1[img_1 >= 0])
            std_1 = np.std(img_1[img_1 >= 0])
            mean_2 = np.mean(img_2[img_2 >= 0])
            std_2 = np.std(img_2[img_2 >= 0])
            mean_3 = np.mean(img_3[img_3 >= 0])
            std_3 = np.std(img_3[img_3 >= 0])

            feature = np.hstack([mean_1, std_1, mean_2, std_2, mean_3, std_3])
            feature = feature / sum(feature)
            #print(feature)
            histo = np.row_stack([histo, feature])
            #print(histo)
            #img[img < 0] = 0
        histo = histo[1:, :]
        print(histo)
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
            i).zfill(2) + '_mean_std_' + str(j).zfill(4) + '.csv', histo)
        
