import numpy as np
import joblib
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import skimage.segmentation as segmentation
import copy as cp
import skimage.color as color
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import auc


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


knn = joblib.load("knn_model_ms.m")
accu = 0
for i in range(5):
    mask = [0]
    seg_label_ = [0]
    for j in range(60):
        test_image_file = 'HighwayDriving/Test/TestSeq' + str(i).zfill(2) + '/image/TestSeq' \
                          + str(i).zfill(2) + '_RGB_Image_' + str(j).zfill(4) + '.png'
        labeled_test_image_file = 'HighwayDriving/Test/TestSeq' + str(i).zfill(2) + '/label/TestSeq' \
                                  + str(i).zfill(2) + '_ColorLabel_' + str(j).zfill(4) + '.png'

        image = plt.imread(test_image_file)
        labeled_image = plt.imread(labeled_test_image_file)
        image_gray = rgb2gray(image)
        radius = 20
        no_points = 1 * radius
        image_lbp = local_binary_pattern(image_gray, no_points, radius, method='uniform')
        image_lbp = np.abs(image_lbp)
        image_lbp /= image_lbp.max()

        image_1 = image[:, :, 0] * image_lbp
        image_2 = image[:, :, 1] * image_lbp
        image_3 = image[:, :, 2] * image_lbp

        image = image.astype(np.float)
        seg = segmentation.slic(image, n_segments=300, sigma=5)
        #print(seg)
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

            mean_1 = np.mean(img_1[img_1 >= 0])
            std_1 = np.std(img_1[img_1 >= 0])
            mean_2 = np.mean(img_2[img_2 >= 0])
            std_2 = np.std(img_2[img_2 >= 0])
            mean_3 = np.mean(img_3[img_3 >= 0])
            std_3 = np.std(img_3[img_3 >= 0])

            feature = np.hstack([mean_1, std_1, mean_2, std_2, mean_3, std_3])
            #print(feature)
            histo = np.row_stack([histo, feature])
            #img[img < 0] = 0
            #plt.imshow(img, cmap='gray')
            #plt.show()
        histo = histo[1:, :]
        scaler = MinMaxScaler()
        scaler.fit(histo)
        data_test = scaler.transform(histo)
        knn_predict = knn.predict(histo)

        seg_label2 = cp.copy(seg)+1
        #print(seg_label2)
        #print(knn_predict)

        for l in range(seg.max().astype(int)):
            knn_label = knn_predict[l].astype(int)#kmeans.labels_[i]
            l += 2
            #seg_label = cp.copy(seg)
            seg_label2[seg_label2 == l] = knn_label#[i - 3]

        mask_1 = binary_mask(labeled_image, include_lane=True)
        mask = np.vstack([mask, mask_1.astype(int).reshape(mask_1.shape[0] * mask_1.shape[1], 1)])
        seg_label_ = np.vstack([seg_label_, seg_label2.reshape(seg_label2.shape[0] * seg_label2.shape[1], 1)])
        accu += 1 - np.sum(abs(seg_label2-mask_1.astype(int)))/(seg_label2.shape[0] * seg_label2.shape[1])
        print(1 - np.sum(abs(seg_label2-mask_1.astype(int)))/(seg_label2.shape[0] * seg_label2.shape[1]))

    mask = mask[1:]
    seg_label_ = seg_label_[1:]
    fpr, tpr, thresholds = metrics.roc_curve(mask, seg_label_)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, marker='o', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.legend(loc = "lower right")
plt.show()
#print(accu / (4 * 60))
fig = plt.figure("Superpixels")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(segmentation.mark_boundaries(color.label2rgb(seg_label2, image, kind='overlay'), seg_label2))
plt.axis('off')
plt.show()
print(accu / (5*60))
