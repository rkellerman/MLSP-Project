import skimage.segmentation as segmentation
import skimage.color as color
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from sklearn.neighbors import KNeighborsClassifier
from metric_learn import MMC_Supervised
from metric_learn import Covariance


def binary_mask(labeled_image, include_lane=False):

	mask = np.all(labeled_image == 128/255.0, axis=2)

	if include_lane:
		mask1 = labeled_image[:,:,0] == 1.0
		mask2 = labeled_image[:,:,1] == 1.0
		mask3 = labeled_image[:,:,2] == 0.0

		lane_mask = np.all(np.stack([mask1, mask2, mask3], axis=2), axis=2)
		mask = np.any(np.stack([mask, lane_mask], axis=2), axis=2)


	#plt.imshow(mask)
	#plt.show()

	return mask

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

if __name__ == '__main__':

    features_train = np.zeros((1,3))
    labels_train = np.zeros((1))

    for i in range(2):
        for j in range(60):
            if j >= 1:
                break
            image_file = 'HighwayDriving/Train/TrainSeq' + str(i).zfill(2) + '/image/TrainSeq' + str(i).zfill(2) + '_RGB_Image_' + str(j).zfill(4) + '.png'
            labeled_image_file = 'HighwayDriving/Train/TrainSeq' + str(i).zfill(2) + '/label/TrainSeq' + str(i).zfill(2) + '_ColorLabel_' + str(j).zfill(4) + '.png'




            image = plt.imread(image_file)
            image_gray = rgb2gray(image)
            labeled_image = plt.imread(labeled_image_file)

            true_mask = binary_mask(labeled_image, include_lane=True)

            plt.imshow(image_gray, cmap='gray')
            plt.show()

            print(image.shape)

            image /= 255.0




            feature_image = image.reshape(image.shape[0]*image.shape[1], image.shape[2])

            print(feature_image.shape)

            image *= 255.0

            numSegments = 300
			# apply SLIC and extract (approximately) the supplied number
			# of segments
            segments = segmentation.slic(image_gray*255.0, n_segments = numSegments, sigma = 5)

            print(segments)
            print(np.max(segments))

            fig = plt.figure("Ground Truth")
            ax = fig.add_subplot(1, 1, 1)
			
            ax.imshow(segmentation.mark_boundaries(color.label2rgb(true_mask, image_gray, kind='overlay'), true_mask))
            plt.show()


            fig = plt.figure("Ground Truth + Superpixels")
            ax = fig.add_subplot(1, 1, 1)
			
            ax.imshow(segmentation.mark_boundaries(color.label2rgb(true_mask, image_gray, kind='overlay'), segments))
            plt.show()

			

			# loop over the unique segment values

            unique_segs = np.unique(segments)
            num_segs = np.size(unique_segs)
                
            '''
                Feature Extraction
            '''
            # settings for LBP
            #radius = 2
            #n_points = 8 * radius
            
            segment_features = np.zeros((num_segs,3))
            for seg_idx in range(num_segs):
                pos_pxl = np.argwhere(segments==unique_segs[seg_idx])
                num_pxl = np.size(pos_pxl,0)
                height_pxl = np.sum(pos_pxl[0,:])
                rgb_pxl = np.zeros((1,3))
                for i in range(num_pxl):
                    rgb_pxl = rgb_pxl+image[pos_pxl[i,:][0],pos_pxl[i,:][1],:]
                segment_features[seg_idx,:] = rgb_pxl/num_pxl
            # label each superpixel with reference to binary mask     
            segment_labels = np.zeros(len(unique_segs)) 
            
            cluster_mask = np.zeros(image_gray.shape[:2], dtype = "uint8")

            for (i, segVal) in enumerate(unique_segs):
				# construct a mask for the segment
                seg_mask = np.zeros(image_gray.shape[:2], dtype = "uint8")
                seg_mask[segments == segVal] = 1.0

                if np.sum(np.bitwise_and(seg_mask, true_mask)) / np.sum(seg_mask) >= 0.5:
                    segment_labels[i] = 1
                    cluster_mask[segments == segVal] = 1
				

			# show the output of SLIC
            fig = plt.figure("Derived Labels for Superpixels from Ground Truth")
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(segmentation.mark_boundaries(color.label2rgb(cluster_mask, image_gray, kind='overlay'), segments))
            plt.show()

			# show the output of SLIC
            fig = plt.figure("New overlay using only labeled superpixels")
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(segmentation.mark_boundaries(color.label2rgb(cluster_mask, image_gray, kind='overlay'), cluster_mask))
            plt.show()
            
            features_train = np.append(features_train,segment_features,axis=0)
            labels_train = np.append(labels_train,segment_labels,axis=0)
    
'''
    Train Distance Metric Learning model
'''
#mmc = MMC_Supervised(max_iter=1000, convergence_threshold=1e-6)
#mmc_metric = mmc.fit(features_train,labels_train)
#features_train = mmc_metric.transform(features_train)
knn = KNeighborsClassifier(n_neighbors=3,weights='distance')
knn.fit(features_train, labels_train)


'''
    Testing
    
'''
image_file = 'HighwayDriving/Test/TestSeq04/image/TestSeq04_RGB_Image_0001.png'



image = plt.imread(image_file)
image_gray = rgb2gray(image)
plt.imshow(image_gray, cmap='gray')
plt.show()

print(image.shape)

image /= 255.0


feature_image = image.reshape(image.shape[0]*image.shape[1], image.shape[2])

print(feature_image.shape)



image *= 255.0

numSegments = 300

# apply SLIC and extract (approximately) the supplied number
# of segments
segments = segmentation.slic(image_gray*255.0, n_segments = numSegments, sigma = 5)

print(segments)
print(np.max(segments))

			
# loop over the unique segment values

unique_segs = np.unique(segments)
num_segs = np.size(unique_segs)

# construct feature for each superpixel as the mean rgb value
segment_features = np.zeros((num_segs,3))
for seg_idx in range(num_segs):
    pos_pxl = np.argwhere(segments==unique_segs[seg_idx])
    num_pxl = np.size(pos_pxl,0)
    rgb_pxl = np.zeros((1,3))
    for i in range(num_pxl):
        rgb_pxl = rgb_pxl+image[pos_pxl[i,:][0],pos_pxl[i,:][1],:]
    segment_features[seg_idx,:] = rgb_pxl/num_pxl

labels_pred = knn.predict(segment_features)

cluster_mask = np.zeros(image_gray.shape[:2], dtype = "uint8")
for (i, segVal) in enumerate(unique_segs):
    if labels_pred[i] == 1: cluster_mask[segments == segVal] = 1
				

# show the output of SLIC
fig = plt.figure("Derived Labels for Superpixels from Ground Truth")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(segmentation.mark_boundaries(color.label2rgb(cluster_mask, image_gray, kind='overlay'), segments))
plt.show()

# show the output of SLIC
fig = plt.figure("New overlay using only labeled superpixels")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(segmentation.mark_boundaries(color.label2rgb(cluster_mask, image_gray, kind='overlay'), cluster_mask))
plt.show()

features_train = np.append(features_train,segment_features,axis=0)
labels_train = np.append(labels_train,segment_labels,axis=0)
