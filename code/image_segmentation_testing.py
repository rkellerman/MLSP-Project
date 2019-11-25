import skimage.segmentation as segmentation
import skimage.color as color
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2


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

	

	for i in range(15):
		for j in range(60):

			if j >= 1:
				break

			image_file = '../data/HighwayDriving/Train/TrainSeq' + str(i).zfill(2) + '/image/TrainSeq' + str(i).zfill(2) + '_RGB_Image_' + str(j).zfill(4) + '.png'
			labeled_image_file = '../data/HighwayDriving/Train/TrainSeq' + str(i).zfill(2) + '/label/TrainSeq' + str(i).zfill(2) + '_ColorLabel_' + str(j).zfill(4) + '.png'




			image = plt.imread(image_file)
			image_gray = rgb2gray(image)
			labeled_image = plt.imread(labeled_image_file)

			true_mask = binary_mask(labeled_image, include_lane=True)

			plt.imshow(image_gray, cmap='gray')
			plt.show()

			print(image.shape)

			image /= 255.0



			'''
				KMeans Clustering Based Segmentation
			'''

			feature_image = image.reshape(image.shape[0]*image.shape[1], image.shape[2])

			print(feature_image.shape)

			#kmeans = cluster.KMeans(n_clusters=4).fit(feature_image)


			#kmeans_segmented_image = kmeans.cluster_centers_[kmeans.labels_]
							
			#kmeans_segmented_image = kmeans_segmented_image.reshape(image.shape)
			#kmeans_segmented_image *= 255.0

			#plt.imshow(kmeans_segmented_image)
			#plt.show()

			# experimenting with different clustering algorithms

			'''
				SLIC Based Segmentation
			'''

			numSegments = 300

			segments = segmentation.slic(image*255.0, n_segments=numSegments, sigma=5)
			fig = plt.figure("Color Segmentation and Averaging")
			ax = fig.add_subplot(1, 1, 1)

			averaged = color.label2rgb(segments, image*255.0, kind='avg')

			ax.imshow(segmentation.mark_boundaries(averaged, segments))
			plt.show()

			sub_segments = segmentation.slic(averaged, n_segments=3, sigma=10)
			fig = plt.figure("Superpixel of Averaged")
			ax = fig.add_subplot(1, 1, 1)

			sub_averaged = color.label2rgb(sub_segments, image*255.0, kind='overlay')

			ax.imshow(sub_averaged)
			plt.show()



			

			
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
			
			ax.imshow(segmentation.mark_boundaries(image*255.0, segments))
			plt.show()

			

			# loop over the unique segment values

			unique_segs = np.unique(segments)
			segment_labels = np.zeros(len(unique_segs))

			cluster_mask = np.zeros(image_gray.shape[:2], dtype = "uint8")

			for (i, segVal) in enumerate(unique_segs):

				# extract features from a segment




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
				



				







