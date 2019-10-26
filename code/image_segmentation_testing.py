import skimage.segmentation as segmentation
import skimage.color as color
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import numpy as np

def binary_mask(labeled_image, include_lane=False):

	mask = np.all(labeled_image == 128/255.0, axis=2)

	if include_lane:
		mask1 = labeled_image[:,:,0] == 1.0
		mask2 = labeled_image[:,:,1] == 1.0
		mask3 = labeled_image[:,:,2] == 0.0

		lane_mask = np.all(np.stack([mask1, mask2, mask3], axis=2), axis=2)
		mask = np.any(np.stack([mask, lane_mask], axis=2), axis=2)


	plt.imshow(mask)
	plt.show()

	return mask

if __name__ == '__main__':

	test_image_file = '../data/HighwayDriving/Train/TrainSeq00/image/TrainSeq00_RGB_Image_0000.png'
	labeled_test_image_file = '../data/HighwayDriving/Train/TrainSeq00/label/TrainSeq00_ColorLabel_0010.png'

	image = plt.imread(test_image_file)
	labeled_image = plt.imread(labeled_test_image_file)

	mask = binary_mask(labeled_image, include_lane=True)

	plt.imshow(image)
	plt.show()

	print(image.shape)

	image /= 255.0


	'''
		KMeans Clustering Based Segmentation
	'''

	feature_image = image.reshape(image.shape[0]*image.shape[1], image.shape[2])

	print(feature_image.shape)

	kmeans = cluster.KMeans(n_clusters=4).fit(feature_image)


	kmeans_segmented_image = kmeans.cluster_centers_[kmeans.labels_]
                    
	kmeans_segmented_image = kmeans_segmented_image.reshape(image.shape)
	kmeans_segmented_image *= 255.0

	plt.imshow(kmeans_segmented_image)
	plt.show()

	# experimenting with different clustering algorithms

	'''
		SLIC Based Segmentation
	'''

	image *= 255.0

		# loop over the number of segments
	for numSegments in (100, 200, 300):
		# apply SLIC and extract (approximately) the supplied number
		# of segments
		segments = segmentation.slic(image, n_segments = numSegments, sigma = 5)
	 
		# show the output of SLIC
		fig = plt.figure("Superpixels -- %d segments" % (numSegments))
		ax = fig.add_subplot(1, 1, 1)
		ax.imshow(segmentation.mark_boundaries(color.label2rgb(segments, image, kind='avg'), segments))
		plt.show()

		fig = plt.figure("Superpixels -- %d segments" % (numSegments))
		ax = fig.add_subplot(1, 1, 1)
		
		ax.imshow(segmentation.mark_boundaries(color.label2rgb(mask, image, kind='overlay'), segments))
		



		plt.show()







