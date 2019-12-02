import numpy as np
import cv2
import os
import skimage.color as color
import skimage.segmentation as segmentation
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import StratifiedKFold as KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support

def show(im):

	cv2.imshow('im', im)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def binary_mask(labeled_image, include_lane=False):

	mask = np.all(labeled_image == 128, axis=2)

	if include_lane:
		mask1 = labeled_image[:,:,0] == 0
		mask2 = labeled_image[:,:,1] == 255
		mask3 = labeled_image[:,:,2] == 255

		lane_mask = np.all(np.stack([mask1, mask2, mask3], axis=2), axis=2)
		mask = np.any(np.stack([mask, lane_mask], axis=2), axis=2)


	return mask



def load_data(path):

	'''
		Layout of image data object is as follows:

		{
			"Train": [
						{...},
						{...}
					  ],
			"Test":  [
						{
							"images": [...],
							"labels": [...],
							"segments": [...]
						},
						{...}]
		}
	'''

	path = os.path.join(path)

	image_data = {}

	for data in ['Train', 'Test']:
		image_data[data] = []

		if data == 'Train':
			num_sequences = 15
		else:
			num_sequences = 5


		for i in range(num_sequences):

			sequence_dict = {}

			images = []
			labels = []
			segments = []

			# there are 60 images per sequence
			for j in range(1):

				image_file = '../data/HighwayDriving/' + data + '/' + data + 'Seq' + str(i).zfill(2) + '/image/' + data + 'Seq' + str(i).zfill(2) + '_RGB_Image_' + str(j).zfill(4) + '.png'
				labeled_image_file = '../data/HighwayDriving/' + data + '/' + data + 'Seq' + str(i).zfill(2) + '/label/' + data + 'Seq' + str(i).zfill(2) + '_ColorLabel_' + str(j).zfill(4) + '.png'

				image = cv2.imread(image_file, cv2.IMREAD_COLOR)
				labeled_image = cv2.imread(labeled_image_file, cv2.IMREAD_COLOR)

				segment = segmentation.slic(image, n_segments = 300, sigma = 5)

				images.append(image)
				labels.append(labeled_image)
				segments.append(segment)

			sequence_dict['images'] = np.stack(images, axis=0)
			sequence_dict['labels'] = np.stack(labels, axis=0)
			sequence_dict['segments'] = np.stack(segments, axis=0)

			image_data[data].append(sequence_dict)


	return image_data

def format_data(data):

	X = []
	y = []

	for sequence in data:
		for i in range(len(sequence['images'])):

			image = sequence['images'][i]
			label = sequence['labels'][i]
			segments = sequence['segments'][i]

			vectors, labels = preprocess(image, label, segments)

			X.append(vectors)
			y.append(labels)


	X = np.vstack(X)
	y = np.vstack(y)

	return X, y


def preprocess(image, label, segments):

	vectors = []
	labels = []


	true_mask = binary_mask(label, include_lane=True)
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	unique_segs = np.unique(segments)
	segment_labels = np.zeros(len(unique_segs))

	cluster_mask = np.zeros(image_gray.shape[:2], dtype = "uint8")


	for (i, segVal) in enumerate(unique_segs):

		seg_mask = np.zeros(image_gray.shape[:2], dtype = "uint8")
		seg_mask[segments == segVal] = 1.0

		hist = np.histogram(image_gray[segments==segVal].flatten(), bins=np.arange(257), density=True)[0]

		label = 0
		if np.sum(np.bitwise_and(seg_mask, true_mask)) / np.sum(seg_mask) >= 0.5:
			segment_labels[i] = 1
			cluster_mask[segments == segVal] = 1

			label = 1
		vectors.append(hist)
		labels.append(label)

	vectors = np.vstack(vectors)
	labels = np.vstack(labels)

				

	# show the output of SLIC
	#show(segmentation.mark_boundaries(color.label2rgb(cluster_mask, image_gray, kind='overlay'), segments))

	return vectors, labels

def train_model(X, y):

	#clf = KNeighborsClassifier(n_neighbors=10, metric='manhattan', weights='distance').fit(X, y)
	#return clf
	
	neighbors = np.array([2**i for i in range(10)])

	neighbor_score = []
	classifiers = []

	for n in neighbors:

		kf = KFold(n_splits=5, shuffle=True)

		scores = []
		clfs = []
		for train_index, test_index in kf.split(X, y):
			X_train, X_valid = X[train_index], X[test_index]
			y_train, y_valid = y[train_index], y[test_index]

			clf = KNeighborsClassifier(n_neighbors=n, metric='manhattan', weights='distance').fit(X_train, y_train)

			y_pred = clf.predict(X_valid)

			precision, recall, f1, support = precision_recall_fscore_support(y_valid, y_pred, average='binary')
			print(precision, recall, f1, support)

			scores.append(f1)
			clfs.append(clf)

		scores = np.array(scores)

		neighbor_score.append(np.mean(scores))
		classifiers.append(clfs[np.argmax(scores)])

		print(n, np.mean(scores))

	neighbor_score = np.array(neighbor_score)

	plt.plot(neighbors, neighbor_score)
	plt.show()

	return classifiers[np.argmax(neighbor_score)].fit(X,y)

def evaluate(data, classifier):

	for sequence in data:
		for i in range(len(sequence['images'])):

			image = sequence['images'][i]
			label = sequence['labels'][i]

			segments = segmentation.slic(image, n_segments = 300, sigma = 5)

			#true_mask = binary_mask(label, include_lane=True)
			image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			unique_segs = np.unique(segments)
			segment_labels = np.zeros(len(unique_segs))

			cluster_mask = np.zeros(image_gray.shape[:2], dtype = "uint8")

			for (i, segVal) in enumerate(unique_segs):

				seg_mask = np.zeros(image_gray.shape[:2], dtype = "uint8")
				seg_mask[segments == segVal] = 1.0

				hist = np.histogram(image_gray[segments==segVal].flatten(), bins=np.arange(257), density=True)[0]

				cluster_mask[segments == segVal] = classifier.predict(hist.reshape(1, -1))

			# show the output of SLIC
			fig = plt.figure("Derived Labels for Superpixels from Ground Truth")
			ax = fig.add_subplot(1, 1, 1)
			ax.imshow(color.label2rgb(cluster_mask, image_gray, kind='overlay'))
			plt.show()

	return []



if __name__ == '__main__':

	
	image_data = load_data('../data/HighwayDriving')
	pickle.dump(image_data, open('image_data_small.pickle', 'wb'))

	image_data = pickle.load(open('image_data_small.pickle', 'rb'))

	X_train, y_train = format_data(image_data['Train'])

	#print(X_train.shape)
	#print(y_train.shape)

	np.savez('train_matrix_small.npz', X=X_train, y=y_train)
	

	train_data = np.load('train_matrix_small.npz')
	X_train, y_train = train_data['X'], train_data['y']
	y_train = y_train.ravel()

	clf = train_model(X_train, y_train)
	with open('knn_model.pkl', 'wb') as f:
		pickle.dump(clf, f)

	with open('knn_model.pkl', 'rb') as f:
		clf = pickle.load(f)

	evaluate(image_data['Test'], clf)

	

	





