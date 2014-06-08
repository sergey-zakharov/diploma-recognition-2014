import cv2
import numpy as np

import feature_extractor as fe

class Classifier:
	knn = []
	num = 4
	def getSamplesAndResponsesFromFiles(self):
		image_manager = fe.ImageManager("./learn_data/image_map")
		image_manager.loadImageDict()
		im_dict = image_manager.getImageDict()
		samples = []
		responses = []
		for filename, num_name in im_dict.items():
			im = cv2.imread("./learn_data/originals/" + filename)
			gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
			# histogram features
			hist_values_list = fe.get_greyscale_hist_features(gray)
			hist_values_list = map(np.float32,hist_values_list)
			samples.append(np.array(hist_values_list))
			# same neighbours feature
			grey_same_neighbours, total_points = fe.countPointsWithNeighboursOfSameColour(gray, num)
			grey_same_neighbours_perc = float(grey_same_neighbours)/float(total_points)*100
			grey_same_neighbours_perc = map(np.float32, grey_same_neighbours_perc)
			samples.append(np.array(grey_same_neighbours_perc))
			
			with open("./learn_data/" + num_name+'-method.txt', "r") as myfile:
				splitted = myfile.readline().replace('\n', '').split(" ")
				print "./learn_data/" + num_name+'.txt', splitted
				responses.append(np.float32(splitted[0]))
		return samples, responses

	def getSamplesToPredict(self):
		pass

	def train(self):
		print "Train classifier"
		samples, responses = self.getSamplesAndResponsesFromFiles()
		self.knn = cv2.KNearest()
		self.knn.train(np.array(samples), np.array(responses), isRegression=True)

	def test(self, sample_img):
		k = 10
		gray = cv2.cvtColor(sample_img,cv2.COLOR_BGR2GRAY)
		sample = fe.get_greyscale_hist_features(gray)
		sample = np.array(sample,np.float32).reshape((1,len(sample)))
		return self.knn.find_nearest(sample, k)[0]

class Regression:
	knn = []
	def getSamplesAndResponsesFromFiles(self):
		image_manager = fe.ImageManager("./learn_data/image_map")
		image_manager.loadImageDict()
		im_dict = image_manager.getImageDict()
		samples = []
		responses = []
		for filename, num_name in im_dict.items():
			im = cv2.imread("./learn_data/originals/" + filename)
			gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
			# histogram features
			hist_values_list = fe.get_greyscale_hist_features(gray)
			hist_values_list = map(np.float32,hist_values_list)
			samples.append(np.array(hist_values_list))
			# same neighbours feature
			grey_same_neighbours, total_points = fe.countPointsWithNeighboursOfSameColour(gray, num)
			grey_same_neighbours_perc = float(grey_same_neighbours)/float(total_points)*100
			grey_same_neighbours_perc = map(np.float32, grey_same_neighbours_perc)
			with open("./learn_data/" + num_name+'-method.txt', "r") as myfile:
				splitted = myfile.readline().replace('\n', '').split(" ")
				if splitted[1] == "0"
					print "./learn_data/" + num_name+'.txt', splitted
					responses.append(np.float32(splitted[1]))
		return samples, responses

	def getSamplesToPredict(self):
		pass

	def train(self):
		print "Train regression"
		samples, responses = self.getSamplesAndResponsesFromFiles()
		self.knn = cv2.KNearest()
		self.knn.train(np.array(samples), np.array(responses), isRegression=True)

	def test(self, sample_img):
		k = 10
		gray = cv2.cvtColor(sample_img,cv2.COLOR_BGR2GRAY)
		sample = fe.get_greyscale_hist_features(gray)
		sample = np.array(sample,np.float32).reshape((1,len(sample)))
		return self.knn.find_nearest(sample, k)[0]