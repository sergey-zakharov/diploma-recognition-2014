import cv2
import numpy as np

import feature_extractor as fe

class Classifier:
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
			
			feature_values = []
			# histogram features
			hist_values_list = fe.get_greyscale_hist_features(gray)
			hist_values_list = map(np.float32,hist_values_list)
			feature_values += hist_values_list
			# same neighbours feature
			num = 3
			grey_same_neighbours, total_points = fe.countPointsWithNeighboursOfSameColour(gray, num)
			grey_same_neighbours_perc = float(grey_same_neighbours)/float(total_points)*100
			grey_same_neighbours_perc = np.float32(grey_same_neighbours_perc)
			feature_values += grey_same_neighbours_perc
			num = 4
			grey_same_neighbours, total_points = fe.countPointsWithNeighboursOfSameColour(gray, num)
			grey_same_neighbours_perc = float(grey_same_neighbours)/float(total_points)*100
			grey_same_neighbours_perc = np.float32(grey_same_neighbours_perc)
			feature_values += grey_same_neighbours_perc
			num = 7
			grey_same_neighbours, total_points = fe.countPointsWithNeighboursOfSameColour(gray, num)
			grey_same_neighbours_perc = float(grey_same_neighbours)/float(total_points)*100
			grey_same_neighbours_perc = np.float32(grey_same_neighbours_perc)
			feature_values += grey_same_neighbours_perc

			samples.append(np.array(feature_values))

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
		print "Responses:", responses
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
			
			feature_values = []
			# histogram features
			hist_values_list = fe.get_greyscale_hist_features(gray)
			hist_values_list = map(np.float32,hist_values_list)
			feature_values += hist_values_list
			# same neighbours feature
			num = 3
			grey_same_neighbours, total_points = fe.countPointsWithNeighboursOfSameColour(gray, num)
			grey_same_neighbours_perc = float(grey_same_neighbours)/float(total_points)*100
			grey_same_neighbours_perc = np.float32(grey_same_neighbours_perc)
			feature_values += grey_same_neighbours_perc
			num = 4
			grey_same_neighbours, total_points = fe.countPointsWithNeighboursOfSameColour(gray, num)
			grey_same_neighbours_perc = float(grey_same_neighbours)/float(total_points)*100
			grey_same_neighbours_perc = np.float32(grey_same_neighbours_perc)
			feature_values += grey_same_neighbours_perc
			num = 7
			grey_same_neighbours, total_points = fe.countPointsWithNeighboursOfSameColour(gray, num)
			grey_same_neighbours_perc = float(grey_same_neighbours)/float(total_points)*100
			grey_same_neighbours_perc = np.float32(grey_same_neighbours_perc)
			feature_values += grey_same_neighbours_perc

			samples.append(np.array(feature_values))
			
			with open("./learn_data/" + num_name+'-method.txt', "r") as myfile:
				splitted = myfile.readline().replace('\n', '').split(" ")
				print "./learn_data/" + num_name+'.txt', splitted
				if splitted[0] == "0":
					responses.append(np.float32(splitted[1]))
		return samples, responses

	def getSamplesToPredict(self):
		pass

	def train(self):
		print "Train regression"
		samples, responses = self.getSamplesAndResponsesFromFiles()
		#print "Samples:", samples
		print "Responses:", responses
		self.knn = cv2.KNearest()
		self.knn.train(np.array(samples), np.array(responses), isRegression=True)

	def test(self, sample_img):
		k = 10
		gray = cv2.cvtColor(sample_img,cv2.COLOR_BGR2GRAY)
		sample = fe.get_greyscale_hist_features(gray)
		sample = np.array(sample,np.float32).reshape((1,len(sample)))
		return self.knn.find_nearest(sample, k)[0]