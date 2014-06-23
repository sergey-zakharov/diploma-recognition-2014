import cv2
import sys
import os
import numpy as np
import timeit

from subprocess import call

import knnClassifier as cl
import quality_assessment as qa
import settings

DEBUG = False
USE_KNN_IN_CLASSIFICATION = False # else using neural network
USE_KNN_IN_REGRESSION = False # else using neural network

global_bin_regressor = None
adabtive_bin_regressor_1 = None
adabtive_bin_regressor_2 = None
classifier = []
samples = []
responses = []

def train(nhidden = -1):
	global classifier
	global samples
	global responses

	# train classifier
	print "Creating classifier"
	classifier = cl.Classifier()
	print "Getting samples and responses"
	samples, responses = cl.getSamplesAndResponsesFromFiles()
	print "Training classifier"
	
	if USE_KNN_IN_CLASSIFICATION:
		classifier.train(samples, np.array([np.float32(row[0]) for row in responses]))
	else:
		if nhidden != -1:
			classifier.initAndTrainNeuralNetwork(samples, [row[0] for row in responses], nhidden)
		else:
			print "Specify nhidden parameter in train()"
			return

def run(knn_num_neigh=-1):
	global classifier
	global global_bin_regressor
	global adabtive_bin_regressor_1
	global adabtive_bin_regressor_2
	
	global samples
	global responses

	resimg = None

	for filename in os.listdir('./test_data/originals/'):
		if filename.split(".")[1] == 'jpg':
			# testing
			print "\nTesting:", filename
			file_name = filename
			file_raw_name = file_name.split('.')[-2].split('/')[-1]
			test_filename = "./test_data/originals/" + file_raw_name + ".jpg"
			original_text_file  = "./test_data/originals/" + file_raw_name + ".txt"

			print "Testing " + test_filename
			im = cv2.imread(test_filename)
			
			if USE_KNN_IN_CLASSIFICATION:
				meth = str(int(classifier.test(im, knn_num_neigh)))
			else:
				meth = str(int(classifier.predictNeural(im)))
			print "Method: ", meth
			#raw_input("Press Enter to continue...")
			if meth == "0": # cv2.THRESH_BINARY global binarization
				# train regressor
				print "Global binarization selected: going to find threshold"

				loc_samples = []
				loc_responses = []
				for i, response in enumerate(responses):
					if str(int(response[0])) == "0":
						loc_samples.append(samples[i])
						loc_responses.append(response[1])
				if global_bin_regressor == None:
					print "Training global threshold regressor"
					loc_responses = map(np.float32, loc_responses)
					loc_responses = np.array(loc_responses)
					
					global_bin_regressor = cl.Regression()
					if USE_KNN_IN_REGRESSION:

						global_bin_regressor.train(loc_samples, loc_responses)
					else:
						global_bin_regressor.initAndTrainNeuralNetwork(loc_samples, loc_responses, nhidden)
					

				# get threshold
				if USE_KNN_IN_REGRESSION:
					thres = int(global_bin_regressor.test(im))
				else:
					thres = global_bin_regressor.predictNeural(im)
				
				print "global threshold: ", thres
				# prepare image
				gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
				ret, resimg = cv2.threshold(gray, thres, 255, cv2.THRESH_BINARY)
				
				with open("./test_data/" + test_filename.split('/')[-1].split('.')[0] +'-method.txt', "w+") as myfile:
					myfile.write("cv2.THRESH_BINARY\n" + "threshold:" + str(thres))
			elif meth == "1": # "cv2.ADAPTIVE_THRESH_MEAN_C"
				print "Adaptive threshold by mean selected: going to find threshold"
				
				loc_samples = []
				loc_responses_1 = []
				loc_responses_2 = []
				
				for i, response in enumerate(responses):
					if str(int(response[0])) == "1":
						loc_samples.append(samples[i])
						loc_responses_1.append(np.array(response[2]))
						loc_responses_2.append(np.array(response[3]))
				if adabtive_bin_regressor_1 == None:
					adabtive_bin_regressor_1 = cl.Regression()
					print "Training first adaptive threshold regressor"
					if USE_KNN_IN_REGRESSION:
						adabtive_bin_regressor_1.train(loc_samples, loc_responses_1)
					else:
						adabtive_bin_regressor_1.initAndTrainNeuralNetwork(loc_samples, loc_responses_1, nhidden)

				if adabtive_bin_regressor_2 == None:
					adabtive_bin_regressor_2 = cl.Regression()
					print "Training second adaptive threshold regressor"
					if USE_KNN_IN_REGRESSION:
						adabtive_bin_regressor_2.train(loc_samples, loc_responses_2)
					else:
						adabtive_bin_regressor_2.initAndTrainNeuralNetwork(loc_samples, loc_responses_2, nhidden)

				if USE_KNN_IN_REGRESSION:
					first_answer = adabtive_bin_regressor_1.test(im)
					second_answer = adabtive_bin_regressor_2.test(im)
				else:
					first_answer = adabtive_bin_regressor_1.predictNeural(im)
					second_answer = adabtive_bin_regressor_2.predictNeural(im)

				first_answer = int(first_answer)
				if first_answer%2 == 0:
					first_answer+=1
				print "block size:", first_answer
				print "C:", second_answer

				gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
				resimg = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, abs(int(first_answer)), int(second_answer))

				with open("./test_data/" + test_filename.split('/')[-1].split('.')[0] +'-method.txt', "w+") as myfile:
					myfile.write("cv2.ADAPTIVE_THRESH_MEAN_C\n" + "block size:" + str(abs(first_answer)) + "\nC:" + str(second_answer))
			# prepare files
			save_path = "./test_data/" + test_filename.split('/')[-1].split('.')[0] + ".jpg"
			recognize_path = "./test_data/" + test_filename.split('/')[-1].split('.')[0] + "-rec"

			cv2.imwrite(save_path,resimg)
			
			# recognize
			call_tesseract_list = ["tesseract", save_path, recognize_path]
			call(call_tesseract_list)
			
			recognized_text = ""
			recognized_text_file = open(recognize_path+".txt", 'r+')
			for line in recognized_text_file:
				recognized_text+=line

			print "Recognized text:", recognized_text
		if DEBUG == True:
			break

	


if __name__ == '__main__':
	if USE_KNN_IN_CLASSIFICATION:
		knn_num_neighs = [5, 7, 9, 11, 13]
	else:
		nhiddens = [8]

	if USE_KNN_IN_CLASSIFICATION:
		start_train = timeit.default_timer()
		train()
		stop_train = timeit.default_timer()
		print "Train time:", str(stop_train - start_train), "seconds"
		for knn_num_neigh in knn_num_neighs:
			print "\n\nKNN number of neighbours =", knn_num_neigh
			start = timeit.default_timer()

			run(knn_num_neigh)

			stop = timeit.default_timer()
			print "Recognition time:", str(stop - start), "seconds"
			
			if DEBUG != True:
				qa.run(knn_num_neigh=knn_num_neigh)
	else:
		for nhidden in nhiddens:
			print "\n\nNeuralNet number of hidden nodes =", nhidden
			start_train = timeit.default_timer()
			train(nhidden)
			stop_train = timeit.default_timer()
			print "Train time:", str(stop_train - start_train), "seconds"
			start = timeit.default_timer()

			run()

			stop = timeit.default_timer()
			print "Recognition time:", str(stop - start), "seconds"
			
			if DEBUG != True:
				qa.run(nhidden=nhidden)
	print "Train time:", str(stop_train - start_train), "seconds"