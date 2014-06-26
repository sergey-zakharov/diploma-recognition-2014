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

USE_KNN_IN_REGRESSION = False
USE_SVM_IN_REGRESSION = True
USE_ANN_IN_REGRESSION = False

USE_KNN_IN_CLASSIFICATION = False
USE_SVM_IN_CLASSIFICATION = True
USE_ANN_IN_CLASSIFICATION = False

if 	(USE_KNN_IN_CLASSIFICATION and USE_SVM_IN_CLASSIFICATION) or \
	(USE_KNN_IN_CLASSIFICATION and USE_ANN_IN_CLASSIFICATION) or \
	(USE_ANN_IN_CLASSIFICATION and USE_SVM_IN_CLASSIFICATION):
	print "Use only ONE type of classifier!"
	exit(1)

if not USE_KNN_IN_CLASSIFICATION and not USE_SVM_IN_CLASSIFICATION:
	USE_ANN_IN_CLASSIFICATION = True

if 	(USE_KNN_IN_REGRESSION and USE_SVM_IN_REGRESSION) or \
	(USE_KNN_IN_REGRESSION and USE_ANN_IN_REGRESSION) or \
	(USE_ANN_IN_REGRESSION and USE_SVM_IN_REGRESSION):
	print "Use only ONE type of regressor!"
	exit(1)

if not USE_KNN_IN_REGRESSION and not USE_SVM_IN_REGRESSION:
	USE_ANN_IN_REGRESSION = True


global_bin_regressor = None
adabtive_bin_regressor_1 = None
adabtive_bin_regressor_2 = None
classifier = []
samples = []
responses = []

def train(nhidden = -1, type_i=cv2.SVM_LINEAR, C=2.67, gamma=5.383):
	global classifier
	global samples
	global responses

	# train classifier
	print "Creating classifier"
	classifier = cl.Classifier()
	
	print "Training classifier"
	if USE_KNN_IN_CLASSIFICATION:
		classifier.train(samples, np.array([np.float32(row[0]) for row in responses]))
	elif USE_ANN_IN_CLASSIFICATION:
		if nhidden != -1:
			classifier.initAndTrainNeuralNetwork(samples, [row[0] for row in responses], nhidden)
		else:
			print "Specify nhidden parameter in train()"
			return
	elif USE_SVM_IN_CLASSIFICATION:
		classifier.trainSVM(samples, [row[0] for row in responses], type_i, C, gamma)

def run(knn_num_neigh=-1, nhidden= -1, type_i=cv2.SVM_LINEAR, p=2., C=2.67, gamma=5.383):
	global classifier
	global global_bin_regressor
	global adabtive_bin_regressor_1
	global adabtive_bin_regressor_2
	
	global samples
	global responses

	resimg = None

	for filename in os.listdir('./test_data/testset2/originals/'):
		if filename.split(".")[1] == 'jpg':
			# testing
			print "\nTesting:", filename
			file_name = filename
			file_raw_name = file_name.split('.')[-2].split('/')[-1]
			test_filename = "./test_data/testset2/originals/" + file_raw_name + ".jpg"
			
			print "Testing " + test_filename
			im = cv2.imread(test_filename)
			
			if USE_KNN_IN_CLASSIFICATION:
				meth = str(int(classifier.test(im, knn_num_neigh)))
			elif USE_ANN_IN_CLASSIFICATION:
				meth = str(int(classifier.predictNeural(im)))
			elif USE_SVM_IN_CLASSIFICATION:
				meth = str(int(classifier.predictSVM(im)))

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
					elif USE_ANN_IN_REGRESSION:
						global_bin_regressor.initAndTrainNeuralNetwork(loc_samples, loc_responses, nhidden)
					elif USE_SVM_IN_REGRESSION:
						global_bin_regressor.trainSVM(loc_samples, loc_responses, type_i, p, C, gamma)

				# get threshold
				if USE_KNN_IN_REGRESSION:
					thres = int(global_bin_regressor.test(im))
				elif USE_ANN_IN_REGRESSION:
					thres = global_bin_regressor.predictNeural(im)
				elif USE_SVM_IN_REGRESSION:
					thres = global_bin_regressor.predictSVM(im)

				print "global threshold: ", thres
				# prepare image
				gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
				ret, resimg = cv2.threshold(gray, thres, 255, cv2.THRESH_BINARY)
				
				with open("./test_data/testset2/" + test_filename.split('/')[-1].split('.')[0] +'-method.txt', "w+") as myfile:
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
					elif USE_ANN_IN_REGRESSION:
						adabtive_bin_regressor_1.initAndTrainNeuralNetwork(loc_samples, loc_responses_1, nhidden)
					elif USE_SVM_IN_REGRESSION:
						adabtive_bin_regressor_1.trainSVM(loc_samples, loc_responses_1, type_i, p, C, gamma)

				if adabtive_bin_regressor_2 == None:
					adabtive_bin_regressor_2 = cl.Regression()
					print "Training second adaptive threshold regressor"
					if USE_KNN_IN_REGRESSION:
						adabtive_bin_regressor_2.train(loc_samples, loc_responses_2)
					elif USE_ANN_IN_REGRESSION:
						adabtive_bin_regressor_2.initAndTrainNeuralNetwork(loc_samples, loc_responses_2, nhidden)
					elif USE_SVM_IN_REGRESSION:
						adabtive_bin_regressor_2.trainSVM(loc_samples, loc_responses_2, type_i, p, C, gamma)

				if USE_KNN_IN_REGRESSION:
					first_answer = adabtive_bin_regressor_1.test(im)
					second_answer = adabtive_bin_regressor_2.test(im)
				elif USE_ANN_IN_REGRESSION:
					first_answer = adabtive_bin_regressor_1.predictNeural(im)
					second_answer = adabtive_bin_regressor_2.predictNeural(im)
				elif USE_SVM_IN_REGRESSION:
					first_answer = adabtive_bin_regressor_1.predictSVM(im)
					second_answer = adabtive_bin_regressor_2.predictSVM(im)

				first_answer = abs(int(first_answer))
				if first_answer%2 == 0:
					first_answer+=1

				if first_answer == 1:
					first_answer = 3
				print "block size:", first_answer
				print "C:", second_answer

				gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
				resimg = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, abs(int(first_answer)), int(second_answer))

				with open("./test_data/testset2/" + test_filename.split('/')[-1].split('.')[0] +'-method.txt', "w+") as myfile:
					myfile.write("cv2.ADAPTIVE_THRESH_MEAN_C\n" + "block size:" + str(abs(first_answer)) + "\nC:" + str(second_answer))
			# prepare files
			save_path = "./test_data/testset2/" + test_filename.split('/')[-1].split('.')[0] + ".jpg"
			recognize_path = "./test_data/testset2/" + test_filename.split('/')[-1].split('.')[0] + "-rec"

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
	
	print "Getting samples and responses"
	samples, responses = cl.getSamplesAndResponsesFromFiles()
	
	if USE_KNN_IN_CLASSIFICATION:
		start_train = timeit.default_timer()
		train()
		stop_train = timeit.default_timer()
		print "Train time:", str(stop_train - start_train), "seconds"
		for knn_num_neigh in knn_num_neighs:
			print "\n\nKNN number of neighbours =", knn_num_neigh
			start = timeit.default_timer()

			run(knn_num_neigh=knn_num_neigh)

			stop = timeit.default_timer()
			print "Recognition time:", str(stop - start), "seconds"
			
			if DEBUG != True:
				qa.run(knn_num_neigh=knn_num_neigh)
	elif USE_ANN_IN_CLASSIFICATION:
		for nhidden in nhiddens:
			print "\n\nNeuralNet number of hidden nodes =", nhidden
			start_train = timeit.default_timer()
			train(nhidden=nhidden)
			stop_train = timeit.default_timer()
			print "Train time:", str(stop_train - start_train), "seconds"
			start = timeit.default_timer()

			run()

			stop = timeit.default_timer()
			print "Recognition time:", str(stop - start), "seconds"
			qa.run(nhidden=nhidden)

	elif USE_SVM_IN_CLASSIFICATION:
		types = {
			"LINEAR" : cv2.SVM_LINEAR,
			"RBF" : cv2.SVM_RBF
			}
		#for type_name, type_i in types.iteritems(): # type for classificator
			#for C in [0.2, 0.25, 0.3, 0.4, 0.5]:		# C for classificator
				#for gamma in [2., 2.5, 3., 2.5, 4., 4.5, 5., 5.383, 5.5, .6, 6.5]: # gamma for RBF for classificator
		type_name = "LINEAR"
		type_i = cv2.SVM_LINEAR

		gamma = 2.
		C = 0.2
		
		start_train = timeit.default_timer()
		train(type_i = type_i, C=C, gamma=gamma) # train classificator
		stop_train = timeit.default_timer()
		print "Train time:", str(stop_train - start_train), "seconds"
		start = timeit.default_timer()

		#for type_name_r, type_i_r in types.iteritems(): # type for regressors
			#for C_r in [0.2, 0.25, 0.3, 0.4, 0.5]:	# C for regressors
				#for gamma_r in [2., 2.5, 3., 2.5, 4., 4.5, 5., 5.383, 5.5, .6, 6.5]: # gamma for RBF in regressors
		#p_range = np.arange(27.,32.,0.2)
		for p_r in [30.9, 30.95, 30.98, 31.03, 31.05, 31.08, 31.13,31.15, 31.25]: # p for regressors
			type_name_r = "LINEAR"
			type_i_r = cv2.SVM_LINEAR
			
			#p_r = 31.05
			gamma_r = 2.
			C_r = 0.2
			
			global_bin_regressor = None
			adabtive_bin_regressor_1 = None
			adabtive_bin_regressor_2 = None
			run(type_i=type_i_r, C=C_r, p=p_r, gamma=gamma_r)
			
			text = "SVM: \n For classification: type = " + type_name + ", gamma = " + str(gamma) + ", C=" + str(C)
			text+="\n For regression: p= " + str(p_r) + ", type = " + type_name_r + ", gamma = " + str(gamma_r) + ", C=" + str(C_r) + "\n"
			qa.run(text = text)

	print "Train time:", str(stop_train - start_train), "seconds"