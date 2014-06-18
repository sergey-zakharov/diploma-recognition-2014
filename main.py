import knnClassifier as cl
import quality_assessment as qa
import cv2
import sys
import os
import numpy as np
import timeit

from subprocess import call

if __name__ == '__main__':
	global_bin_regressor = None
	adabtive_bin_regressor_1 = None
	adabtive_bin_regressor_2 = None
	resimg = None

	start = timeit.default_timer()

	# train classifier
	print "Creating classifier"
	classifier = cl.Classifier()
	print "Getting samples and responses"
	samples, responses = cl.getSamplesAndResponsesFromFiles()
	print "Training classifier"
	#print responses
	classifier.train(samples, np.array([np.float32(row[0]) for row in responses]))
	#classifier.initAndTrainNeuralNetwork()

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
			
			meth = str(int(classifier.test(im)))
			print "Method: ", meth
			if meth == "0": # cv2.THRESH_BINARY global binarization
				# train regressor
				print "Global binarization selected: going to find threshold"
				global_bin_regressor = cl.Regression()

				print "Train global threshold regression"
				global_bin_regressor.train(samples, [np.float32(row[1]) for row in responses])
				loc_samples = []
				loc_responses = []
				for i, response in enumerate(responses):
					if response[0] == "0":
						loc_samples.append(samples[i])
						loc_responses.append(response[1])
				if global_bin_regressor == None:
					print "Training global threshold regressor"
					global_bin_regressor.train(loc_samples, loc_responses)
				# get threshold
				
				thres = int(global_bin_regressor.test(im))
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
				
				if adabtive_bin_regressor_1 == None:
					adabtive_bin_regressor_1 = cl.Regression()
				if adabtive_bin_regressor_2 == None:
					adabtive_bin_regressor_2 = cl.Regression()

				for i, response in enumerate(responses):
					if str(int(response[0])) == "1":
						loc_samples.append(samples[i])
						loc_responses_1.append(np.array(response[2]))
						loc_responses_2.append(np.array(response[3]))
				print "Training first adaptive threshold regressor"
				adabtive_bin_regressor_1.train(loc_samples, loc_responses_1)
				print "Training second adaptive threshold regressor"
				adabtive_bin_regressor_2.train(loc_samples, loc_responses_2)

				first_answer = adabtive_bin_regressor_1.test(im)
				second_answer = adabtive_bin_regressor_2.test(im)

				print "block size:", first_answer
				print "C:", second_answer

				gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
				resimg = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, int(first_answer), int(second_answer))

				with open("./test_data/" + test_filename.split('/')[-1].split('.')[0] +'-method.txt', "w+") as myfile:
					myfile.write("cv2.ADAPTIVE_THRESH_MEAN_C\n" + "block size:" + str(first_answer) + "\nC:" + str(second_answer))
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

	stop = timeit.default_timer()
	print "Overall time:", str(stop - start)


