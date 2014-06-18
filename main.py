import knnClassifier as cl
import quality_assessment as qa
import cv2
import sys

from subprocess import call

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print "Pass 1 argument - path to image, please"
		print 'there are ' + str(len(sys.argv)) + ' params'
		print 'Params: '+ str(sys.argv)
		exit(0)
	samples, responses = cl.getSamplesAndResponsesFromFiles()
	# train classifier
	cv2.namedWindow('resultImage', cv2.WINDOW_NORMAL)
	classifier = cl.Classifier()
	classifier.train(samples, responses)
	#classifier.initAndTrainNeuralNetwork()

	# testing
	file_name = sys.argv[1]
	file_raw_name = file_name.split('.')[-2].split('/')[-1]
	test_filename = "./test_data/originals/" + file_raw_name + ".jpg"
	original_text_file  = "./test_data/originals/" + file_raw_name + ".txt"

	print "Testing " + test_filename
	im = cv2.imread(test_filename)
	
	meth = str(classifier.test(im))
	if meth == "0": # cv2.THRESH_BINARY global binarization
		# train regressor
		print "Global binarization selected: going to find threshold"
		regressor = cl.Regression()
		loc_samples = []
		loc_responses = []
		for i, response in enumerate(responses):
			if response[0] == "0"
				loc_samples.append(samples[i])
				loc_responses.append(response[1])
		regressor.train(loc_samples, loc_responses)
		# get threshold
		
		thres = int(regressor.test(im))
		print thres
		# prepare image
		gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		ret, resimg = cv2.threshold(gray, thres, 255, cv2.THRESH_BINARY)
	elif meth == "1": # "cv2.ADAPTIVE_THRESH_MEAN_C"
		print "Adaptive threshold by mean selected: going to find threshold"
		
		loc_samples = []
		loc_responses_1 = []
		loc_responses_2 = []

		for i, response in enumerate(responses):
			if response[0] == "1"
				loc_samples.append(samples[i])
				loc_responses_1.append(np.array(response[2]))
				loc_responses_2.append(np.array(response[3]))
		regressor.train(loc_samples, loc_responses_1)
		first_answer = regressor.test(im)
		loc_samples.append(first_answer)
		regressor.train(loc_samples, loc_responses_2)
		second_answer = regressor.test(im)

		gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		resimg = cv2.adaptiveThreshold(gey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, first_answer, second_answer)

	cv2.imshow('resultImage',resimg)

	# prepare files
	save_path = "./test_data/" + test_filename.split('/')[-1].split('.')[0] + "-proc.jpg"
	recognize_path = "./test_data/" + test_filename.split('/')[-1].split('.')[0] + "-rec"
	cv2.imwrite(save_path,resimg)
	# recognize
	call_tesseract_list = ["tesseract", save_path, recognize_path]
	call(call_tesseract_list)
	#call_cat = ["cat", "./test_data/" + recognize_path+".txt"]
	#call(call_cat)

	#original_text = ""
	#original_text_file = open(original_text_file, 'r+')
	#for line in original_text_file:
	#	original_text+=line
	#print "Original text:", original_text
	
	recognized_text = ""
	recognized_text_file = open(recognize_path+".txt", 'r+')
	for line in recognized_text_file:
		recognized_text+=line
	print "Recognized text:", recognized_text
	# assess

	#print "Assessment:", qa.getRatio(original_text, recognized_text)

	while(1):
		k = cv2.waitKey(1) #& 0xFF
		if k == 27:
			break
	cv2.destroyAllWindows()

	


