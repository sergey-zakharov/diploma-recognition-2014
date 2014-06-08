import knnClassifier as cl
import quality_assessment as qa
import cv2
import sys

from subprocess import call

if __name__ == '__main__':
	# training
	cv2.namedWindow('resultImage', cv2.WINDOW_NORMAL)
	regressor = cl.Regression()
	regressor.train()

	classifier = cl.Classifier()
	classifier.train()

	# testing
	if len(sys.argv) != 2:
		print "Pass 1 argument - path to image, please"
		print 'there are ' + str(len(sys.argv)) + ' params'
		print 'Params: '+ str(sys.argv)
		exit(0)
	
	file_name = sys.argv[1]
	file_raw_name = file_name.split('.')[-2].split('/')[-1]
	test_filename = "./test_data/originals/" + file_raw_name + ".jpg"
	original_text_file  = "./test_data/originals/" + file_raw_name + ".txt"

	print "Testing " + test_filename
	im = cv2.imread(test_filename)
	#cv2.imshow('resultImage',im)
	#while(1):
	#	k = cv2.waitKey(1) #& 0xFF
	#	if k == 27:
	#		break
	thres = int(classifier.test(im))
	print thres
	# use this thres
	# if method == "cv2.Thresh_binary:"
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	ret, resimg = cv2.threshold(gray, thres, 255, cv2.THRESH_BINARY)
	
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

	

