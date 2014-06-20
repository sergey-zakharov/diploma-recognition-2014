import Levenshtein
import feature_extractor as fe

import re
import datetime

# TODO throuth garbage in 'recognized'
def getRatio(original, recognized):
	return Levenshtein.ratio(original, recognized)

def filter_string(str1, str2="0123456789abcdefghijklmnopqrstuvwxyz"):
	result = ""
	for c in str1:
		if c in str2 or c == " ":
			result += c
	return result

def baseCheckRecognitionQuality(result_file_suffix, result_file_prefix='./learn_data/', ground_file_prefix='./learn_data/originals/gt/gt_', im_manager_filename="./learn_data/image_map"):
	im_manager = fe.ImageManager(im_manager_filename)
	im_manager.loadImageDict()
	image_map = im_manager.getImageDict()
	
	im_number = 0
	sum_ratio = 0

	# for each file in image map try to get files from gt (prepropcessing) and init-result of recognition, and pass them into getRatio
	for filename, number in image_map.iteritems():
		ground = ""
		init_recog = ""
		print "\n[" + filename + "(" + number + ".jpg)]"
		try:
			filename = filename.split(".")[0] + ".txt"
			filename = ground_file_prefix + filename
			f = open(filename,"r")
			lines = f.readlines()
			for line in lines:
				line = line.split("\"")[1].lower() + "\n"
				ground += line
			ground = ground.lower()
			ground = ground.replace("\n", " ")
			ground = filter_string(ground)
		except IOError:
			print "No file:", filename, "\nPassing example\n"
			continue
		try:
			filename = result_file_prefix + number + result_file_suffix
			f = open(filename ,"r")
			lines = f.readlines()
			for line in lines:
				
				if line[0] == '-':
					break
				else:
					line = line.strip()
					line = line.lower()
					line = line.replace("\n", " ")
					line = filter_string(line)
					init_recog += line
		except IOError:
			print "No file:", filename, "\nPassing example"
			continue
		print "ground:", ground
		print "init_recog:", init_recog
		ratio = getRatio(ground, init_recog)
		print "Ratio:", ratio
		sum_ratio += ratio
		im_number += 1
		#todo
	return (sum_ratio/im_number)*100
def checkInitRecognitionQuality():
	return baseCheckRecognitionQuality("-recog-result-init.txt")

def checkManualSelectionRecognitionQuality():
	# for each file in image map try to get files from gt (prepropcessing) and result of recognition, and pass them into getRatio
	return baseCheckRecognitionQuality("-recog-result.txt")

def checkMachineSelectionRecognitionQualityOnTest():
	# for each file in test_data/image_map try to get files from gt (prepropcessing) and result of machine-setted recognition, and pass them into getRatio
	return baseCheckRecognitionQuality("-rec.txt", result_file_prefix='./test_data/', ground_file_prefix='./learn_data/originals/gt/gt_', im_manager_filename="./test_data/image_map")

def checkManualSelectionRecognitionQualityOnTest():
	# for each file in test_data/image_map try to get files from gt (prepropcessing) and result of  recognition, and pass them into getRatio
	return baseCheckRecognitionQuality("-recog-result-init.txt", result_file_prefix='./test_data/', ground_file_prefix='./learn_data/originals/gt/gt_', im_manager_filename="./test_data/image_map")

def run(knn_num_neighs):
	#print getRatio(unicode("Airtours holidays"), unicode("Airtours olidays"))
	results = "For KNN with knn_num_neighs =" + str(knn_num_neighs) + ":\n"
	results += "Overall init recognition quality on 'Learn' dataset: "+ str(checkInitRecognitionQuality()) + "%\n" +\
	"Overall manual recognition quality on 'Learn' dataset: "+ str(checkManualSelectionRecognitionQuality()) + "%\n"+\
	"Overall manual recognition quality on 'Test' dataset: "+ str(checkManualSelectionRecognitionQualityOnTest()) + "%\n"+\
	"Overall machine recognition quality on 'Test' dataset: "+ str(checkMachineSelectionRecognitionQualityOnTest()) + "%"
	
	now = datetime.datetime.now()

	with open('./results.txt', "a") as myfile:
		myfile.write("\n" + str(now) + ":\n" + results + '\n')
	print "\n" + str(now)
	print results

if __name__ == '__main__':
	run()
