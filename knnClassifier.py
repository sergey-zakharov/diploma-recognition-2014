import cv2
import numpy as np

import feature_extractor as fe

class Classifier:
	knn = []
	nnet = []
	
	def initAndTrainNeuralNetwork(self, inputs_f, targets_f, nhidden):
		# The number of elements in an input vector, i.e. the number of nodes
		# in the input layer of the network.
		ninputs = len(inputs_f[0])

		# 8 hidden nodes.  If you change this to, for instance, 2, the network
		# won't work well.
		#nhidden = 8

		# We should have one output for each input vector (i.e., the digits
		# 0-9).
		noutput = 1

		# Create arrays for input and output. OpenCV neural networks expect
		# each row to correspond to a single input or target output vector.
		inputs = np.empty( (len(inputs_f), len(inputs_f[0])), 'float' )
		#targets_f = map(np.array, targets_f)
		#targets = np.array(targets_f)
		targets = -1 * np.ones( (len(inputs_f), len(inputs_f)), 'float' )
		# Convert input strings to binary zeros and ones, and set the output
		# array to all -1's with ones along the diagonal.
		#print len(inputs)
		#print "inputs_f", inputs_f
		#print "targets_f", targets_f
		#print "targets", targets
		
		# for i in range(len(inputs)):
		# 	inputs[i,:] = inputs_f[i]
		# 	targets[i,i] = targets_f[i][0]
		inputs = np.array(inputs_f)
		targets = np.array([ np.array([np.float32(row)]) for row in targets_f])

		# Create an array of desired layer sizes for the neural network
		layers = np.array([ninputs, nhidden, noutput])

		# Create the neural network
		self.nnet = cv2.ANN_MLP(layers)

		# Some parameters for learning.  Step size is the gradient step size
		# for backpropogation.
		step_size = 0.01

		# Momentum can be ignored for this example.
		momentum = 0.0

		# Max steps of training
		nsteps = 10000

		# Error threshold for halting training
		max_err = 0.0001

		# When to stop: whichever comes first, count or error
		condition = cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS

		# Tuple of termination criteria: first condition, then # steps, then
		# error tolerance second and third things are ignored if not implied
		# by condition
		criteria = (condition, nsteps, max_err)

		# params is a dictionary with relevant things for NNet training.
		params = dict( term_crit = criteria, 
			train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP, 
			bp_dw_scale = step_size, 
			bp_moment_scale = momentum )

		#print "inputs", inputs
		#print "targets", targets
		# Train our network
		num_iter = self.nnet.train(inputs, targets, None, params=params)

		## check neural network on train data
		
		# Create a matrix of predictions
		predictions = np.empty_like(targets)

		# See how the network did.
		self.nnet.predict(inputs, predictions)

		# Compute sum of squared errors
		sse = np.sum( (targets - predictions)**2 )

		# Compute # correct
		true_labels = targets
		pred_labels = [ np.array([1.]) if prediction > 0.5 else np.array([0.]) for prediction in predictions] # if prediction > 0.5, it is 1, else it is 0
		
		num_correct = np.sum( true_labels == pred_labels )

		print 'ran for %d iterations' % num_iter
		print 'sum sq. err:', sse
		print 'accuracy:', float(num_correct) / len(true_labels)

	def predictNeural(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		inputs_f = []
		inputs_f.append(fe.get_features(gray))
		#print "raw data about image", inputs_f

		# Create a matrix of predictions
		
		inputs_f.append(np.array([0., 10., 22., 26., 43., 37.,
    30., 27., 36., 48., 59., 70.,
    80., 82., 80., 75., 66., 57.,
    49., 123., 213., 123., 29., 25.,
    25., 24., 25., 30., 38., 57.,
    70., 92., 115., 122., 174., 199.,
   224., 255., 235., 116., 43., 26.,
    20., 18., 15., 1., 14., 12.,
    12., 10., 11., 13., 17., 18.,
    11., 10., 124., 18., 13., 13.,
    37., 17.,  5.,  1.,  8.80175781, 2.59635413, 1.29720053]))
		inputs = np.array(inputs_f)
		predictions =  np.empty( (len(inputs), 1), 'float' )

		self.nnet.predict(inputs, predictions)
		# Compute # correct
		pred_labels = [1 if prediction > 0.5 else 0 for prediction in predictions]
		
		#print pred_labels
		return pred_labels[0]

	def getSamplesToPredict(self):
		pass

	def train(self, samples, responses):
		self.knn = cv2.KNearest()
		self.knn.train(np.array(samples), responses, isRegression=True)

	def test(self, sample_img, numNeigh=11):
		k = numNeigh
		gray = cv2.cvtColor(sample_img,cv2.COLOR_BGR2GRAY)
		sample = fe.get_features(gray)
		sample = np.array(sample,np.float32).reshape((1,len(sample)))
		nearest = self.knn.find_nearest(sample, k)
		return nearest[0]

class Regression:
	knn = []
	nnet = []

	def getSamplesToPredict(self):
		pass

	def train(self, samples, responses):
		self.knn = cv2.KNearest()
		self.knn.train(np.array(samples), np.array(responses), isRegression=True)

	def test(self, sample_img):
		k = 10
		gray = cv2.cvtColor(sample_img,cv2.COLOR_BGR2GRAY)
		sample = fe.get_features(gray)
		sample = np.array(sample,np.float32).reshape((1,len(sample)))
		return self.knn.find_nearest(sample, k)[0]
	def initAndTrainNeuralNetwork(self, inputs_f, targets_f, nhidden):
		# The number of elements in an input vector, i.e. the number of nodes
		# in the input layer of the network.
		ninputs = len(inputs_f[0])

		# 8 hidden nodes.  If you change this to, for instance, 2, the network
		# won't work well.
		#nhidden = 8

		# We should have one output for each input vector (i.e., the digits
		# 0-9).
		noutput = 1
		
		# Create arrays for input and output. OpenCV neural networks expect
		# each row to correspond to a single input or target output vector.
		inputs = np.empty( (len(inputs_f), len(inputs_f[0])), 'float' )
		#targets_f = map(np.array, targets_f)
		#targets = np.array(targets_f)
		targets = -1 * np.ones( (len(inputs_f), len(inputs_f)), 'float' )
		# Convert input strings to binary zeros and ones, and set the output
		# array to all -1's with ones along the diagonal.
		#print len(inputs)
		#print "inputs_f", inputs_f
		#print "targets_f", targets_f
		#print "targets", targets
		
		# for i in range(len(inputs)):
		# 	inputs[i,:] = inputs_f[i]
		# 	targets[i,i] = targets_f[i][0]
		
		inputs = np.array(inputs_f)
		targets = np.array([ np.array([np.float32(row)]) for row in targets_f])

		# Create an array of desired layer sizes for the neural network
		layers = np.array([ninputs, nhidden, noutput])

		# Create the neural network
		self.nnet = cv2.ANN_MLP(layers)

		# Some parameters for learning.  Step size is the gradient step size
		# for backpropogation.
		step_size = 0.01

		# Momentum can be ignored for this example.
		momentum = 0.0

		# Max steps of training
		nsteps = 10000

		# Error threshold for halting training
		max_err = 0.0001

		# When to stop: whichever comes first, count or error
		condition = cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS

		# Tuple of termination criteria: first condition, then # steps, then
		# error tolerance second and third things are ignored if not implied
		# by condition
		criteria = (condition, nsteps, max_err)

		# params is a dictionary with relevant things for NNet training.
		params = dict( term_crit = criteria, 
						train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP, 
						bp_dw_scale = step_size, 
						bp_moment_scale = momentum )

		#print "inputs", inputs
		#print "targets", targets
		# Train our network
		num_iter = self.nnet.train(inputs, targets, None, params=params)

		## check neural network on train data
		
		# Create a matrix of predictions
		predictions = np.empty_like(targets)
		
		# See how the network did.
		#print str(self.nnet)
		self.nnet.predict(inputs, predictions)
		
		# Compute sum of squared errors
		sse = np.sum( (targets - predictions)**2 )

		# Compute # correct
		true_labels = targets
		pred_labels = predictions
		
		#print "true_labels", true_labels
		#print "pred_labels", pred_labels

		num_correct = 0
		for i in range(0, len(true_labels)):
			if pred_labels[i] > true_labels[i] - 5 and pred_labels[i] < true_labels[i] + 5:
				num_correct+=1

		print 'ran for %d iterations' % num_iter
		print 'sum sq. err:', sse
		print 'accuracy:', float(num_correct) / len(true_labels)

	def predictNeural(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		inputs_f = []
		inputs_f.append(fe.get_features(gray))
		#print "raw data about image", inputs_f

		# Create a matrix of predictions
		
		inputs_f.append(np.array([0., 10., 22., 26., 43., 37.,
    30., 27., 36., 48., 59., 70.,
    80., 82., 80., 75., 66., 57.,
    49., 123., 213., 123., 29., 25.,
    25., 24., 25., 30., 38., 57.,
    70., 92., 115., 122., 174., 199.,
   224., 255., 235., 116., 43., 26.,
    20., 18., 15., 1., 14., 12.,
    12., 10., 11., 13., 17., 18.,
    11., 10., 124., 18., 13., 13.,
    37., 17.,  5.,  1.,  8.80175781, 2.59635413, 1.29720053]))
		inputs = np.array(inputs_f)
		predictions =  np.empty( (len(inputs), 1), 'float' )
		# See how the network did.
		#print "inputs", inputs
		#print str(self.nnet)
		self.nnet.predict(inputs, predictions)
		#print "predictions", predictions
		# Compute # correct
		pred_labels = predictions
		
		#print pred_labels
		return pred_labels[0]


def getSamplesAndResponsesFromFiles():
	image_manager = fe.ImageManager("./learn_data/image_map")
	image_manager.loadImageDict()
	im_dict = image_manager.getImageDict()
	samples = []
	responses = []
	count = 0
	for filename, num_name in im_dict.items():
		count += 1
		print "\r" + str(count) + " of " + str(len(im_dict))
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
		feature_values.append(grey_same_neighbours_perc)
		num = 4
		grey_same_neighbours, total_points = fe.countPointsWithNeighboursOfSameColour(gray, num)
		grey_same_neighbours_perc = float(grey_same_neighbours)/float(total_points)*100
		grey_same_neighbours_perc = np.float32(grey_same_neighbours_perc)
		feature_values.append(grey_same_neighbours_perc)
		num = 7
		grey_same_neighbours, total_points = fe.countPointsWithNeighboursOfSameColour(gray, num)
		grey_same_neighbours_perc = float(grey_same_neighbours)/float(total_points)*100
		grey_same_neighbours_perc = np.float32(grey_same_neighbours_perc)
		feature_values.append(grey_same_neighbours_perc)

		samples.append(np.array(feature_values))
		
		with open("./learn_data/" + num_name+'-method.txt', "r") as myfile:
			splitted = myfile.readline().replace('\n', '').split(" ")
			print "./learn_data/" + num_name+'.txt', splitted
			if len(splitted) == 4:
				array = np.array([np.float32(splitted[0]), np.float32(splitted[1]), np.float32(splitted[2]), np.float32(splitted[3])]) 
			else:
				array = np.array([np.float32(splitted[0]), np.float32(splitted[1]), 0, 0]) 
			responses.append(array)
	return samples, responses

if __name__ == '__main__':
	classifier = Classifier()
	classifier.initAndTrainNeuralNetwork()
	img = cv2.imread("./test_data/originals/104.jpg")
	print img
	classifier.predictNeural(img)