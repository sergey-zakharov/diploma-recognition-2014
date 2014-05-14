import cv2
import numpy as np
import matplotlib.pyplot as plt


def testAffine():
	img = cv2.imread('drawing.jpg')
	rows,cols,ch = img.shape

	pts1 = np.float32([[33,34],[138,34],[33,140]])
	pts2 = np.float32([[10,100],[200,50],[100,250]])

	M = cv2.getAffineTransform(pts1,pts2)

	dst = cv2.warpAffine(img,M,(cols,rows))

	plt.subplot(121),plt.imshow(img),plt.title('Input')
	plt.subplot(122),plt.imshow(dst),plt.title('Output')
	plt.show()

def testPersTransformation():
	img = cv2.imread('sudokusmall.jpg')
	rows,cols,ch = img.shape

	pts1 = np.float32([[30,32],[15,188],[182,24],[194,191]])
	pts2 = np.float32([[0,0],[0,300],[300,0],[300,300]])

	M = cv2.getPerspectiveTransform(pts1,pts2)

	dst = cv2.warpPerspective(img,M,(300,300))

	plt.subplot(121),plt.imshow(img),plt.title('Input')
	plt.subplot(122),plt.imshow(dst),plt.title('Output')
	plt.show()

def testImgThreshod():
	#orig = cv2.imread('./images/lock.jpg',0)
	img = cv2.imread('./images/free.jpg',0) #comp.jpg eme.jpg building.jpg lock.jpg food.jpg
	#img = cv2.medianBlur(img,5)

	# ret,gth1 = cv2.threshold(img,5,255,cv2.THRESH_BINARY)
	# ret,gth2 = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
	# ret,gth3 = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
	# ret,gth4 = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
	# ret,gth5 = cv2.threshold(img,200,255,cv2.THRESH_BINARY)

	ret,gth1 = cv2.threshold(img,5,255,cv2.THRESH_TRUNC)
	ret,gth2 = cv2.threshold(img,5,255,cv2.THRESH_BINARY)
	ret,gth3 = cv2.threshold(img,100,255,cv2.THRESH_TRUNC)
	ret,gth4 = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
	ret,gth5 = cv2.threshold(img,200,255,cv2.THRESH_TRUNC)
	ret,gth6 = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
	
	th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)

	th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

	titles = ['Modified Image' , 'Global Thresholding (v = 5, trunc)',
	            'Global Thresholding (v = 5)', 'Global Thresholding (v = 100, trunc)', 'Global Thresholding (v = 100)', 'Global Thresholding (v = 200, trunc)',  'Global Thresholding (v = 200)', 'ADAPTIVE_THRESH_GAUSSIAN']
	images = [img, gth1, gth2, gth3, gth4, gth5, gth6, th3]

	for i in xrange(8):
	    plt.subplot(3,3,i+1),plt.imshow(images[i],'gray')
	    plt.title(titles[i])
	    plt.xticks([]),plt.yticks([])
	plt.show()

testImgThreshod()