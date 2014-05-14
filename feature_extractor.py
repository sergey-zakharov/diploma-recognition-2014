import cv2
import numpy as np

def get_greyscale_hist_features(im):
    ''' Output 256 features' values of grayscale histogram '''
    h = np.zeros((300,256,3))
    if len(im.shape)!=2:
        print "hist_lines applicable only for grayscale images"
        #print "so converting image to grayscale for representation"
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    hist_item = cv2.calcHist([im],[0],None,[256],[0,256])
    print "hist_item", hist_item
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
    print "hist_item norm", hist_item
    hist=np.int32(np.around(hist_item))
    print "hist", hist
    result = []
    for value in hist:
        result.append(value[0])
    return result

if __name__ == '__main__':
    import sys

    if len(sys.argv)>1:
        im = cv2.imread(sys.argv[1])
    else :
        im = cv2.imread('./images/food.jpg')
        print "usage : python hist.py <image_file>"
    print get_greyscale_hist_features(im)