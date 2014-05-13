import cv2
import numpy as np

def hist_lines(im):
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
    for x,y in enumerate(hist):
        cv2.line(h,(x,0),(x,y),(255,255,255))
    print 'h=', h
    print "h type" ,type(h)
    y = np.flipud(h)
    print 'y=', y
    return y

def get_grey_hist_by_image(im):
    h = np.zeros((300,256,3))
    if len(im.shape)!=2:
        print "hist_lines applicable only for grayscale images"
        #print "so converting image to grayscale for representation"
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    hist_item = cv2.calcHist([im],[0],None,[256],[0,256])
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
    hist=np.int32(np.around(hist_item))
    return hist

if __name__ == '__main__':

    import sys

    if len(sys.argv)>1:
        im = cv2.imread(sys.argv[1])
    else :
        im = cv2.imread('./images/food.jpg')
        print "usage : python hist.py <image_file>"
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    while True:
        k = cv2.waitKey(0)&0xFF
        lines = hist_lines(im)
        cv2.imshow('histogram',lines)
        cv2.imshow('image',gray)
        if k == 27:
            print 'ESC'
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()