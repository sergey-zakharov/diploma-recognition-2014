import cv2
import numpy as np
import sys
from subprocess import call

img = []
resimg = []
global_threshold = 0
is_init_image = True
def updateImage(threshold):
    global img
    global resimg
    global global_threshold
    global is_init_image
    global_threshold = threshold = cv2.getTrackbarPos('threshold','image')
    ret, resimg = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
    cv2.imshow('image',resimg)
    is_init_image = False

def thresholdSwitcher():
    # Create a black image, a window
    global img
    global resimg
    global global_threshold
    global is_init_image
    if len(sys.argv) != 2:
        print "Pass 1 argument - path to image, please"
        print 'there are ' + str(len(sys.argv)) + ' params'
        print 'Params: '+ str(sys.argv)
        return
    resimg = img = cv2.imread(sys.argv[1],0)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    print img
    # create trackbars for color change
    cv2.createTrackbar('threshold','image',0,255,updateImage)
    cv2.imshow('image',img)
    
    try:
        print "press 's' to recognize and save "
        print "press 'esc' to exit"
        while(1):
            
            k = cv2.waitKey(1) #& 0xFF
            if k == 27:
                break
            elif k == 115: # 's'
                splitted_path = sys.argv[1].split('/')
                filename = splitted_path[-1].split('.')[0]
                ending = ''
                if is_init_image:
                    ending = "-init"
                else:
                    ending = "-thres"
                save_path = '/'.join(splitted_path[:-1]) + "/thresholded/" + filename + "-thres.jpg"
                recognize_path = '/'.join(splitted_path[:-1]) + "/thresholded/" + filename + ending
                ret = cv2.imwrite(save_path,resimg) #sys.argv[1][:-4] + "-thres.jpg"
                if ret == 1:
                    print 'Image saved to ' + save_path
                    call_tesseract_list = ["tesseract", save_path, recognize_path]
                    print 'Recognition: ' + str(call_tesseract_list)
                    call(call_tesseract_list)
                    call_cat = ["cat", recognize_path+".txt"]
                    print('------Recognition result-------')
                    call(call_cat)
                    print('-------------------------------')
                    print('Threshold: ' + str(global_threshold) + "/255")
                    print('-------------------------------')
                    line = '-------------------------------'
                    threshold_text = 'Threshold: ' + str(global_threshold) + "/255"
                    with open(recognize_path+'.txt', "a") as myfile:
                        myfile.write(line + '\n' + threshold_text)
                else:
                    print 'Failed to save'
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()

#testSwitchersExample()
thresholdSwitcher()