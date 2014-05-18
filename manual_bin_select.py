import cv2
import numpy as np
import sys
from subprocess import call

img = []
resimg = []
global_threshold = 0
is_init_image = True
threshold_type = cv2.THRESH_BINARY
threshold_type_name = "cv2.THRESH_BINARY"
image_dict = {}
image_directory = ""
globalblurRate = 0

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

#######################################################################################################

def updateImageForSelector(threshold):
    global img
    global resimg
    global global_threshold
    global is_init_image
    global threshold_type

    global_threshold = threshold = cv2.getTrackbarPos('threshold','image')
    blur = img
    if globalblurRate != 0:
        blur = cv2.GaussianBlur(img,(globalblurRate, globalblurRate),0)
    ret, resimg = cv2.threshold(blur, threshold, 255, threshold_type)
    cv2.imshow('image',resimg)
    is_init_image = False

def updateImageForAdaptiveThresChange():
    global img
    global resimg
    global global_threshold
    global threshold_type
    global threshold_type_name

def updateImageForOtsuThresChange(otsu_is_on):
    global img
    global resimg
    global global_threshold
    global threshold_type
    global threshold_type_name
    global globalblurRate
    #otsu_is_on = cv2.getTrackbarPos('threshold','image')
    if otsu_is_on:
        threshold_type += cv2.THRESH_OTSU
        threshold_type_name += "+cv2.THRESH_OTSU"
    else:
        threshold_type -= cv2.THRESH_OTSU
        threshold_type_name.replace("+cv2.THRESH_OTSU", "")
    blur = img
    if globalblurRate != 0:
        blur = cv2.GaussianBlur(img,(globalblurRate, globalblurRate),0)
    ret, resimg = cv2.threshold(blur, global_threshold, 255, threshold_type)
    cv2.imshow('image',resimg)
    is_init_image = False

def updateImageForGaussianBlurChange(blurRate):
    global img
    global resimg
    global global_threshold
    global threshold_type
    global threshold_type_name
    global globalblurRate

    if blurRate%2 == 1:
        globalblurRate = blurRate
        blur = cv2.GaussianBlur(img,(globalblurRate, globalblurRate),0)
        ret, resimg = cv2.threshold(blur, global_threshold, 255, threshold_type)
        cv2.imshow('image',resimg)
        is_init_image = False
    elif blurRate == 0:
        globalblurRate = blurRate
    else:
        pass

def updateImageDict():
    global image_dict
    global image_directory

     # get image_map file and creat dictionary of name and real file name
    image_map_file = open('./'+image_directory+'/image_map', 'r+')
    for line in image_map_file:
        splitted = line.split(' ')
        image_dict[splitted[0]] = splitted[1].replace('\n', '')

def getNewNameFromImageDict():
    global image_dict
    try:
        return str(int(image_dict[max(image_dict, key=image_dict.get)]) + 1)
    except ValueError:
        return 1

def writeToImageDict(filename):
    global image_dict
    new_name = str(getNewNameFromImageDict())
    image_dict[filename] = new_name
    with open("./"+image_directory+"/image_map", "a") as myfile:
        myfile.write(filename + " " + new_name +"\n")
    updateImageDict()
    return new_name

def manualThresholdTypeSelector():
    # Create a black image, a window
    global img
    global resimg
    global global_threshold
    global threshold_type
    global is_init_image
    global image_dict
    global image_directory

    if len(sys.argv) != 2:
        print "Pass 1 argument - path to image, please"
        print 'there are ' + str(len(sys.argv)) + ' params'
        print 'Params: '+ str(sys.argv)
        return
    resimg = img = cv2.imread(sys.argv[1],0)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('threshold','image', 0, 255, updateImageForSelector)
    cv2.createTrackbar('isOtsu','image', 0, 1, updateImageForOtsuThresChange)
    cv2.createTrackbar('gaussianBlur','image', 3, 21, updateImageForGaussianBlurChange)
    # TODO create several checkboxes for adaptive types
    cv2.imshow('image',img)
    splitted_path = sys.argv[1].split('/')
    filename = splitted_path[-1]
    image_directory = splitted_path[1]
    print "Image directory", image_directory    
    updateImageDict()
    try:
        print "press 's' to recognize and save "
        print "press 'esc' to exit"
        while(1):
            
            k = cv2.waitKey(1) #& 0xFF
            if k == 27:
                break
            elif k == 115: # 's'
                ending = ''
                if is_init_image:
                    ending = "-init"
                full_name = filename
                if full_name not in image_dict:
                    print image_dict
                    name = writeToImageDict(filename)
                else:
                    name = image_dict[filename]
                save_path = "./"+image_directory+"/" + name + ".jpg"
                bin_type_path = "./"+image_directory+"/" + name + "-method" # ex: cv2.THRESH_BINARY 123
                recognize_path = "./"+image_directory+"/" + name + "-recog-result" + ending
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
                    threshold_text = 'Threshold: ' + threshold_type_name +' ' + str(global_threshold) + "/255"
                    threshold_type_text = threshold_type_name +' ' + str(global_threshold) + "\nblur: " + str(globalblurRate)
                    with open(recognize_path+'.txt', "a") as myfile:
                        myfile.write(line + '\n' + threshold_text)
                    #myfile.closed
                    if not is_init_image:
                        with open(bin_type_path+'.txt', "w+") as myfile:
                            myfile.write(threshold_type_text)
                    #myfile.closed
                else:
                    print 'Failed to save'
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()

#testSwitchersExample()
#thresholdSwitcher()
manualThresholdTypeSelector()