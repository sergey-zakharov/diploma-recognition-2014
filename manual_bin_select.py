import cv2
import numpy as np
import sys
import os
from subprocess import call

img = []
resimg = []
global_threshold = 0
is_init_image = True
threshold_type = cv2.THRESH_BINARY
threshold_type_before_switch = cv2.THRESH_BINARY
threshold_type_name = "0" # 0 - "cv2.THRESH_BINARY", 1 - "cv2.ADAPTIVE_THRESH_MEAN_C"
threshold_type_name_before_switch = '0' #"cv2.THRESH_BINARY"
image_dict = {}
image_directory = ""
globalblurRate = 0
isInverted = False

def updateImage(threshold):
    global img
    global resimg
    global global_threshold
    global is_init_image
    global globalblurRate

    global_threshold = threshold = cv2.getTrackbarPos('block size','image')
    param = cv2.getTrackbarPos('const','image')
    #ret, resimg = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
    blur = img
    if globalblurRate != 0:
        blur = cv2.GaussianBlur(img,(globalblurRate, globalblurRate),0)
    resimg = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, threshold, param)
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
    cv2.createTrackbar('block size','image',0, 500, updateImage)
    cv2.createTrackbar('const','image',0, 100, updateImage)
    cv2.createTrackbar('gaussianBlur','image', 3, 21, updateImageForGaussianBlurChange)
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
                print save_path
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
    global isInverted

    global_threshold = threshold = cv2.getTrackbarPos('threshold','image')
    blur = img
    if globalblurRate != 0:
        blur = cv2.GaussianBlur(img,(globalblurRate, globalblurRate),0)
    if isInverted:
        blur = (255 - blur)
    #print threshold_type
    #if threshold_type == cv2.ADAPTIVE_THRESH_MEAN_C:
    #    resimg = cv2.adaptiveThreshold(blur, 255, threshold_type, cv2.THRESH_BINARY, 7, threshold)
    #else:
    ret, resimg = cv2.threshold(blur, threshold, 255, threshold_type)
    cv2.imshow('image',resimg)
    is_init_image = False

def updateImageForThresAdaptiveChange(is_adaptive_on):
    global img
    global resimg
    global global_threshold
    global threshold_type
    global threshold_type_before_switch
    global threshold_type_name
    global threshold_type_name_before_switch
    if is_adaptive_on:
        threshold_type_before_switch = threshold_type
        threshold_type = cv2.ADAPTIVE_THRESH_MEAN_C
        threshold_type_name_before_switch = threshold_type_name
        threshold_type_name = "1" # "cv2.ADAPTIVE_THRESH_MEAN_C"
    else:
        threshold_type = threshold_type_before_switch
        threshold_type_name = threshold_type_name_before_switch
    blur = img
    if globalblurRate != 0:
        blur = cv2.GaussianBlur(img,(globalblurRate, globalblurRate),0)
    print threshold_type_name
    resimg = cv2.adaptiveThreshold(blur, 255, threshold_type, cv2.THRESH_BINARY, 5, 5)
    cv2.imshow('image',resimg)
    is_init_image = False


def updateImageForOtsuThresChange(otsu_is_on):
    global img
    global resimg
    global global_threshold
    global threshold_type
    global threshold_type_name
    global globalblurRate
    global isInverted
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
    if threshold_type == cv2.ADAPTIVE_THRESH_GAUSSIAN_C:
        resimg = cv2.adaptiveThreshold(blur, 255, threshold_type, cv2.THRESH_BINARY, 5, global_threshold)
    else:
        ret, resimg = cv2.threshold(blur, threshold, 255, threshold_type)
    cv2.imshow('image',resimg)
    is_init_image = False

def updateImageForGaussianBlurChange(blurRate):
    global img
    global resimg
    global global_threshold
    global threshold_type
    global threshold_type_name
    global globalblurRate

    blur=img
    if blurRate%2 == 1:
        globalblurRate = blurRate
        blur = cv2.GaussianBlur(img,(globalblurRate, globalblurRate),0)
    elif blurRate == 0:
        globalblurRate = blurRate
    else:
        pass
    if threshold_type == cv2.ADAPTIVE_THRESH_GAUSSIAN_C:
        resimg = cv2.adaptiveThreshold(blur, 255, threshold_type, cv2.THRESH_BINARY, 5, global_threshold)
    else:
        ret, resimg = cv2.threshold(blur, global_threshold, 255, threshold_type)
    cv2.imshow('image',resimg)
    is_init_image = False

def updateImageDict():
    global image_dict
    global image_directory

     # get image_map file and creat dictionary of name and real file name
    image_map_file = open('./'+image_directory+'/image_map', 'r+')
    for line in image_map_file:
        splitted = line.split(' ')
        print splitted
        image_dict[splitted[0]] = int(splitted[1].replace('\n', ''))

def removeImageFromDict(image_directory, real_name, name, ending):
    # remove files
    delete_path = "./"+image_directory+"/" + name + ".jpg"
    bin_type_path = "./"+image_directory+"/" + name + "-method.txt"
    recognize_path = "./"+image_directory+"/" + name + "-recog-result" + ending + ".txt"
    try:
        print "deleting", delete_path
        os.remove(delete_path)
    except OSError:
        print "no file", delete_path
    try:
        print "deleting", bin_type_path
        os.remove(bin_type_path)
    except OSError:
        print "no file", bin_type_path
    try:
        print "deleting", recognize_path
        os.remove(recognize_path)
    except OSError:
        print "no file", recognize_path

    # remove from index
    f = open('./'+image_directory+'/image_map',"r")
    lines = f.readlines()
    f.close()
    f = open('./'+image_directory+'/image_map',"w")
    for line in lines:
        if line.find(real_name) == -1:
            f.write(line)
    f.close()

def getNewNameFromImageDict():
    global image_dict
    try:
        max_num = int(image_dict[max(image_dict, key=image_dict.get)])
        new_num = max_num + 1
        return str(new_num)
    except ValueError:
        return 1

def writeToImageDict(filename):
    global image_dict
    new_name = str(getNewNameFromImageDict())
    print "new_name", new_name
    image_dict[filename] = new_name
    with open("./"+image_directory+"/image_map", "a") as myfile:
        myfile.write(filename + " " + str(new_name) +"\n")
    updateImageDict()
    return str(new_name)

def updateInvertImage(val):
    global resimg
    global isInverted

    if val == 1:
        isInverted = True
    else:
        isInverted = False
    resimg = (255 - resimg)
    cv2.imshow('image',resimg)

def manualThresholdTypeSelector():
    # Create a black image, a window
    global img
    global resimg
    global global_threshold
    global threshold_type
    global is_init_image
    global image_dict
    global image_directory

    threshold_type = cv2.THRESH_BINARY
    if len(sys.argv) != 2:
        print "Pass 1 argument - path to image, please"
        print 'there are ' + str(len(sys.argv)) + ' params'
        print 'Params: '+ str(sys.argv)
        return
    resimg = img = cv2.imread(sys.argv[1],0)
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('threshold','image', 0, 255, updateImageForSelector)
    cv2.createTrackbar('invert','image', 0, 1, updateInvertImage)
    #cv2.createTrackbar('isOtsu','image', 0, 1, updateImageForOtsuThresChange)
    cv2.createTrackbar('gaussianBlur','image', 3, 21, updateImageForGaussianBlurChange)
    #cv2.createTrackbar('adaptiveThres','image', 0, 1, updateImageForThresAdaptiveChange)
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

        full_name = filename
        if full_name not in image_dict:
            print image_dict
            name = writeToImageDict(filename)
        else:
            name = str(image_dict[filename])
        while(1):
            k = cv2.waitKey(1) #& 0xFF
            if k == 27:
                break
            elif k == 114: # 'r'
                ending = ''
                if is_init_image:
                    ending = "-init"
                #remove image from set
                removeImageFromDict(image_directory, filename, name, ending)
            elif k == 115: # 's'
                ending = ''
                if is_init_image:
                    ending = "-init"
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
                    threshold_type_text = threshold_type_name +' ' + str(global_threshold) + "\nblur: " + str(globalblurRate) + "\ninverted: " + str(int(isInverted))
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

thresholdSwitcher()
#manualThresholdTypeSelector()