import cv2
import numpy as np

image_dict ={}

def loadImageDict():
    global image_dict
     # get image_map file and creat dictionary of name and real file name
    image_map_file = open('./learn_data/image_map', 'r+')
    for line in image_map_file:
        splitted = line.split(' ')
        image_dict[splitted[0]] = splitted[1].replace('\n', '')

def getNewNameFromImageDict():
    global image_dict
    return str(int(image_dict[max(image_dict, key=image_dict.get)]) + 1)

def writeToImageDict(filename):
    global image_dict
    new_name = str(getNewNameFromImageDict())
    image_dict[filename] = new_name
    with open("./learn_data/image_map", "a") as myfile:
        myfile.write(filename + " " + new_name +"\n")
    loadImageDict()
    return new_name

def get_numeral_filename(filename):
    if filename in image_dict:
        return image_dict[filename]
    else:
        return writeToImageDict(filename)

def get_greyscale_hist_features(im):
    ''' Output 256 features' values of grayscale histogram '''
    h = np.zeros((300,256,3))
    if len(im.shape)!=2:
        print "hist_lines applicable only for grayscale images"
        #print "so converting image to grayscale for representation"
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    hist_item = cv2.calcHist([im],[0],None,[256],[0,256])
    #print "hist_item", hist_item
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
    #print "hist_item norm", hist_item
    hist=np.int32(np.around(hist_item))
    #print "hist", hist
    result = []
    for value in hist:
        result.append(value[0])
    return result

def toString(element):
    return str(element)

if __name__ == '__main__':
    import sys

    if len(sys.argv)>1:
        filename = sys.argv[1]
        im = cv2.imread(filename)
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        #print get_greyscale_hist_features(im)
        loadImageDict()
        hist_values_list = get_greyscale_hist_features(gray)
        hist_values_list = map(toString, hist_values_list)
        numeral_filename = get_numeral_filename(filename.split('/')[-1])
        print "Get image " + filename.split('/')[-1]
        print "Write feature to ./learn_data/" + numeral_filename + '.txt'
        with open("./learn_data/" + numeral_filename+'.txt', "a") as myfile:
            myfile.write('\n'.join((hist_values_list)))
    else:
        print "usage : python hist.py <image_file>"


