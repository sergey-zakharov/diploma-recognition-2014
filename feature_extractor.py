import cv2
import numpy as np


class ImageManager:
    image_dict ={}
    filename = ""
    def __init__(self, image_dict_filename):
        self.filename = image_dict_filename

    def loadImageDict(self):
         # get image_map file and creat dictionary of name and real file name
        image_map_file = open(self.filename, 'r+')
        for line in image_map_file:
            splitted = line.split(' ')
            self.image_dict[splitted[0]] = splitted[1].replace('\n', '')

    def getNewNameFromImageDict(self):
        return str(int(self.image_dict[max(self.image_dict, key=self.image_dict.get)]) + 1)

    def writeToImageDict(self, filename):
        new_name = str(self.getNewNameFromImageDict())
        self.image_dict[filename] = new_name
        with open("./learn_data/image_map", "a") as myfile:
            myfile.write(filename + " " + new_name +"\n")
        self.loadImageDict()
        return new_name

    def get_numeral_filename(self, filename):
        if filename in self.image_dict:
            return self.image_dict[filename]
        else:
            return self.writeToImageDict(filename)
    def getImageDict(self):
        return self.image_dict

def get_greyscale_hist_features(im):
    ''' Outputs 64 features' values of grayscale histogram '''
    h = np.zeros((300,256,3))
    if len(im.shape)!=2:
        print "hist_lines applicable only for grayscale images"
        #print "so converting image to grayscale for representation"
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    hist_item = cv2.calcHist([im],[0],None,[64],[0,256])
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

def countPointsWithNeighboursOfSameColour(image, num):
    i = 1
    image = image.tolist()
    total = len(image[0])*len(image)
    #print "Total points:", total
    result = 0
    while i < len(image)-1:
        j = 1
        line = image[i]
        while j < len(line)-1:
            count = 0
            if line[j-1]==line[j]: # left
                count+=1
            if line[j+1]==line[j]: # right
                count+=1
            if image[i-1][j]==line[j]: # top
                count+=1
            if image[i+1][j]==line[j]: # bottom
                count+=1
            if image[i-1][j-1]==line[j]: # top left
                count+=1
            if image[i-1][j+1]==line[j]: # top right
                count+=1
            if image[i+1][j-1]==line[j]: # bottom left
                count+=1
            if image[i+1][j+1]==line[j]: # bottom right
                count+=1
            if count >= num:
                result+=1
            j+=1
        i+=1
    return result, total



if __name__ == '__main__':
    import sys

    if len(sys.argv)>1:
        im_manager = ImageManager("./learn_data/image_map")
        filename = sys.argv[1]
        im = cv2.imread(filename)
        n = 1
        num, tot = countPointsWithNeighboursOfSameColour(im, n)
        print "Percent of points with same", n , "neighbours:", "{0:.2f}".format(float(num)/float(tot)*100), "%"
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        #print get_greyscale_hist_features(im)
        im_manager.loadImageDict()
        hist_values_list = get_greyscale_hist_features(gray)
        hist_values_list = map(toString, hist_values_list)
        numeral_filename = im_manager.get_numeral_filename(filename.split('/')[-1])
        print "Get image " + filename.split('/')[-1]
        print "Write feature to ./learn_data/" + numeral_filename + '.txt'
        with open("./learn_data/" + numeral_filename+'.txt', "a") as myfile:
            myfile.write('\n'.join((hist_values_list)))
    else:
        print "usage : python feature_extractor.py <image_file>"


