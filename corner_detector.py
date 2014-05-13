import cv2
import numpy as np

filename = './images/IMG_2544.JPG'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.01)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
height, width, depth = img.shape
new_image = np.zeros((height, width, depth), np.uint8)
new_image[:] = (255, 255, 255)
new_image[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',new_image)
cv2.imshow('orig',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()