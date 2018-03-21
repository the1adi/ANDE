from multiprocessing import Process, Queue
import cv2
import numpy as np
import requests
import polyline
import json
import time
import math
import main
import time

start_time = time.time()

count = 0
image = cv2.imread('./glare.jpg')
B, G, R = cv2.split(image)

#cv2.imshow('Original Image',image)
#cv2.waitKey(0)
'''=
zeros = np.zeros(image.shape[:2], dtype = "uint8")
#cv2.imshow('Red',cv2.merge([zeros,zeros,R]))
#cv2.imshow('Green',cv2.merge([zeros,G,zeros]))
#cv2.imshow('Blue',cv2.merge([B,zeros,zeros]))
#cv2.imshow('Grayscale',gray_image)
cv2.waitKey(0)



#--------------------------------------------------------GRAYSCALE MANIPULATION----------------------------------------------------#

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#print("The shape of the grayscale is ",gray_image.shape)

#--------------------------------------------------------GRAYSCALE MANIPULATION----------------------------------------------------#


#--------------------------------------------------------HSV MANIPULATION----------------------------------------------------#
'''


hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

H, S, V = cv2.split(hsv_image)

height, width, channels = hsv_image.shape
print("The image height is {} and its width is {}\n".format(height, width))
#cv2.imshow('Value channel', hsv_image[:, :, 2])


#height = # of rows
#width =  # of columns


for i in range(0,height-1):
    for j in range(0,width-1):
        if (V[i][j] > 50):
            V[i][j] = 45.954*math.log10(1+V[i][j])





merged_image = cv2.merge([H, S, V])
new_RGB = cv2.cvtColor(merged_image, cv2.COLOR_HSV2BGR)
cv2.imshow('Original Image',image)

print(time.time() - start_time)
#cv2.imshow('Merged Image',new_RGB)


print('The total number of values above the range is {}'.format(count))




cv2.waitKey(0)
cv2.destroyAllWindows()



