import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

# read 1st file annotation
file = open("L:/Spring 2021/DA 401/10k US Faces Data/annotations/Face Annotations/Images and Annotations/3_landmarks.txt", "r")
coordinates = file.read()
list_coordinates = coordinates.split()
x_coordinate, y_coordinate = [], []
for i in range(len(list_coordinates)):
    # even = x, odd = y
    if i % 2 == 0:
        x_coordinate.append(float(list_coordinates[i]))
    else:
        y_coordinate.append(float(list_coordinates[i]))

# read the image
img = plt.imread("L:/Spring 2021/DA 401/10k US Faces Data/annotations/Face Annotations/Images and Annotations/3.jpg")
bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # since opencv reads in BGR channel while the pic is in RGB channel
height, width, channels = bw_img.shape         # get dimensions to rescale later

for i in range(0, 8):
    # 0-7 is left eyebrow. 0 is tail of eyebrow. Points go clockwise
    # 8-23 is face shape. 8 is second point to the left of middle of forehead. Points go counter-clockwise
    # 24-31 is right eyebrow. 24 is lower point of head of eyebrow. Points go clockwise
    # 32-44 is nose. 32 is top left point. Points go counter-clockwise
    # 45-52 is left eye. 45 is eye outer corner, 49 is eye corner. Points go clockwise
    # 53-60 is right eye. 53 is eye corner. 57 is eye outer corner. Points go clockwise.
    # 61-69 is upper lip. 61 is left lip corner. Points go clockwise.
    # 70-76 is lower lip. 70 is lower left point of lip. Points go counter-clockwise
    bw_img[int(y_coordinate[i]), int(x_coordinate[i])] = [0, 0, 255]

# resize image
output = cv2.resize(bw_img, (width*2, height*2))

# print image
cv2.imshow('image', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# todo: repeat same for other areas of face
