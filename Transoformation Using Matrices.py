import cv2
import numpy as np
import math

# Rotation
cow = cv2.imread('cow.jpeg')
height, width = cow.shape[:2]
center = (width/2, height/2)
rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=45, scale=3)
rotation_matrix

an_array = np.where(abs(0.70710678-rotation_matrix)<=0.1,0.5, rotation_matrix)

rotated_image = cv2.warpAffine(src=cow, M=rotation_matrix, dsize=(width, height))

cv2.namedWindow('image')
cv2.imshow('image', cow)
cv2.waitKey(0) # close window when a key press is detected
cv2.destroyWindow('image')
cv2.waitKey(1)

# Scaling and warping

windows = cv2.imread('windows.jpeg')
windows

import matplotlib.pyplot as plt
import PyQt5

%matplotlib qt5

img_copy = np.copy(windows)

# Convert to RGB so as to display via matplotlib
# Using Matplotlib we can easily find the coordinates
# of the 4 points that is essential for finding the
# transformation matrix
img_copy = cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB)

plt.imshow(img_copy)
x1 = [69.8,202.5]
x2 = [50,296]
y1 = [113,216]
y2 = [99,307]

def get_l2(p1,p2):
    return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

maxWidth = max(int(get_l2(x1,y1)), int(get_l2(x2,y2)))
maxHeight = max(int(get_l2(x1,x2)), int(get_l2(y1,y2)))

input_pts = np.float32([x1, x2, y1, y2])
output_pts = np.float32([[0, 0],
                        [0, maxHeight - 1],
                        [maxWidth - 1, maxHeight - 1],
                        [maxWidth - 1, 0]])

M = cv2.getPerspectiveTransform(input_pts,output_pts)
out = cv2.warpPerspective(windows,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)



cv2.namedWindow('image')
cv2.imshow('image', out)
cv2.waitKey(0)
