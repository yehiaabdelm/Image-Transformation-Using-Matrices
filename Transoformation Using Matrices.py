import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import PyQt5
%matplotlib qt5

# Notes: This works but it's too hardcoded. Also, we need to find out how much a picture is scaled down.


# Scaling and warping

windows = cv2.imread('windows.jpeg')
windows

img_copy = np.copy(windows)
img_copy = cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB)

plt.imshow(img_copy)

img_coor = np.float32([[55,130],[443,297],[56,318],[441,400]])

for i in range(0,4):
    cv2.circle(img_copy,(img_coor[i][0],img_coor[i][1]),5,(255,0,0),-1)

cv2.imshow('coord',img_copy)
cv2.waitKey(0)

height, width = 100, 200

output_pts = np.float32([[0, 0],[width, 0],[0, height],[width , height]])

M = cv2.getPerspectiveTransform(img_coor,output_pts)
M
out = cv2.warpPerspective(windows,M,(width,height),flags=cv2.INTER_LINEAR)

cv2.namedWindow('image')
cv2.imshow('image', out)
cv2.waitKey(0)
