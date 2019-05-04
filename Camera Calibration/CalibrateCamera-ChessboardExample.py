import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

image = mpimg.imread('calibration_test.png')

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

nx = 8
ny = 6
retVal, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# Draw detected corners on the image
img = cv2.drawChessboardCorners(gray, (nx, ny), corners, retVal)
