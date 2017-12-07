import numpy as np
import cv2
rameHistory=400
displacement_threshold_min=20
displacement_threshold_max=200

swipe_threshold = 150
swipe_left=0
swipe_right=1
swipe_top=2
swipe_bottom=3
tempKernel = np.zeros((63, 63), np.double)  # filter to calculate edges within a box
cv2.circle(tempKernel, (31, 31), 1, 1, 30)
handKernel=tempKernel
handThickness=40