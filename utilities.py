import numpy as np
from collections import deque
import action
import cv2
from datetime import datetime
#import matplotlib.pyplot as plt
import time
import sys
from multiprocessing import Process, Value, Array, Queue

def euclidian_dist(r1,r2):
   return  np.sqrt(np.square(r1[0] - r2[0]) + np.square(r1[1] - r2[1]))
