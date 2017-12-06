import numpy as np
from collections import deque
import constants as cnst


class ActionDetector:
    def __init__(self,tracing_points_max):
        self.tracing_points_max=tracing_points_max
        self.tracing_points = deque([])
        self.ready=False
        self.swipe_threshold=cnst.swipe_threshold
        self.swipe=False
        self.angle=0
    def updateTracingPoints(self,x, y):
        # global tracing_points
        if self.tracing_points.__len__() == self.tracing_points_max:
            # ready when filled with initial points
            self.ready=True
            self.tracing_points.popleft()
            self.tracing_points.append([x, y])
            return True
        self.tracing_points.append([x, y])
        return False
    def getTracingPoints(self):
        return self.tracing_points

    def getAction(self):
        if self.ready==False:
            return 0
        arr=np.asarray(self.tracing_points)
        dist=np.linalg.norm(arr[0]-arr[self.tracing_points_max-1])
        if dist>self.swipe_threshold:
            self.swipe=True
        else:
            self.swipe=False
        if self.swipe:
            self.ready=False
            #check if swipe left
            #calculate angle
            diff=arr[0]-arr[self.tracing_points_max-1]
            diff_x=diff[0]
            diff_y=diff[1]
            if (diff_x>diff_y) and diff_x>0:
                print("swipe left ")
            elif (diff_x > diff_y) and diff_x <= 0:
                print("swipe right ")
            elif diff_x <= diff_y and diff_y > 0:
                print("swipe down ")
            elif diff_x <= diff_y and diff_y <= 0:
                print("swipe up ")
            self.tracing_points.clear()


