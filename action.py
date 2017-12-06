import numpy as np
from collections import deque



class ActionDetector:
    def __init__(self,tracing_points_max):
        self.tracing_points_max=tracing_points_max
        self.tracing_points = deque([])
        self.ready=False
        self.swipe_threshold=200
        self.swipe=False
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
            self.tracing_points.clear()
            self.ready=False
            print("swipped with dist",dist)

