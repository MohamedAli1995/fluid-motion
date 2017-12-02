import numpy as np
from collections import deque
import action
import cv2
from datetime import datetime
#import matplotlib.pyplot as plt
import time
import sys
from multiprocessing import Process, Value, Array, Queue


cap = cv2.VideoCapture(0)
boxA=0
boxB=0
topLeft = (0, 0)
bottomDown = (0,0)

###########################################################################################################################################



def detector(sendQueue,receiveQueue,_frame,faces,exitFlag,facesNumber):
    #print("Detector starting...") 
    try:
      _frame, currentFaces, exitFlag, currentFacesNumber = detectSingleFace(_frame,faces,exitFlag,facesNumber,sendQueue)
    except:
      print("caught it boss!")
    currentFacesNumber=0
    currentFaces=np.zeros([1,4])

    while True:
      while(receiveQueue.empty() == False):
         #print("Detector received message ")
         msg = receiveQueue.get()
         _frame,temp1,exitFlag,temp2 = msg
         _frame, currentFaces, exitFlag, currentFacesNumber = detectSingleFace(_frame,currentFaces,exitFlag,currentFacesNumber,sendQueue)
         if exitFlag == 1:
            break
      if exitFlag == 1:
         break

   #print("Detector exiting...")


def tracker(receiveQueue,sendQueue,mainQueue,_frame,faces,exitFlag,facesNumber):
   #print("Tracker starting...")
   boxes = np.zeros([1,4])
   ok,_frame=cap.read()
   currentFrame=_frame
   
   while True:
      while(receiveQueue.empty() == False):
         #print("Tracker received message ")
         msg = receiveQueue.get()
         temp,faces,exitFlag,facesNumber = msg
         if exitFlag == 1:
            break
         currentFrame, faces, exitFlag, facesNumber, boxes = trackFaces(currentFrame,faces,exitFlag,facesNumber,boxes,receiveQueue,sendQueue,mainQueue)
      if exitFlag == 1:
         break
      currentFrame, faces, exitFlag, facesNumber, boxes = trackFaces(currentFrame,faces,exitFlag,facesNumber,boxes,receiveQueue,sendQueue,mainQueue)
   #print("Tracker exiting...")


def detectSingleFace(_frame,faces,exitFlag,facesNumber,sendQueue):

   # Display result
   #cv2.imshow("Detector", _frame)

   faces = face_cascade.detectMultiScale(_frame, 1.3, 5) # haar cascade face detection
   if len(faces) != facesNumber:
      #print("Detected new faces...")
      facesNumber = len(faces)
      sendQueue.put([_frame ,faces ,exitFlag,facesNumber])

   # # Exit if ESC pressed
   # k = cv2.waitKey(30) & 0xff
   # if k == 27:
   #    exitFlag = 1
   #    sendQueue.put([_frame ,faces ,exitFlag,facesNumber])

   return _frame,faces,exitFlag,facesNumber


def trackFaces(currentFrame,faces,exitFlag,facesNumber,boxes,receiveQueue,sendQueue,mainQueue):

  boxesA = boxes
  while(facesNumber != 0 and exitFlag == 0):
    # Create tracker
    tracker = cv2.MultiTracker_create()
    for face in faces:
       ok = tracker.add(cv2.TrackerKCF_create(), currentFrame, (int(face[0]),int(face[1]),int(face[2]),int(face[3])))

    #time = cv2.getTickCount()
    while (exitFlag == 0):

      #print("Tracking...")

      # Start timer
      timer = cv2.getTickCount()

      # Read frame
      ok, currentFrame = cap.read()
      currentFrame = cv2.flip(currentFrame, 1)   #flip image to adjust it
      # Update tracker
      ret, boxes = tracker.update(currentFrame)

      # Calculate Frames per second (FPS)
      fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
      
      if ret:  # Tracking success
        boxesA=boxes  
        print("tracked succesfully")
      else:
        break
      if len(boxesA) == 0:
        print("zero size!!")

      for box in boxesA:
        topLeft = (int(box[0]), int(box[1]))
        #TODO://remove this!
        boxA=box[2]
        boxB=box[3]
        bottomDown = (int(box[0] + box[2]), int(box[1] + box[3]))
        currentFrame[topLeft[1]:bottomDown[1] ,topLeft[0]:bottomDown[0]] = [0,0,0];
        #cv2.rectangle(currentFrame, topLeft, bottomDown, (255,0,0), 2, 1)  # Draw box     

      # Display tracker type on frame
      #cv2.putText(currentFrame,"KCF Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

      # Display FPS on frame
      #cv2.putText(currentFrame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

      # Display result
      #cv2.imshow("Tracking", currentFrame)

      

      if(sendQueue.empty() == True):   #if detector did not consume the frame continue
        sendQueue.put([currentFrame ,faces ,exitFlag,facesNumber])
 
      #Check if main program did not take the last frame update the frame
      try:
        mainQueue.get_nowait()
        mainQueue.put([currentFrame])
      except:
        mainQueue.put([currentFrame])

      # Exit if ESC pressed
      # k = cv2.waitKey(1) & 0xff
      # if k == 27:
      #   exitFlag = 1
      #   sendQueue.put([currentFrame ,faces ,exitFlag,facesNumber])
      #   break

      if(receiveQueue.empty() == 0):  # new face
        break

    #print("Exited Tracking...")
    # Read frame
    ok, currentFrame = cap.read()
    currentFrame = cv2.flip(currentFrame, 1)   #flip image to adjust it
    if(sendQueue.empty() == True):
      sendQueue.put([currentFrame ,faces ,exitFlag,facesNumber])

    # Exit if ESC pressed
    # k = cv2.waitKey(1) & 0xff
    # if k == 27:
    #   exitFlag = 1
    #   sendQueue.put([currentFrame ,faces ,exitFlag,facesNumber])
    #   break

    for box in boxesA:
        topLeft = (int(box[0]), int(box[1]))
        #TODO://remove this!
        boxA=box[2]
        boxB=box[3]
        bottomDown = (int(box[0] + box[2]), int(box[1] + box[3]))
        currentFrame[topLeft[1]:bottomDown[1] ,topLeft[0]:bottomDown[0]] = [0,0,0];

    #Check if main program did not take the last frame update the frame
    try:
      mainQueue.get_nowait()
      mainQueue.put([currentFrame])
    except:
      mainQueue.put([currentFrame])

    # Display result
    #cv2.imshow("Tracking", currentFrame)

    if(receiveQueue.empty() == 0):  # new face
      break

  # Read frame
  
  ok, currentFrame = cap.read()
  #print("camera frame :")
  #print(ok)
  currentFrame = cv2.flip(currentFrame, 1)   #flip image to adjust it
  tempFrame=currentFrame
  if(sendQueue.empty() == True):
    sendQueue.put([currentFrame ,faces ,exitFlag,facesNumber])

  # Exit if ESC pressed
  # k = cv2.waitKey(1) & 0xff
  # if k == 27:
  #   exitFlag = 1
  #   sendQueue.put([currentFrame ,faces ,exitFlag,facesNumber])

  

  for box in boxesA:
        topLeft = (int(box[0]), int(box[1]))
        #TODO://remove this!
        boxA=box[2]
        boxB=box[3]
        bottomDown = (int(box[0] + box[2]), int(box[1] + box[3]))
        currentFrame[topLeft[1]:bottomDown[1] ,topLeft[0]:bottomDown[0]] = [0,0,0];

  #Check if main program did not take the last frame update the frame
  try:
    mainQueue.get_nowait()
    mainQueue.put([currentFrame])
  except:
    mainQueue.put([currentFrame])


  # Display result
  #cv2.imshow("Tracking", currentFrame)
  #print("SDGSDGDS")
  return tempFrame,faces,exitFlag,facesNumber,boxesA

##################################################################################################################

