#!/usr/bin/python

from multiprocessing import Process, Value, Array, Queue
import time
import cv2
import sys
import numpy as np

face_cascade = cv2.CascadeClassifier('front_face.xml')

def detector(sendQueue,receiveQueue,frame,faces,exitFlag,facesNumber):
   print("Detector starting...") 
   frame, currentFaces, exitFlag, currentFacesNumber = detectSingleFace(frame,faces,exitFlag,facesNumber,sendQueue)
   while True:
      while(receiveQueue.empty() == False):
         print("Detector received message ")
         msg = receiveQueue.get()
         frame,temp1,exitFlag,temp2 = msg
         frame, currentFaces, exitFlag, currentFacesNumber = detectSingleFace(frame,currentFaces,exitFlag,currentFacesNumber,sendQueue)
         if exitFlag == 1:
            break
      if exitFlag == 1:
         break
   print("Detector exiting...")


def tracker(receiveQueue,sendQueue,frame,faces,exitFlag,facesNumber):
   print("Tracker starting...")
   boxes = np.zeros([1,4])
   currentFrame, faces, exitFlag, facesNumber, boxes = trackFaces(frame,faces,exitFlag,facesNumber,boxes,receiveQueue,sendQueue)
   while True:
      while(receiveQueue.empty() == False):
         print("Tracker received message ")
         msg = receiveQueue.get()
         temp,faces,exitFlag,facesNumber = msg
         if exitFlag == 1:
            break
         currentFrame, faces, exitFlag, facesNumber, boxes = trackFaces(currentFrame,faces,exitFlag,facesNumber,boxes,receiveQueue,sendQueue)
      if exitFlag == 1:
         break
      currentFrame, faces, exitFlag, facesNumber, boxes = trackFaces(currentFrame,faces,exitFlag,facesNumber,boxes,receiveQueue,sendQueue)
   print("Tracker exiting...")


def detectSingleFace(frame,faces,exitFlag,facesNumber,sendQueue):

   # Display result
   cv2.imshow("Detector", frame)

   faces = face_cascade.detectMultiScale(frame, 1.3, 5) # haar cascade face detection

   if len(faces) != facesNumber:
      print("Detected new faces...")
      facesNumber = len(faces)
      sendQueue.put([frame ,faces ,exitFlag,facesNumber])
      print(sendQueue.empty())

   # Exit if ESC pressed
   k = cv2.waitKey(30) & 0xff
   if k == 27:
      exitFlag = 1
      sendQueue.put([frame ,faces ,exitFlag,facesNumber])

   return frame,faces,exitFlag,facesNumber


def trackFaces(frame,faces,exitFlag,facesNumber,boxes,receiveQueue,sendQueue):

   boxes = boxes
   while(facesNumber != 0 and exitFlag == 0):

      # Create tracker
      tracker = cv2.MultiTracker_create()

      for face in faces:
         ok = tracker.add(cv2.TrackerKCF_create(), frame, (int(face[0]),int(face[1]),int(face[2]),int(face[3])))

      #time = cv2.getTickCount()
      while (exitFlag == 0):

         #print("Tracking...")

         # Start timer
         timer = cv2.getTickCount()

         # Update tracker
         ret, boxes = tracker.update(frame)

         # Calculate Frames per second (FPS)
         fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

         if ret:  # Tracking success
            for box in boxes:
               topLeft = (int(box[0]), int(box[1]))
               bottomDown = (int(box[0] + box[2]), int(box[1] + box[3]))
               cv2.rectangle(frame, topLeft, bottomDown, (255,0,0), 2, 1)  # Draw box
         else:
            break

         # Display tracker type on frame
         cv2.putText(frame,"KCF Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

         # Display FPS on frame
         cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

         # Display result
         cv2.imshow("Tracking", frame)

         # Read frame
         ok, frame = video.read()
         frame = cv2.flip(frame, 1)   #flip image to adjust it
         if(sendQueue.empty() == True):   #if detector did not consume the frame continue
            sendQueue.put([frame ,faces ,exitFlag,facesNumber])

         # Exit if ESC pressed
         k = cv2.waitKey(1) & 0xff
         if k == 27:
            exitFlag = 1
            sendQueue.put([frame ,faces ,exitFlag,facesNumber])
            break

         if(receiveQueue.empty() == 0):  # new face
            break

      print("Exited Tracking...")

      # Read frame
      ok, frame = video.read()
      frame = cv2.flip(frame, 1)   #flip image to adjust it
      if(sendQueue.empty() == True):
         sendQueue.put([frame ,faces ,exitFlag,facesNumber])

      # Exit if ESC pressed
      k = cv2.waitKey(1) & 0xff
      if k == 27:
         exitFlag = 1
         sendQueue.put([frame ,faces ,exitFlag,facesNumber])
         break

      for box in boxes:
         topLeft = (int(box[0]), int(box[1]))
         bottomDown = (int(box[0] + box[2]), int(box[1] + box[3]))
         cv2.rectangle(frame, topLeft, bottomDown, (255,0,0), 2, 1)  # Draw box

      # Display result
      cv2.imshow("Tracking", frame)

      if(receiveQueue.empty() == 0):  # new face
         break

   # Read frame
   ok, frame = video.read()
   frame = cv2.flip(frame, 1)   #flip image to adjust it
   if(sendQueue.empty() == True):
      sendQueue.put([frame ,faces ,exitFlag,facesNumber])

   # Exit if ESC pressed
   k = cv2.waitKey(1) & 0xff
   if k == 27:
      exitFlag = 1
      sendQueue.put([frame ,faces ,exitFlag,facesNumber])

   for box in boxes:
      topLeft = (int(box[0]), int(box[1]))
      bottomDown = (int(box[0] + box[2]), int(box[1] + box[3]))
      cv2.rectangle(frame, topLeft, bottomDown, (255,0,0), 2, 1)  # Draw box

   # Display result
   cv2.imshow("Tracking", frame)

   return frame,faces,exitFlag,facesNumber,boxes



if __name__ == '__main__':

   # Read video
   video = cv2.VideoCapture(0)
   video.set(cv2.CAP_PROP_FRAME_WIDTH,640);
   video.set(cv2.CAP_PROP_FRAME_HEIGHT,480);

   # Exit if video not opened.
   if not video.isOpened():
      print("Could not open video")
      sys.exit()

   # Read first frame.
   ok, frame = video.read()
   frame = cv2.flip(frame, 1)#flip image to adjust it

   if not ok:
      print("Cannot read video file")
      sys.exit()

   q1 = Queue()
   q2 = Queue() 

   # Create new processes
   facesDetector = Process(target = detector, args = (q1,q2,frame,np.zeros([1,4]),0,0))
   facesTracker = Process(target = tracker, args = (q1,q2,frame,np.zeros([1,4]),0,0))

   # Start new processes
   facesDetector.start()
   facesTracker.start()

   facesDetector.join()
   facesDetector.terminate()
   facesTracker.terminate()

   video.release()
   cv2.destroyAllWindows()
   
      
