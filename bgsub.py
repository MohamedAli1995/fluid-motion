import numpy as np

from collections import deque
import action
actionDetector=action.ActionDetector(10)
import cv2
from datetime import datetime

import matplotlib.pyplot as plt
import time;

frameHistory=800
hand_region_Window=[60,60,70,100]
wristx=0
wristy=0
faceExisted_once=False;
a=0
b=0
c=0
d=0

begin=datetime.now();
frames_count=0;
frame_max_x=0;
frame_max_y=0;
def updateHandRect(frame,maxLoc,skin_ycrcb):
    [x,y]=maxLoc
    [tempx,tempy]=maxLoc[0],maxLoc[1]
    [tempx_right, tempy_bottom] = maxLoc[0],maxLoc[1]
    while(np.sum(np.sum(frame[tempy-50:tempy+50,tempx-5:tempx+5]))!=0):
        tempx-=1
    while (np.sum(np.sum(frame[tempy - 50:tempy + 50, tempx_right -5:tempx_right +5]))!= 0 and tempx_right+5<frame_max_x):
        tempx_right += 1

    while (np.sum(np.sum(frame[tempy-5:tempy+5 , x-50:x+50])) != 0):
        tempy -= 1

    tempy_bottom=tempy+150 # TODO: make height as a ratio of width


    #if tempy or tempx falls below 0
    tempy=np.maximum(0,tempy)
    tempy=np.minimum(frame_max_y-1,tempy)

    tempy_bottom = np.maximum(0, tempy_bottom)
    tempy_bottom = np.minimum(frame_max_y - 1, tempy_bottom)

    tempx=np.maximum(0,tempx)
    tempx = np.minimum(frame_max_x - 1, tempx)

    tempx_right=np.maximum(0,tempx_right)
    tempx_right = np.minimum(frame_max_x - 1, tempx_right)
   # print("( ",tempx," , ",tempy," ) ")


    #print(" top left : (x1,y1)(x2,y2)  (",tempx," , ",tempy," ) (",tempx_right," , ",tempy_bottom," ) ")
    return [tempx,tempy,tempx_right,tempy_bottom]


def getWristPoints(frame,y1,y2,x1,x2):
    y1=np.maximum(0,y1)
    y2=np.maximum(0,y2)
    x1=np.maximum(0,x1)
    x2=np.maximum(0,x2)
    frame=frame[y1:y2,x1:x2]
    #print(frame.shape)

    edged_hand = cv2.Canny(frame, 35, 125)
    cv2.imshow("edged_hand",edged_hand)
    return [5,5]
start_time=time.time() #used to call fudnctions every x seconds
forget_frames=30;

call_time=0.5
faceExist = False;  # to figure out wether at least face exist or no
faceWasDetectedBefore=False;

face_cascade = cv2.CascadeClassifier('front_face.xml')
def detectSingleFace(frame):
    # haar cascade face detection
    faces = face_cascade.detectMultiScale(frame, 1.3, 5) #using haar cascade classifier to detect faces in image
    if len(faces)!=1:
        return [False, (0, 0, 0, 0)]
        faceExist=True;



    return [True,faces[0]]


def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c, r, w, h = window
    roi = frame[r:r + h, c:c + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist


kalman_first=True
posterior=0
# kalman filter
def skeleton_tracker3(frame, hand_points):
    # Open output file


    frameCounter = 0

    # detect face in first frame
    c, r, w, h = hand_points
    pt = (0, c + w / 2, r + h / 2)
    # Write track point for first frame

    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c, r, w, h)
    state = np.array([c + w / 2, r + h / 2, 0, 0], dtype='float64')  # initial position
    kalman = cv2.KalmanFilter(4, 2, 0)
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                        [0., 1., 0., .1],
                                        [0., 0., 1., 0.],
                                        [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)
    kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state
    measurement = np.array([c + w / 2, r + h / 2], dtype='float64')
    global kalman_first
    global posterior
    if kalman_first==True:
        kalman_first=False
        return [-1,-1]


    prediction = kalman.predict()  # prediction
    x, y, w, h = hand_points  # checking measurement
    measurement = np.array([x + w / 2, y + h / 2], dtype='float64')

    if not (x == 0 and y == 0 and w == 0 and h == 0):
        posterior = kalman.correct(measurement)
    if x == 0 and y == 0 and w == 0 and h == 0:
        x, y, w, h = prediction
    else:
        x, y, w, h = posterior
    pt = (frameCounter, x + w / 2, y + h / 2)
    #print(" points corrected ",pt)
    return [int(pt[1]),int(pt[2])]
    #print(" original points ",int((hand_points[0]+hand_points[2])/2), int((hand_points[1]+hand_points[3])/2))
    cv2.circle(frame,(int((x+w)/2),int((y+h)/2)),10,[255,125,125])
    #img2 = cv2.rectangle(frame, (int(x), int(y)), (int(x + 3), int(y + 3)), 255, 2)
    #img2 = cv2.rectangle(frame, (int(x)-100, int(y)-100), (int(x) + 100, int(y) + 100),  [0,0,0], 2)
    #img2[int(y)-100:int(y) + 100, int(x)-100:int(x) + 100] = [0, 0, 0];


### rectangle points
x1=0
y1=0
x2=0
y2=0
edged_fgbmask=0
tempx1_new, tempy1_new = 0, 0
draw=False
def mouseHandler(event,x,y,flags,param):
    global x1,x2,y1,y2,draw
    if event == cv2.EVENT_LBUTTONDOWN:
        x1=x
        y1=y
        draw=False
    elif event == cv2.EVENT_LBUTTONUP:
        x2=x
        y2=y
        draw=True
        area=np.abs(x2-x1)*np.abs(y2-y1)
        sum=np.sum(edged_fgbmask[y1:y2, x1:x2])
        edge_per_area=np.double(np.double(sum)/np.double(area))
        print(" sum of edge pixels per area = ")
        print(edge_per_area)
        print("box size : ",x2-x1,"    |||   ", y2-y1)
to_detect_hands=False

#TODO:THIS FUNCTION
def mouseHandler(event,x,y,flags,param):
    global x1,x2,y1,y2,draw
    if event == cv2.EVENT_LBUTTONDOWN:
        x1=x
        y1=y
        draw=False
    elif event == cv2.EVENT_LBUTTONUP:
        x2=x
        y2=y
        draw=True
        area=np.abs(x2-x1)*np.abs(y2-y1)
        sum=np.sum(edged_fgbmask[y1:y2, x1:x2])
        edge_per_area=np.double(np.double(sum)/np.double(area))
        print(" sum of edge pixels per area = ")
        print(edge_per_area)
        print("box size : ",x2-x1,"    |||   ", y2-y1)




cap = cv2.VideoCapture(0)

cap.set(5,75)
cap.set(3,640)
cap.set(4,480)

fgbg = cv2.createBackgroundSubtractorMOG2(history=frameHistory,detectShadows=False) # background subtractor object to subtract current frame from average of history frames
cv2.namedWindow('edged_fgbmask')
#cv2.setMouseCallback('edged_fgbmask', mouseHandler)
x=0
y=0
w=0
h=0

skin_ycrcb_mint = np.array((0, 133, 77))  # threshold for skin color max
skin_ycrcb_maxt = np.array((255, 173, 127))  # threshold for skin color

#matrice for erosion and dilation filled with ones
kernel_erosion = np.ones((5, 5), np.uint8)
kernel_dilation = np.ones((3, 3), np.uint8)

cv2.namedWindow("original Frame");
cv2.namedWindow("edged_fgbmask");
cv2.namedWindow("fgmask x normalized skin");
cv2.namedWindow("fgmask");
cv2.namedWindow("fgmask colored");
cv2.namedWindow("skin_ycrcb");

cv2.moveWindow('original Frame',0,0)
cv2.moveWindow('edged_fgbmask',720,0)
cv2.moveWindow('fgmask x normalized skin',1400,0)
cv2.moveWindow('fgmask',0,700)
cv2.moveWindow('fgmask colored',720,700)
cv2.moveWindow('skin_ycrcb',1420,700)
frame_counter_temp=0
count=0;
while(1):
    ret, frame = cap.read()
    [frame_max_y,frame_max_x,dummy]=frame.shape
    frame = cv2.flip(frame, 1)#flip image to adjust it
    ######################
    ########## used to call face detection every x seconds only
    if faceWasDetectedBefore==False:
        faceExist,face=detectSingleFace(frame)

    elif (time.time()-start_time>call_time):
        faceExist, face = detectSingleFace(frame)
        start_time=time.time()
    ############################################################
    ############################################################

    frame_counter_temp+=1
    print(frame_counter_temp)

    if faceExist:
        [x, y, w, h] = face
        faceExisted_once=True; # to know that face existed once in the image then  we should track it
        #print(" ( ",x," , ",y," , ", w , " , ",h," ) ")
        if faceWasDetectedBefore==False:
            faceWasDetectedBefore=True;

    if faceWasDetectedBefore:

        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #blacking the face area
        frame[y:y + w, x:x + h] = [0, 0, 0];


    test_time_1=datetime.now()

    #currently not used
    non_blurred_frame=frame

    #frame difference is store in fgmask
    fgmask = fgbg.apply(frame)  #apply background subtractor to get current frame - average(history frames )

    # convert RGB to YCR_CB to threshold skin color
    im_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

    # normalizing  fgmask ( frame-avg(history frames)
    fgmask_normalized=np.uint8((fgmask)/255)

    # thresholding frame to get skin color only
    skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)

    # TODO:fake numbers for matrices and iterations to get best results
    skin_ycrcb = cv2.erode(skin_ycrcb, kernel_dilation, iterations=1)
    skin_ycrcb = cv2.dilate(skin_ycrcb, kernel_dilation, iterations=1)


    normalized_skin=np.uint8((skin_ycrcb)/255)#normalizing  skin_ycrbcb
    # convert greyscale normalized skin_ycrbcb to rgb  to be multiplied by frame (rgb)
    normalized_skin_3D=cv2.cvtColor(normalized_skin, cv2.COLOR_GRAY2RGB);

    # moving skin parts.
    fgmask_on_skin_normalized=np.uint8(fgmask_normalized*normalized_skin)

    # TODO:fake numbers for matrices and iterations to get best results
    fgmask_on_skin_normalized = cv2.erode(fgmask_on_skin_normalized, kernel_dilation, iterations=1)
    fgmask_on_skin_normalized = cv2.dilate(fgmask_on_skin_normalized, kernel_dilation, iterations=1)

    #fgmask_on_skin = cv2.erode(fgmask_on_skin, kernel_erosion, iterations=1)

    #fgmask_on_skin_normalized=np.uint8(np.double(fgmask_on_skin)/255.0) #normalized skin color of moving parts * original frame
    fgmask_on_skin_3D=np.uint8(cv2.cvtColor(fgmask_on_skin_normalized, cv2.COLOR_GRAY2RGB));


    #now we have moving skin parts in the original image this is needed for testing with our eyes only
    rgb_masked_image=fgmask_on_skin_3D*frame

    edged_fgbmask = cv2.Canny(rgb_masked_image, 35, 150) #applying canny edge detection


    kernel = np.ones((101, 101), np.double) / (101*101) #filter to calculate edges within a box

    edges_per_area_image = cv2.filter2D(np.double(edged_fgbmask), -1, kernel)# edges per area ( fixed area)
    minVal, maxVal, minLoc, maxLoc=cv2.minMaxLoc(edges_per_area_image) #getting maximum point of maximum edges per area value

    edges_per_area_image[maxLoc[1]-100:maxLoc[1]+100,maxLoc[0]-100:maxLoc[0]+100]=0
    minVal2, maxVal2, minLoc2, maxLoc2=cv2.minMaxLoc(edges_per_area_image) #getting maximum point of maximum edges per area value

    test_time_1=datetime.now()-test_time_1
    test_time_1=test_time_1.total_seconds()
   # print(" biggest opeartion delay  ",test_time_1)
    #print("min val : "  ,minVal," max val ",maxVal," min Loc ",minLoc," max Loc ",maxLoc)

    tempx1,tempy1=maxLoc
    tempx2,tempy2=maxLoc2

    # yield the current window

    if faceExisted_once: # if face exist then draw box over point of maximum edges per area
        #print("fab..")
       if maxVal2>10:
           [left1, top1, right1, bottom1] = updateHandRect(edged_fgbmask, maxLoc2, frame)  # get top and left points of hands not working well
           tempx2_new, tempy2_new = tempx1, tempy1
           #  cv2.rectangle(frame, (tempx2 - 100, tempy2 - 100), (tempx2 + 100, tempy2 + 100), [255, 255, 255], 3)


       if maxVal>10 and forget_frames <=0: # forget_frames used to wait until stablization
            to_detect_hands = True;
            [left,top,right,bottom]=updateHandRect(edged_fgbmask,maxLoc,frame)#get top and left points of hands not working well
            tempx1_new, tempy1_new = tempx1,tempy1

                #edged_fgbmask[tempy1_new-100:tempy1_new+100,tempx1_new-100:tempx1_new+100]
                #edged_fgbmask = cv2.dilate(edged_fgbmask, kernel_dilation, iterations=1)

            #print("pixels summation : ", pixels_summation/10000," max value ",maxVal)
       else:
            if(forget_frames>-10):
                forget_frames-=1
            #print("max value ",maxVal)
    if to_detect_hands:
        #update points traced for action detection
        actionDetector.updateTracingPoints(tempx1_new,tempy1_new)
        #[a,b,c,d]=hand_region_Window
        ret=actionDetector.getTracingPoints()
        #print(ret)
        actionDetector.getAction()
        #print(tracing_points)
        #if(ret==True):
            #action_state,action=getAction(tempx1_new,tempy1_new)
        #getWristPoints(skin_ycrcb,tempy1_new - b,tempy1_new +d,tempx1_new - a,tempx1_new + c)
        #sprint(" top left : (x1,y1)(x2,y2)  (", a, " , ", b, " ) (",c, " , ",d, " ) ")
        cv2.rectangle(frame, (left, top), (right, bottom), [0,255 , 255], 3)
        cv2.rectangle(frame, (tempx1_new-100, tempy1_new-100), (tempx1_new+100, tempy1_new+100), [0,255 , 0], 3)

        # drawing 2nd rectangles over the 2nd hand
        #cv2.rectangle(frame, (tempx2_new-100, tempy2_new-100), (tempx2_new+100, tempy2_new+100), [0,255 , 0], 3)
        #cv2.rectangle(frame, (left1, top1), (right1, bottom1), [0,255 , 255], 3)
        ##########################

        #cv2.rectangle(skin_ycrcb, (tempx1_new-a, tempy1_new-b), (tempx1_new+c, tempy1_new+d), 155, 3)
   # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   


#print("contours size ", len(contours)," max x ",max_x," max_y ",max_y)
    frames_count+=1;

    cv2.imshow('fgmask x normalized skin', fgmask_on_skin_normalized*255)

    cv2.imshow('fgmask',fgmask)
    cv2.imshow('fgmask colored',rgb_masked_image)


    #cv2.imshow('edges_per_area_image', edges_per_area_image)
   # print(x1)
    if draw:
        cv2.rectangle(edged_fgbmask, (tempx1_new-50, tempy1_new-50), (tempx1_new+100, tempy1_new+100), 255, 3)
    cv2.imshow("skin_ycrcb",skin_ycrcb)
    cv2.imshow('edged_fgbmask',edged_fgbmask)
    cv2.imshow('fgmask',skin_ycrcb)

    cv2.imshow('original Frame', frame)


    k = cv2.waitKey(30) & 0xff
    if k==114:
        cv2.imwrite("hand"+str(count)+".jpg",frame[tempy1_new-100:tempy1_new+100,tempx1_new-100:tempx1_new+100])
        count+=1;
        #reset counter if 'r' is pressed //debugging mode
        #frame_counter_temp=0
    if k == 27:
        break

end_time=datetime.now()
diff=(end_time-begin).total_seconds()
print("diff ",diff)
print(" end time  : " ,end_time)
print(" start time  : " ,start_time)

print("FPS : ", frames_count/(diff))
cap.release()
cv2.destroyAllWindows()