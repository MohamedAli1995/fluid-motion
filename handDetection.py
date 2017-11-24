import numpy as np
import cv2
def detect_one_face(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)

    if len(faces) == 0:
        return (0, 0, 0, 0)
    return faces[0]


def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c, r, w, h = window
    roi = frame[r:r + h, c:c + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist


# kalman filter
def skeleton_tracker3(v, file_name):
    # Open output file


    frameCounter = 0
    # read first frame
    ret, frame = v.read()
    frame=cv2.flip(frame,1)
    if ret == False:
        return

    # detect face in first frame
    c, r, w, h = detect_one_face(frame)
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

    while (1):
        ret, frame = v.read()  # read another frame
        frame = cv2.flip(frame, 1)
        if ret == False:
            break

        prediction = kalman.predict()  # prediction
        x, y, w, h = detect_one_face(frame)  # checking measurement
        measurement = np.array([x + w / 2, y + h / 2], dtype='float64')

        if not (x == 0 and y == 0 and w == 0 and h == 0):
            posterior = kalman.correct(measurement)
        if x == 0 and y == 0 and w == 0 and h == 0:
            x, y, w, h = prediction
        else:
            x, y, w, h = posterior
        pt = (frameCounter, x + w / 2, y + h / 2)

        img2 = cv2.rectangle(frame, (int(x), int(y)), (int(x + 3), int(y + 3)), 255, 2)
        img2 = cv2.rectangle(frame, (int(x)-100, int(y)-100), (int(x) + 100, int(y) + 100),  [0,0,0], 2)
        img2[int(y)-100:int(y) + 100, int(x)-100:int(x) + 100] = [0, 0, 0];
        cv2.imshow('img2', img2)
        k = cv2.waitKey(25) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k) + ".jpg", img2)

        frameCounter = frameCounter + 1



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



face_cascade = cv2.CascadeClassifier('front_face.xml')

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2(history=150,detectShadows=False)
cv2.namedWindow('edged_fgbmask')
cv2.setMouseCallback('edged_fgbmask', mouseHandler)
while(1):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # haar cascade face detection
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    faceExist=False;
    if len(faces)!=0:
        #print("zero faces ////")
        faceExist=True;

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0,255), 2)
        frame[y:y + w, x:x + h] = [0,0,0];

    non_blurred_frame=frame
    fgmask = fgbg.apply(frame)
    im_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    fgmask_normalized=np.uint8(np.double(fgmask)/255.0)
    skin_ycrcb_mint = np.array((0, 133, 77))
    skin_ycrcb_maxt = np.array((255, 173, 127))
    skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)
    normalized_skin=np.uint8(np.double(skin_ycrcb)/255.0)
    normalized_skin_3D=cv2.cvtColor(normalized_skin, cv2.COLOR_GRAY2RGB);
    fgmask_on_skin=np.uint8(fgmask*normalized_skin)
    fgmask_on_skin_normalized=np.uint8(np.double(fgmask_on_skin)/255.0)
    fgmask_on_skin_3D=np.uint8(np.double(cv2.cvtColor(fgmask_on_skin, cv2.COLOR_GRAY2RGB))/255.0);
    rgb_masked_image=fgmask_on_skin_3D*frame
    edged_fgbmask = cv2.Canny(normalized_skin_3D*frame, 35, 125)*fgmask_on_skin_normalized


    kernel = np.ones((100, 100), np.double) / (100*100)

    edges_per_area_image = cv2.filter2D(np.double(edged_fgbmask), -1, kernel)
    minVal, maxVal, minLoc, maxLoc=cv2.minMaxLoc(edges_per_area_image)
    print("min val : "  ,minVal," max val ",maxVal," min Loc ",minLoc," max Loc ",maxLoc)
    tempx1,tempy1=maxLoc

    # yield the current window

    if faceExist:
        #print("fab..")
        if maxVal>5.0:
            to_detect_hands = True;
            tempx1_new, tempy1_new=tempx1,tempy1
    if to_detect_hands:
        cv2.rectangle(frame, (tempx1_new - 60, tempy1_new - 60), (tempx1_new + 70, tempy1_new + 100), [255, 0, 0], 3)
    else:
        print("face error ")
   # cv2.imshow('fgmask x normalized skin', fgmask_on_skin)
    cv2.imshow('original Frame', frame)
    #cv2.imshow('fgmask',fgmask)
    #cv2.imshow('fgmask colored',rgb_masked_image)


   # cv2.imshow('edges_per_area_image', edges_per_area_image)
   # print(x1)
    if draw:
        cv2.rectangle(edged_fgbmask, (x1, y1), (x2, y2), 255, 3)


   # cv2.imshow('edged_fgbmask',edged_fgbmask)
    #cv2.imshow('fgmask',skin_ycrcb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()