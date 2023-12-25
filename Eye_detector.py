
##Importing libraries

from imutils import face_utils as Futility

import cv2 as ComputerVision

import imutils as ImageUtilities
import dlib as digitalLibrary
from scipy.spatial import distance as length
import cv2

ColortoBWconverter = ComputerVision.COLOR_BGR2GRAY
THICKNESS = 2
COLOR = (0,0,255)

sensitivity = 0 #Threshold to consider eye as closed
ww = 450
det = 0

countFrm = 15  # increase eye close time

check = digitalLibrary.get_frontal_face_detector()

forecast = digitalLibrary.shape_predictor("faceShape.dat")

(LeftEyeStart, LeftEyeEnd) = Futility.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(RightEyeStart, RightEyeEnd) = Futility.FACIAL_LANDMARKS_68_IDXS["right_eye"]

Internalcamera = ComputerVision.VideoCapture(0)
# Initialize the Internal Camera

def getRatio(gridEye):
    global length
    
    ratio1 = length.euclidean(gridEye[1], gridEye[5])
    ratio2 = length.euclidean(gridEye[2], gridEye[4])
    ratio3 = length.euclidean(gridEye[0], gridEye[3])
    ratio = (ratio1 + ratio2) / (2.0 * ratio3)
    return ratio

def drawEyeCurve(eyeCurve):
    global ComputerVision
    global RawImage
    
    ComputerVision.drawContours(RawImage, [eyeCurve], -1, COLOR, thickness=THICKNESS)
    

def getLeftEyeCurveDetails(Lrat):
    global ComputerVision
    
    x = ComputerVision.convexHull(Lrat)
    return x

def getRightEyeCurveDetails(Rrat):
    global ComputerVision
    x = ComputerVision.convexHull(Rrat)
    return x



while True:
    done,RawImage =Internalcamera.read()
    # Read the Image from Internal Camera
    if (done == 0):
        print("Unable to get Image data")
        continue

    RawImage = ImageUtilities.resize(RawImage, width= ww)
    #Resizing the Image in suitable size
    BWimg = ComputerVision.cvtColor(RawImage, ComputerVision.COLOR_BGR2GRAY)
    # Making the Color Black&White for size Reduction and faster Processing
    BWimg = ImageUtilities.resize(BWimg, width= ww)
    #Resizing the B&W Image in suitable size
    multiFaceData = check(BWimg, 0)
    # Get data of all the multiple faces found
    det = 0
    #print(det)

    for singleFaceData in multiFaceData:
        # getting the detected face locations
        x1,y1,x2,y2 = singleFaceData.left(),singleFaceData.top(),singleFaceData.right(),singleFaceData.bottom()

        # Take all the faces one by one from image
        det = 0
        geometry = forecast(BWimg, singleFaceData)
        # Try Predicting Location of the face
        geometry = Futility.shape_to_np(geometry)
        # Change the face data into Numpy array for numerical processing
        
        LEye = geometry[LeftEyeStart:LeftEyeEnd]
        #Get the Geometrical details of LeftEye
        if(LEye.any()):
            det = 1
        Lratio = getRatio(LEye)
        #Get the Aspect ratio of the left eye
        if(Lratio < 0):
            continue
       
        REye = geometry[RightEyeStart:RightEyeEnd]
        #Get the Geometrical details of RightEye
        if(REye.any()):
            det = 1
        
        Rratio = getRatio(REye)
        #Get the Aspect ratio of the right eye
        if(Rratio < 0):
            continue
        
        finalRatio = (Lratio + Rratio )/ 2.0
        # Get the arithmetic mean of both ratios
        
        LEyeCurve = getLeftEyeCurveDetails(LEye)
        drawEyeCurve(LEyeCurve)
        #Get the curve definition of Left Eye and Draw it
        
        REyeCurve = getRightEyeCurveDetails(REye)
        drawEyeCurve(REyeCurve)

        # drawing rectangle around the detected face
        ComputerVision.rectangle(RawImage, (x1,y1), (x2,y2), (255,0,255), 3)

        #Get the curve definition of Right Eye and draw it
    ComputerVision.imshow("Eye Detection", RawImage)

    #Show thw Update RawImage on Screen
    got = ComputerVision.waitKey(125) & 0xFF
    if got == 27:
        break
cv2.destroyAllWindows()
Internalcamera.release()