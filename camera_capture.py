
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
# initialize the camera and grab a reference to the raw camera capture

camera = PiCamera()
rawCapture = PiRGBArray(camera)
# allow the camera to warmup
i = 0
name = "frame{}.png"
while True :
    if cv2.waitKey(1) and 0xFF == ord('m')
        camera.capture(rawCapture, format="bgr")
        image = rawCapture.array
        cv2.imwrite(name.format(i), image)
    
