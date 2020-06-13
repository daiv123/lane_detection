
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
# initialize the camera and grab a reference to the raw camera capture

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

time.sleep(0.1)

# allow the camera to warmup
i = 0
name = "frame{}.png"
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    cv2.imshow("frame", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('m'):
            cv2.imwrite(name.format(i), image)
            i += 1
    elif key == ord('q'):
        break
    rawCapture.truncate(0)

