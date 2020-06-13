from time import sleep
from picamera import PiCamera
import sys
import os

if len(sys.argv) != 2:
    print("usage: pi_camera <output_directory>")
    sys.exit(2)

outDir = sys.argv[1]
os.chdir(outDir)

camera = PiCamera()
camera.start_preview()
sleep(2)
key = input("Enter number of frames: ")
counter = 1

while key != 'q':

    while not key.isnumeric()
        key = input("Enter number of frames: ")
    
    num_frames = int(key)
    for filename in camera.capture_continuous('img{counter:03d}.jpg'):
        print('Captured %s' % filename)
        print(int(key)-num_frames + 1)

        num_frames-=1
        counter += 1

        if num_frames == 0 :
            break

        sleep(1)
    