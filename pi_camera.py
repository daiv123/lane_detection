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

sleep(2)
key = input("Enter number of frames: ")
while not key.isnumeric():
    key = input("Enter number of frames: ")

num_frames = int(key)
for filename in camera.capture_continuous('img{counter:03d}.jpg'):
    print('Captured %s' % filename)
    print(int(key)-num_frames + 1)

    num_frames-=1
    
    if num_frames == 0 :
        key = input("Enter number of frames: ") 
        while not key.isnumeric():
            key = input("Enter number of frames: ")

    sleep(1)
       
