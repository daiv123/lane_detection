import cv2
import numpy as np
import sys
import os
import lane_detection_utils as ldu

def rotate(image, angle) :
    img = image[:]
    num_rows, num_cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), angle, 1)
    img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
    final = img_rotation[12 : num_rows-12, 10:num_cols-10]
    return final

def process(image) :

    img_processed = ldu.pipeline(image)    
    # img_processed = image[len(image)-150:, :]
    return img_processed

def main() :

    if len(sys.argv) != 3:
        print("usage: process_images.py source_dir labled_dir")
        exit()
    images = [(cv2.imread(file.path), file.name) for file in os.scandir(sys.argv[1]) if (file.is_file() and file.name[0] != '.')]
    
    source_dict = {}
    
    for img, img_name in images :
        source_dict[img_name] = img

    images = [(cv2.imread(file.path), file.name) for file in os.scandir(sys.argv[2]) if (file.is_file() and file.name[0] != '.')]
    os.chdir(sys.argv[2])
    dataset = []
    for img, img_name in images :
        channel_r = img[:,:,2]
        label = cv2.inRange(channel_r, 254, 255)

        cv2.imshow("red", label)
        cv2.waitKey(0)



if __name__ == "__main__":
    main()