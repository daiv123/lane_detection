import cv2
import numpy as np
import sys
import os
import lane_detection_utils as ldu
import pickle

def rotate(image, angle) :
    img = image[:]
    num_rows, num_cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), angle, 1)
    img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
    final = img_rotation[12 : num_rows-12, 10:num_cols-10]
    return final

def process(image) :

    img_processed = ldu.pipeline(image)
    label = ldu.pipeline(image, overlay=False)    
    # img_processed = image[len(image)-150:, :]
    return (img_processed, label)

def main() :


    data_set = []

    if len(sys.argv) != 5:
        print("usage: process_images.py data_set pruned_data_set outdir data_set_name")
        exit()
    do_all = False
    images = [(cv2.imread(file.path), file.name) for file in os.scandir(sys.argv[1]) if (file.is_file() and file.name[0] != '.')]
    pruned_image_names = [file.name for file in os.scandir(sys.argv[2]) if (file.is_file() and file.name[0] != '.')]
    
    os.chdir(sys.argv[3])
    for img, img_name in images :
        if img_name in pruned_image_names:
            data_set.append((img,process(img)[1]))
            cv2.imwrite(img_name, process(img)[1])
        # else :
        #     cv2.imwrite(img_name, img)

        
    print("num_images_processed:", len(data_set))
    pickle.dump(data_set, open(sys.argv[4], "wb"))

if __name__ == "__main__":
    main()