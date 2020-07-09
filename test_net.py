import pickle
import sys
import cnn
import numpy as np
import cv2
import torch
import os

model = torch.load("conv_net2")
model.eval()
images = [cv2.imread(file.path) for file in os.scandir(sys.argv[1]) if (file.is_file() and file.name[0] != '.')]

shrunk_images = []
for image in images :
    shrunk_images.append(cv2.resize(image, (180, 40)))
images = shrunk_images
image_tensor = torch.tensor(images, dtype= torch.float32)/255
image_tensor = image_tensor.permute(0,3,1,2)
label_tensor = model.forward(image_tensor)*255
label_tensor = label_tensor.permute(0,2,3,1)
labels = label_tensor.detach().numpy()

for image, label in zip(images, labels):
    cv2.imshow("image", image)
    cv2.imshow("label", label)
    cv2.moveWindow("label", 100, 10)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
