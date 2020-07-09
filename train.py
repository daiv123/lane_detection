import pickle
import sys
import cnn
import numpy as np
import cv2
import torch
import os
from torchvision import transforms as trans

data_set = pickle.load(open(sys.argv[1], "rb"))

print("loaded data from file")

train_set_torch, train_labels_torch, dev_set_torch = data_set

print(train_set_torch.size())
print(dev_set_torch.size())

losses,xhats, net = cnn.fit(train_set_torch, train_labels_torch, dev_set_torch, n_iter = 1000, batch_size=100)

# xhats = xhats.permute(0,2,3,1)
# xhats = xhats.detach().numpy()
# d