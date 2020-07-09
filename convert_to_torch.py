import pickle
import sys
import cnn
import numpy as np
import cv2
import torch

data_set = pickle.load(open(sys.argv[1], "rb"))

print("loaded data from file")

np.random.shuffle(data_set)
shrunk_data_set = []
for (image, label) in data_set :
    shrunk_data_set.append((cv2.resize(image, (180, 40)), cv2.resize(label, (180, 40))))
    shrunk_data_set.append((cv2.flip(cv2.resize(image, (180, 40)), 1), cv2.flip(cv2.resize(label, (180, 40)), 1)))

data_set = shrunk_data_set
np.random.shuffle(data_set)
print("shrunk image sizes by 4")

dev_set, trash = zip(*data_set[:25])
train_set, train_labels = zip(*data_set[25:])

train_set = torch.tensor(train_set, dtype= torch.float32)/255
train_labels = torch.tensor(train_labels, dtype= torch.float32)/255
dev_set = torch.tensor(dev_set, dtype= torch.float32)/255

print("converted to tensor")

train_set_torch = train_set.permute(0,3,1,2)
train_labels_torch = train_labels[:,None,:,:]
# train_labels_torch = train_labels.permute(0,3,1,2)
dev_set_torch = dev_set.permute(0,3,1,2)

print("tensor rearanged")



print(train_labels_torch.size())
print(dev_set_torch.size())

pickle.dump((train_set_torch, train_labels_torch, dev_set_torch), open("data_torch.p", "wb"))