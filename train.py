import pickle
import sys
import cnn
import numpy as np
import cv2
import torch
from torchvision import transforms as trans

data_set = pickle.load(open(sys.argv[1], "rb"))
train_set, train_labels = zip(*data_set[:100])
dev_set, trash = zip(*data_set[100:])

train_set = torch.tensor(train_set, dtype= torch.float32)/255
train_labels = torch.tensor(train_labels, dtype= torch.float32)/255
dev_set = torch.tensor(dev_set, dtype= torch.float32)/255

train_set_torch = train_set.permute(0,3,1,2)
train_labels_torch = train_labels[:,None,:,:]
# train_labels_torch = train_labels.permute(0,3,1,2)
dev_set_torch = dev_set.permute(0,3,1,2)

print(train_labels_torch.size())
print(dev_set_torch.size())

losses,xhats, net = cnn.fit(train_set_torch, train_labels_torch, dev_set_torch, n_iter = 100, batch_size=20)

xhats = xhats.permute(0,2,3,1)
xhats = xhats.detach().numpy()

cv2.imshow("a", xhats[0]*(255/np.max(xhats[0])))

cv2.imshow("b", xhats[1]*(255/np.max(xhats[1])))

cv2.imshow("c", xhats[2]*(255/np.max(xhats[2])))
cv2.imshow("d", xhats[3]*(255/np.max(xhats[3])))
cv2.imshow("e", xhats[4]*(255/np.max(xhats[4])))
cv2.waitKey(0)