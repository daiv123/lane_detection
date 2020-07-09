import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NeuralNet(torch.nn.Module):
    def __init__(self, lrate, loss_fn, in_size,out_size):
        """
        Initialize the layers of your neural network
        @param lrate: The learning rate for the model.
        @param loss_fn: The loss function
        @param in_size: Dimension of input
        @param out_size: Dimension of output
        """
        super(NeuralNet, self).__init__()
        """
        1) DO NOT change the name of self.encoder & self.decoder
        2) Both of them need to be subclass of torch.nn.Module and callable, like
           output = self.encoder(input)
        3) Use 2d conv for extra credit part.
           self.encoder should be able to take tensor of shape [batch_size, 1, 28, 28] as input.
           self.decoder output tensor should have shape [batch_size, 1, 28, 28].
        """
        pool_size = (2,2)
        kernal_size = (3,3)
        dropout_rate = 0.2
        self.pipeline = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 8, kernal_size),
            nn.ReLU(),
            nn.Conv2d(8,16,kernal_size),
            nn.ReLU(),

            nn.MaxPool2d(pool_size),

            nn.Conv2d(16, 16, kernal_size),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(16, 32, kernal_size),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(32, 32, kernal_size),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),

            nn.MaxPool2d(pool_size),

            nn.Upsample(scale_factor= 2),

            nn.ConvTranspose2d(32, 32, kernal_size),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.ConvTranspose2d(32, 16, kernal_size),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.ConvTranspose2d(16, 16, kernal_size),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),

            nn.Upsample(scale_factor = 2),

            nn.ConvTranspose2d(16, 8, kernal_size),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernal_size)

        )
        


        self.in_size = in_size
        self.out_size = out_size
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.parameters(),weight_decay=1e-5)

    def get_parameters(self):
        """ Get the parameters of your network
        @return params: a list of tensors containing all parameters of the network
        """
        return self.parameters()

    def forward(self, x):
        """ A forward pass of your autoencoder
        @param x: an (N, in_size) torch tensor
        @return xhat: an (N, out_size) torch tensor of output from the network.
                      Note that self.decoder output needs to be reshaped from
                      [N, 1, 28, 28] to [N, out_size] beforn return.
        """
        out = self.pipeline(x)
        return out

    def step(self, x, labels):
        # x [100, 784]
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        xhat = self(x)

        L = self.loss_fn(xhat, labels)

        self.optimizer.zero_grad()
        L.backward()
        self.optimizer.step()
        return L.item()

def fit(train_set,train_labels, dev_set,n_iter,batch_size=100):
    """ Fit a neural net.  Use the full batch size.
    @param train_set: an (N, 784) torch tensor
    @param dev_set: an (M, 784) torch tensor
    @param n_iter: int, the number of batches to go through during training (not epoches)
                   when n_iter is small, only part of train_set will be used, which is OK,
                   meant to reduce runtime on autograder.
    @param batch_size: The size of each batch to train on.
    # return all of these:
    @return losses: list of total loss (as type float) after each iteration. Ensure len(losses) == n_iter
    @return xhats: an (M, out_size) NumPy array of reconstructed data.
    @return net: A NeuralNet object
    # NOTE: This must work for arbitrary M and N
    """

    print("initiating")
    net = NeuralNet(0.1, nn.MSELoss(), 28*28, 28*28)
    losses = []
    N = len(train_set)
    print(N)
    
    for k in range(128) :
        i = 0
        while i * batch_size < N and i < n_iter:
            if (i+1)*batch_size < N :
                batch = train_set[i*batch_size : (i+1)*batch_size]
                labels = train_labels[i*batch_size : (i+1)*batch_size]
            else :
                batch = train_set[i*batch_size : N]
                labels = train_labels[i*batch_size : N]
            
            loss = net.step(batch,labels)
            losses.append(loss)
            i+=1
    print("round", k , "done")      
    
    
    #use
    xhats = net.forward(dev_set)
    torch.save(net, "conv_net3")
    print("saved as conv_net3")
 
    return losses,xhats, net
