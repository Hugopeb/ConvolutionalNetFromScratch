import torch
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


########### MODEL #############

import torch.nn.functional as F

class kernel():
    def __init__(self, input_shape, filter_size):
        self.filter_size = filter_size # (3)
        self.input_shape = input_shape # shape of the image (1, 28, 28)
        self.input_size = self.input_shape[1] # dimension of the image (28)
        self.num_filters = self.input_shape[0] # num of input channels (num of filters)
        self.output_size = self.input_size - self.filter_size + 1 # (26)

        self.filters = []
        for i in range(self.num_filters):
            self.filters.append(torch.randn(self.filter_size, self.filter_size))

        self.filters = torch.stack(self.filters, dim = 0)
        self.bias = torch.randn(self.output_size, self.output_size)
        

class conv_layer():
    def __init__(self, input_shape, num_kernels, filter_size):
        self.num_kernels = num_kernels
        self.filter_size = filter_size
        self.input_shape = input_shape
        self.input_size = input_shape[1]
        self.num_filters = input_shape[0]
        self.output_size = self.input_size - self.filter_size + 1
        
        self.kernels = []
        for i in range(self.num_kernels):
            self.kernels.append(kernel(self.input_shape, self.filter_size))


    def forward(self, X_input):
        self.input = X_input
        self.output = torch.zeros(self.num_kernels, self.output_size, self.output_size)

        for i, kernel in enumerate(self.kernels):
            for j in range(self.num_filters):
                channel_input = self.input[j]
                channel_filter = kernel.filters[j]

                self.output[i] += F.conv2d(
                    channel_input.float().unsqueeze(0).unsqueeze(0),
                    channel_filter.unsqueeze(0).unsqueeze(0)
                ).squeeze()
                
            self.output[i] += kernel.bias
        return self.output

    
    def backwards(self, grad_output, lr = 0.01):
        grad_filter = torch.zeros(self.num_kernels, self.num_filters, 
                                  self.filter_size, self.filter_size)

        grad_input = torch.zeros(self.num_filters, self.input_size, self.input_size)
        
        for i, kernel in enumerate(self.kernels):
            kernel_grad_output = grad_output[i]
            
            for j in range(kernel.num_filters):
                channel_input = self.input[j]
                channel_filter = kernel.filters[j]
                
                grad_filter[i, j] += F.conv2d(
                    channel_input.float().unsqueeze(0).unsqueeze(0),
                    kernel_grad_output.unsqueeze(0).unsqueeze(0)
                ).squeeze()

                channel_filter_flipped = torch.flip(channel_filter, dims = [0,1])

                grad_input[j] += F.conv2d(
                    kernel_grad_output.unsqueeze(0).unsqueeze(0),
                    channel_filter_flipped.unsqueeze(0).unsqueeze(0),
                    padding = (self.filter_size-1, self.filter_size-1)
                ).squeeze()


        for i, kernel in enumerate(self.kernels):
            kernel.filters -= lr * grad_filter[i]

            grad_bias = grad_output[i]
            kernel.bias -= lr * grad_bias 

        return grad_input
    

########## AUXILIARY FUNCTIONS ###########


class reshape_layer():
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, X_input):
        return torch.reshape(X_input, self.output_shape)

    def backwards(self, grad_output):
        return torch.reshape(grad_output, self.input_shape)
    

def get_batches(data, labels, batch_size = 32, shuffle = True):
    number_of_samples = labels.shape[0]
    indices = torch.arange(number_of_samples)
    x = data[indices]
    y = labels[indices]
    
    if shuffle:
        indices = torch.randperm(number_of_samples)
        x = data[indices]
        y = labels[indices]
        
    for idx in range(0, number_of_samples, batch_size):
        x_batch = x[idx:idx+batch_size]
        y_batch = y[idx:idx+batch_size]
        yield x_batch, y_batch

        
########## ACTIVATION FUNCTIONS ##########

class softmax():
    def forward(self, X_input):
        return torch.softmax(X_input, dim = 1)

    def backwards(self, grad_output):
        return grad_output # we will calculate it inside the training loop

class tanh():
    def forward(self, X_input):
        self.input = X_input
        self.output = torch.tanh(self.input)
        return self.output

    def backwards(self, grad_output):
        grad_input = grad_output * (1- self.output**2)
        return grad_input
        
class ReLU():
    def forward(self, X_input):
        self.mask = X_input > 0
        return self.mask * X_input

    def backwards(self, grad_output):
        return self.mask * grad_output
    
    
########## FULLY CONNECTED NETWORK ###########


class dense():
    def __init__(self, output_size, input_size):
        self.ouput_size = output_size
        self.input_size = input_size
        self.W = torch.randn(output_size, input_size) * 0.01
        self.bias = torch.randn(output_size, ) * 0.01

    def forward(self, X_input):
        self.input = X_input
        
        Z = self.input @ self.W.T + self.bias
        return Z

    def backwards(self, grad_output, lr = 0.01):
        self.dW = grad_output.T @ self.input 
        self.db = grad_output.sum(dim = 0)
        self.dX = grad_output @ self.W

        self.W -= self.dW * lr
        self.bias -= self.db * lr
        
        return self.dX
    

########## CONVOLUTIONAL NET ###########

C1 = conv_layer((1, 28, 28), num_kernels = 8, filter_size = 3)
C2 = conv_layer((8, 26, 26), num_kernels = 16, filter_size = 3)

first_layer = dense(256, 9216)
second_layer = dense(10, 256)

ReSHAPE = reshape_layer((16, 24, 24), (1, 9216))

network = [
    C1,
    tanh(),
    C2,
    tanh(),
    ReSHAPE,
    first_layer,
    tanh(),
    second_layer,
    softmax()
]


########### LOAD THE DATA ###########

train_data = datasets.MNIST(
    root = "./train_MNIST",
    download = True,
    train = True,
    transform = ToTensor()
)

test_data = datasets.MNIST(
    root = "./test_MNIST",
    download = True,
    train = False,
    transform = ToTensor()
)

train_images = train_data.data / 255
train_labels = train_data.targets 

test_images = test_data.data / 255
test_labels = test_data.targets

########### TRAINING LOOP ###########

num_epochs = 1

for epoch in range(num_epochs):
    epoch_cross_entropy = 0
    for x_batch, y_batch in get_batches(train_images[0:128], train_labels[0:128], batch_size = 1):
        output = torch.reshape(x_batch, (1,28,28))
    
        for layer in network:
            output = layer.forward(output)
            
        epoch_cross_entropy += -torch.log(output[0, y_batch] + 1e-9)
                
        grad = output.clone()
        grad[0, y_batch] -= 1
    
        for layer in reversed(network):
            grad = layer.backwards(grad)
            
    print("avg_cross_entropy_epoch: %.10f"%(epoch_cross_entropy / 128))
    