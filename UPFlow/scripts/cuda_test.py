import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import tensorflow as tf # Successfully opened dynamic library libcudart.so.10.1

# device = torch.device("cuda")
        
# im1 = [3, 5]
# im2 = [1, 2]

# im1 = torch.Tensor(im1).to(device)
# im2 = torch.Tensor(im2).to(device)

# print("done")

# Let’s define this network:

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CorrelationFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input1, input2, 
            pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
        print("in CorrelationFunction forward")
        ctx.save_for_backward(input1, input2)

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()

            # correlation_cuda.forward(input1, input2, rbot1, rbot2, output, 
            #     pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()

            grad_input1 = input1.new()
            grad_input2 = input2.new()

            # correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
            #     pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply)

        return grad_input1, grad_input2

class Correlation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input1, input2, 
            pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1): 
            # self, self for the object; when you call, it is without - synthactic sugar!
        print("in Correlation forward")

        result = CorrelationFunction(input1, input2, pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply)
        # result = CorrelationFunction(pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply)(input1, input2)

        return result

net = Net()
# print(net)

# params = list(net.parameters())
# print(len(params))
# print(params[0].size())  # conv1's .weight

input = torch.randn(1, 1, 32, 32)

#
leakyRELU = nn.LeakyReLU(0.1, inplace=True)
out_corr_1 = Correlation.apply(input, input, 4, 1, 4, 1, 1, 1)
out_corr_relu_1 = leakyRELU(out_corr_1)

out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1, 10))

output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update