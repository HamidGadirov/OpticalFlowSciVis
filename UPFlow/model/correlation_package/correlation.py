import torch
from torch.nn.modules.module import Module
from torch.autograd import Function
import correlation_cuda

class CorrelationFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2,
            pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
        # print("in CorrelationFunction forward")
        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply

        ctx.save_for_backward(input1, input2)

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()

            correlation_cuda.forward(input1, input2, rbot1, rbot2, output, 
                ctx.pad_size, ctx.kernel_size, ctx.max_displacement, ctx.stride1, ctx.stride2, ctx.corr_multiply)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()

            grad_input1 = input1.new()
            grad_input2 = input2.new()

            correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
                ctx.pad_size, ctx.kernel_size, ctx.max_displacement, ctx.stride1, ctx.stride2, ctx.corr_multiply)

        return grad_input1, grad_input2, None, None, None, None, None, None

# class Correlation(Function):

#     @staticmethod
#     def forward(ctx, input1, input2, 
#             pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1): 
#             # self, self for the object; when you call, it is without - synthactic sugar!
#         # print("in Correlation forward")

#         result = CorrelationFunction.apply(input1, input2, pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply)
#         # result = CorrelationFunction(pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply)(input1, input2)

#         return result

#     @staticmethod
#     def backward(ctx, grad_output):
#         print("in Correlation backward")

"""
# old version:
# class CorrelationFunction(Function):
#     def __init__(self, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
#         super(CorrelationFunction, self).__init__()
#         print("in CorrelationFunction __init__")
#         self.pad_size = pad_size
#         self.kernel_size = kernel_size
#         self.max_displacement = max_displacement
#         self.stride1 = stride1
#         self.stride2 = stride2
#         self.corr_multiply = corr_multiply
#         # self.out_channel = ((max_displacement/stride2)*2 + 1) * ((max_displacement/stride2)*2 + 1)

#     # @staticmethod
#     def forward(self, input1, input2): # self,
#         print("in CorrelationFunction __init__")
#         self.save_for_backward(input1, input2)

#         with torch.cuda.device_of(input1):
#             rbot1 = input1.new()
#             rbot2 = input2.new()
#             output = input1.new()

#             correlation_cuda.forward(input1, input2, rbot1, rbot2, output, 
#                 self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2, self.corr_multiply)

#         print("returning CorrelationFunction forward")
#         return output

#     # @staticmethod
#     def backward(self, grad_output): # self, 
#         input1, input2 = self.saved_tensors

#         with torch.cuda.device_of(input1):
#             rbot1 = input1.new()
#             rbot2 = input2.new()

#             grad_input1 = input1.new()
#             grad_input2 = input2.new()

#             correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
#                 self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2, self.corr_multiply)

#         return grad_input1, grad_input2

# class Correlation(Module):
#     def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
#         super(Correlation, self).__init__()
#         print("in Correlation __init__")
#         print("self.pad_size = ", pad_size)
#         self.pad_size = pad_size
#         self.kernel_size = kernel_size
#         self.max_displacement = max_displacement
#         self.stride1 = stride1
#         self.stride2 = stride2
#         self.corr_multiply = corr_multiply


#     # @staticmethod
#     def forward(self, input1, input2): # self, self for the object; when you call, it is without - synthactic sugar!
#         print("in Correlation forward")

#         result = CorrelationFunction(self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply)(input1, input2)
#         # result = CorrelationFunction(pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply)(input1, input2)

#         return result

class Correlation(torch.autograd.Function):

    # @staticmethod
    def forward(ctx, input1, input2, 
            pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1): 
            # self, self for the object; when you call, it is without - synthactic sugar!
        print("in Correlation forward")
        print("self.pad_size = ", pad_size)

        result = CorrelationFunction(input1, input2, pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply)
        # result = CorrelationFunction(pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply)(input1, input2)

        return result
"""

