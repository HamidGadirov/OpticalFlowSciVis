import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def bilinear_interpolation(original_data_gt):
    # Convert the array to a PyTorch tensor
    tensor_data = torch.from_numpy(original_data_gt)

    # Reshape the tensor to match the expected input shape for interpolate
    tensor_data = tensor_data.unsqueeze(0).unsqueeze(0)

    # Perform bilinear interpolation using PyTorch's interpolate function
    interpolated_tensor = F.interpolate(tensor_data, scale_factor=2, mode='bilinear')

    # Convert the interpolated tensor back to a numpy array
    interpolated_data = interpolated_tensor.squeeze().numpy()

    return interpolated_data

def cubic_spline_interpolation(original_data_gt):
    # Convert the array to a PyTorch tensor
    tensor_data = torch.from_numpy(original_data_gt)

    # Reshape the tensor to match the expected input shape for interpolate
    tensor_data = tensor_data.unsqueeze(0).unsqueeze(0)

    # Perform cubic spline interpolation using PyTorch's interpolate function
    interpolated_tensor = F.interpolate(tensor_data, scale_factor=2, mode='bicubic')

    # Convert the interpolated tensor back to a numpy array
    interpolated_data = interpolated_tensor.squeeze().numpy()

    return interpolated_data

def lanczos_interpolation(original_data_gt):
    # Convert the array to a PIL Image
    image = Image.fromarray(original_data_gt)

    # Resize the image using Lanczos interpolation
    interpolated_image = image.resize((original_data_gt.shape[1]*2, original_data_gt.shape[0]*2), resample=Image.LANCZOS)

    interpolated_data = np.array(interpolated_image)

    return interpolated_data