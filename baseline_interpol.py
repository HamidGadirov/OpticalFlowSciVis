import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def linear_interpolation(original_data):
    # Convert the array to a PyTorch tensor
    tensor_data = torch.from_numpy(original_data)

    # Reshape the tensor to match the expected input shape for interpolate
    # tensor_data = tensor_data.unsqueeze(1)
    tensor_data = tensor_data.permute(1, 2, 0)
    print("t", tensor_data.shape)

    # expected: mini-batch x channels x [optional depth] x [optional height] x width.
    interpolated_tensor = F.interpolate(tensor_data, scale_factor=(2), mode='linear') # nearest

    interpolated_tensor = interpolated_tensor.permute(2, 0, 1)

    # Convert the interpolated tensor to a NumPy array
    interpolated_data = interpolated_tensor.squeeze().numpy()

    interpolated_data = interpolated_data[1::2]

    return interpolated_data

def lanczos_interpolation(original_data):
    # Convert the array to a PIL Image
    image = Image.fromarray(original_data)

    # Resize the image using Lanczos interpolation
    interpolated_image = image.resize((original_data.shape[0] * 2), resample=Image.LANCZOS)

    # Convert the interpolated image to a NumPy array
    interpolated_data = np.array(interpolated_image)

    return interpolated_data