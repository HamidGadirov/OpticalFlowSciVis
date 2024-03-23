import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def linear_interpolation(original_data, factor):
    # Convert the array to a PyTorch tensor
    tensor_data = torch.from_numpy(original_data)

    # Reshape the tensor to match the expected input shape for interpolate
    # tensor_data = tensor_data.unsqueeze(1)
    if tensor_data.ndim == 3: # 2D data
        tensor_data = tensor_data.permute(1, 2, 0)

        # expected: mini-batch x channels x [optional depth] x [optional height] x width.
        interpolated_tensor = F.interpolate(tensor_data, scale_factor=(factor), mode='linear') # nearest
        # print("interpolated_tensor:", interpolated_tensor.shape)
        # input("x")

        interpolated_tensor = interpolated_tensor.permute(2, 0, 1)

        # Convert the interpolated tensor to a NumPy array
        interpolated_data = interpolated_tensor.squeeze().numpy()

    elif tensor_data.ndim == 4: # 3D data
        tensor_data = tensor_data.permute(1, 2, 3, 0)
        print("t", tensor_data.shape)

        interpolated_tensors = []

        # Perform linear interpolation along the first dimension
        for i in range(tensor_data.shape[0]):
            interpolated_slice = F.interpolate(tensor_data[i:i+1].squeeze(), scale_factor=factor, mode='linear', align_corners=False)
            # print("interpolated_slice", interpolated_slice.shape)
            interpolated_tensors.append(np.array(interpolated_slice.numpy()))

        interpolated_data = np.array(interpolated_tensors)
        print("interpolated_data", interpolated_data.shape)

        interpolated_data = interpolated_data.transpose(3, 0, 1, 2)

    interpolated_data = [d for i, d in enumerate(interpolated_data) if i % factor != 0]
    interpolated_data = np.array(interpolated_data)

    return interpolated_data

def lanczos_interpolation(original_data):
    # Convert the array to a PIL Image
    image = Image.fromarray(original_data)

    # Resize the image using Lanczos interpolation
    interpolated_image = image.resize((original_data.shape[0] * 2), resample=Image.LANCZOS)

    # Convert the interpolated image to a NumPy array
    interpolated_data = np.array(interpolated_image)

    return interpolated_data