import numpy as np
import matplotlib.pyplot as plt
import colorsys
import math
import tqdm

import torch

from metrics import calculate_psnr

"""
Assume that your set of n timesteps is T={t_0, t1, ..., t{n-1}}; then, 
what you compute are distances between all pairs of timesteps d(t_k, t_l) with k<l (i.e,. you compute a distance matrix);
the distance is the sum of differences (according to some metrics) of the time step interpolates in between (k<t<l) and the original data. 
This should result in a triangular distance matrix
"""

# def calculate_dist_matrix(original_data_gt, interpolated_data):
#     """Computes the distance matrix between interpolated timesteps using PSNR as the distance metric"""
#     n_timesteps = len(interpolated_data)
#     dist_matrix = np.zeros((n_timesteps, n_timesteps))
#     # lpips_model = lpips.LPIPS(net='alex').cuda() # vgg
#     for i in tqdm(range(n_timesteps)):
#         for j in range(i + 1, n_timesteps):
#             dist = 0.0
#             for k in range(i + 1, j):
#                 interp = interpolated_data[k]
#                 orig = original_data_gt[k]
#                 dist += calculate_psnr(orig, interp) # 25h on SVCG, 9h on Habrok
#                 # orig = np.expand_dims(orig, axis=0)
#                 # interp = np.expand_dims(interp, axis=0)
#                 # orig_tensor = torch.from_numpy(orig).float()
#                 # interp_tensor = torch.from_numpy(interp).float()
#                 # dist += lpips_model(orig_tensor.cuda(), interp_tensor.cuda()) # very long
#             dist_matrix[i, j] = dist
#     return dist_matrix # upper triangular

def calculate_dist_matrix_batch(original_data_gt, model, device):
    n_timesteps = original_data_gt.shape[0]
    dist_matrix = np.zeros((n_timesteps, n_timesteps))
    batch_size = 16  # Adjust batch size as per your available memory

    for i in tqdm.tqdm(range(n_timesteps)):
        for j in range(i + 2, n_timesteps, batch_size):
            dist_batch = np.zeros(batch_size)
            interpolated_batch = []
            data_test_i_batch = []
            data_test_j_batch = []

            for k in range(j - batch_size + 1, j + 1):
                data_test_i_k, data_test_j_k, concat_in_between_k = create_cat(original_data_gt, i, k)
                data_test_i_batch.append(data_test_i_k)
                data_test_j_batch.append(data_test_j_k)

            data_test_i_batch = torch.from_numpy(np.stack(data_test_i_batch)).unsqueeze(1).to(device)
            data_test_j_batch = torch.from_numpy(np.stack(data_test_j_batch)).unsqueeze(1).to(device)

            with torch.no_grad():
                factor = j - i
                for k in range(1, factor):
                    interpol_step = float(k) / factor
                    merged, flow, mask = model.inference(data_test_i_batch, data_test_j_batch, t=interpol_step)
                    merged_last = np.asarray(merged[-1].detach().cpu().numpy())
                    merged_last = merged_last.squeeze(1)
                    interpolated_batch.append(merged_last.transpose((0, 2, 3, 1)))

            interpolated_batch = np.concatenate(interpolated_batch, axis=0)
            dist_batch += calculate_psnr(concat_in_between_k, interpolated_batch)

            dist_matrix[i, j - batch_size + 1:j + 1] = dist_batch

    return dist_matrix

# def calculate_dist_matrix(data_test, model, device):
#     n_timesteps = data_test.shape[0]
#     print(n_timesteps)
#     dist_matrix = np.zeros((n_timesteps, n_timesteps))
#     for i in tqdm.tqdm(range(n_timesteps)):
#         for j in range(i + 2, n_timesteps):
#             dist = 0.0
#             interpolated = []
#             # print(i, j, data_test.shape)

#             # create cat(I0, In), get orig_gt(I1...In-1)
#             data_test_i, data_test_j, concat_in_between = create_cat(data_test, i, j)
#             data_test_i = torch.from_numpy(data_test_i).unsqueeze(0).to(device)
#             data_test_j = torch.from_numpy(data_test_j).unsqueeze(0).to(device)         
#             # print(data_test_i.shape)
#             # print(concat_in_between.shape)
#             # input("cat")

#             # interpolte in between (run inference(I0, In))
#             factor = j - i
#             for j in range(1, factor):
#                 interpol_step = float(j) / factor
#                 # print(interpol_step)
#                 merged, flow, mask = model.inference(data_test_i, data_test_j, t=interpol_step)
#                 merged_last = np.asarray(merged[-1].detach().cpu().numpy()) # from last block
#                 merged_last = merged_last.squeeze(0)
#                 # print(merged_last.shape)
#                 # input("merged")
#                 interpolated.append(merged_last.transpose((1,2,0)))

#             # calc error
#             interpolated = np.array(interpolated)
#             interpolated = interpolated.squeeze(-1)
#             # print(concat_in_between.shape)
#             # print(interpolated.shape)
#             # input("diff")
#             dist += calculate_psnr(concat_in_between, interpolated) 

#             dist_matrix[i, j] = dist

def calculate_dist_matrix(data_test, model, device):
    n_timesteps = data_test.shape[0]
    print(n_timesteps)
    dist_matrix = np.zeros((n_timesteps, n_timesteps))
    for i in tqdm.tqdm(range(n_timesteps)):
        dist = 0.0  # Initialize dist for each i iteration
        for j in range(i + 2, n_timesteps):
            interpolated = []

            # create cat(I0, In) and orig_gt(I1...In-1)
            data_test_i, data_test_j, concat_in_between = create_cat(data_test, i, j)
            data_test_i = torch.from_numpy(data_test_i).unsqueeze(0).to(device)
            data_test_j = torch.from_numpy(data_test_j).unsqueeze(0).to(device)

            # interpolate in between (run inference(I0, In))
            factor = j - i
            interpol_steps = np.linspace(1 / factor, (factor - 1) / factor, factor - 1)
            for interpol_step in interpol_steps:
                merged, _, _ = model.inference(data_test_i, data_test_j, t=interpol_step)
                merged_last = np.asarray(merged[-1].detach().cpu().numpy())  # from last block
                merged_last = merged_last.squeeze(0)
                interpolated.append(merged_last.transpose((1, 2, 0)))

            # calculate error
            interpolated = np.array(interpolated).squeeze(-1)
            dist += calculate_psnr(concat_in_between, interpolated)

            dist_matrix[i, j] = dist

    return dist_matrix

def create_cat(data_test, i, j):
    # n = j - i
    concat_in_between = [np.expand_dims(d, axis=0) for _, d in enumerate(data_test[i+1:j])]
    concat_in_between = np.concatenate(concat_in_between, axis=0)
    # ensure same number of dimensions
    data_test_i = np.expand_dims(data_test[i], axis=0)
    data_test_j = np.expand_dims(data_test[j], axis=0)
    # print(np.array(concat_in_between).shape)
    # print(np.array(data_test_i).shape)
    # print(np.array(data_test_i_n).shape)
    return data_test_i, data_test_j, concat_in_between

# def create_cat(data, i, j):
#     data_i = data[i]
#     data_j = data[j]
#     concat_in_between = np.concatenate(data[i+1:j], axis=0)

#     return data_i, data_j, concat_in_between








