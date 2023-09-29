import numpy as np
import matplotlib.pyplot as plt
import colorsys
import math
import tqdm

import torch
import torch.nn.functional as F

from metrics import calculate_sum_psnr

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

"""
# arbitrary interpol: factor = n
print("arbitrary interpolation")

for k in range(i + 1, j):
    interp = interpolated_data[k]
    orig = original_data_gt[k]
    dist += calculate_psnr(orig, interp) # 25h on SVCG, 9h on Habrok
    # orig = np.expand_dims(orig, axis=0)
    # interp = np.expand_dims(interp, axis=0)
    # orig_tensor = torch.from_numpy(orig).float()
    # interp_tensor = torch.from_numpy(interp).float()
    # dist += lpips_model(orig_tensor.cuda(), interp_tensor.cuda()) # very long
"""

# def calculate_dist_matrix_batch(original_data_gt, model, device):
#     n_timesteps = original_data_gt.shape[0]
#     dist_matrix = np.zeros((n_timesteps, n_timesteps))
#     batch_size = 16  # Adjust batch size as per your available memory

#     for i in tqdm.tqdm(range(n_timesteps)):
#         for j in range(i + 2, n_timesteps, batch_size):
#             dist_batch = np.zeros(batch_size)
#             interpolated_batch = []
#             data_test_i_batch = []
#             data_test_j_batch = []

#             for k in range(j - batch_size + 1, j + 1):
#                 data_test_i_k, data_test_j_k, concat_in_between_k = create_cat(original_data_gt, i, k)
#                 data_test_i_batch.append(data_test_i_k)
#                 data_test_j_batch.append(data_test_j_k)

#             data_test_i_batch = torch.from_numpy(np.stack(data_test_i_batch)).unsqueeze(1).to(device)
#             data_test_j_batch = torch.from_numpy(np.stack(data_test_j_batch)).unsqueeze(1).to(device)

#             with torch.no_grad():
#                 factor = j - i
#                 for k in range(1, factor):
#                     interpol_step = float(k) / factor
#                     merged, flow, mask = model.inference(data_test_i_batch, data_test_j_batch, t=interpol_step)
#                     merged_last = np.asarray(merged[-1].detach().cpu().numpy())
#                     merged_last = merged_last.squeeze(1)
#                     interpolated_batch.append(merged_last.transpose((0, 2, 3, 1)))

#             interpolated_batch = np.concatenate(interpolated_batch, axis=0)
#             dist_batch += calculate_psnr(concat_in_between_k, interpolated_batch)

#             dist_matrix[i, j - batch_size + 1:j + 1] = dist_batch

#     return dist_matrix

def calculate_dist_matrix(data_test, model, device):
    n_timesteps = data_test.shape[0]
    print(n_timesteps)
    dist_matrix = np.zeros((n_timesteps, n_timesteps))
    for i in tqdm.tqdm(range(n_timesteps)):
        for j in range(i + 2, n_timesteps):
            dist = 0.0
            # print(i, j)
            interpolated = []
            # print(i, j, data_test.shape)

            # create concatenations (I0, In) and (I1...In-1)
            data_test_i, data_test_j, concat_in_between = create_cat(data_test, i, j)
            data_test_i = torch.from_numpy(data_test_i).unsqueeze(0).to(device)
            data_test_j = torch.from_numpy(data_test_j).unsqueeze(0).to(device)         
            # print(data_test_i.shape)
            # print(concat_in_between.shape)
            # input("cat")

            # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            # # Plot data_test_i in the first subplot
            # axs[0].imshow(data_test_i.squeeze().detach().cpu().numpy(), cmap='gray')
            # axs[0].set_title('Data_test_i')
            # axs[0].axis('off')
            # # Plot data_test_j in the second subplot
            # axs[1].imshow(data_test_j.squeeze().detach().cpu().numpy(), cmap='gray')
            # axs[1].set_title('Data_test_j')
            # axs[1].axis('off')
            # plt.tight_layout()
            # fig.savefig("droplets_ij.png", dpi=300)
            # # input("x")
            # plt.close()

            # interpolte in between (run inference(I0, In))
            factor = j - i
            # LREST
            """
            for k in range(1, factor):
                interpol_step = float(k) / factor
                # print(factor)
                # print(interpol_step)
                # input("x")
                mask = np.zeros((1, 160, 224), dtype=np.float32)
                mask[0] = 1
                t_mask = mask * interpol_step
                t_mask = torch.from_numpy(t_mask).unsqueeze(0).to(device)
                # print(data_test_i.shape)
                # print(t_mask.shape)
                # input("inference")
                merged, flow, mask = model.inference(data_test_i, data_test_j, t_mask)
                merged_last = np.asarray(merged[-1].detach().cpu().numpy()) # from last block
                merged_last = merged_last.squeeze(0)
                # print(merged_last.shape)
                # input("merged")
                interpolated.append(merged_last.transpose((1,2,0)))
            # print("out loop k")
            """

            # Beseline linear
            tensor_data = torch.cat((data_test_i, data_test_j), dim=0).squeeze()
            # print(tensor_data.shape)
            tensor_data = tensor_data.permute(1, 2, 0)
            interpolated_tensor = F.interpolate(tensor_data, scale_factor=(factor), mode='linear') # nearest
            interpolated_tensor = interpolated_tensor.permute(2, 0, 1)
            # print("interpolated_tensor", interpolated_tensor.shape)
            interpolated_data = interpolated_tensor.detach().cpu().squeeze().numpy()
            interpolated = [d for i, d in enumerate(interpolated_data) if i % factor != 0]

            # calc error
            interpolated = np.array(interpolated)
            # interpolated = interpolated.squeeze(-1) # needed for LREST

            # print(concat_in_between.shape)
            # print(interpolated.shape)
            # fig, axs = plt.subplots(nrows=10, ncols=3, figsize=(10, 10))
            # axs = axs.flatten()
            # for k in range(interpolated.shape[0]):
            #     try:
            #         image = interpolated[k].squeeze()
            #         axs[k].imshow(image, cmap='gray')
            #         axs[k].axis('off')
            #         axs[k].set_title(str(k))  # Add label as the title
            #     except:
            #         break
            # # Adjust the spacing between the subplots
            # plt.tight_layout()
            # # Save the plot to a file
            # fig.savefig("interpolated_droplets.png", dpi=300)
            # # input("x")
            # plt.close()

            # input("diff")
            dist = calculate_sum_psnr(concat_in_between, interpolated) 
            # print(dist, i, j)
            dist_matrix[i, j] = dist
            
    return dist_matrix

# def calculate_dist_matrix_gpt(data_test, model, device):
#     n_timesteps = data_test.shape[0]
#     print(n_timesteps)
#     dist_matrix = np.zeros((n_timesteps, n_timesteps))
#     for i in tqdm.tqdm(range(n_timesteps)):
#         dist = 0.0  # Initialize dist for each i iteration
#         for j in range(i + 2, n_timesteps):
#             interpolated = []

#             # create cat(I0, In) and orig_gt(I1...In-1)
#             data_test_i, data_test_j, concat_in_between = create_cat(data_test, i, j)
#             data_test_i = torch.from_numpy(data_test_i).unsqueeze(0).to(device)
#             data_test_j = torch.from_numpy(data_test_j).unsqueeze(0).to(device)

#             # interpolate in between (run inference(I0, In))
#             factor = j - i
#             interpol_steps = np.linspace(1 / factor, (factor - 1) / factor, factor - 1)
#             for interpol_step in interpol_steps:
#                 merged, _, _ = model.inference(data_test_i, data_test_j, t=interpol_step)
#                 merged_last = np.asarray(merged[-1].detach().cpu().numpy())  # from last block
#                 merged_last = merged_last.squeeze(0)
#                 interpolated.append(merged_last.transpose((1, 2, 0)))

#             # calculate error
#             interpolated = np.array(interpolated).squeeze(-1)
#             dist += calculate_psnr(concat_in_between, interpolated)

#             dist_matrix[i, j] = dist

#     return dist_matrix

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








