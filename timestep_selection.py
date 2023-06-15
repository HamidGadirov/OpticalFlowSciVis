import numpy as np
import matplotlib.pyplot as plt
import colorsys
import math
import tqdm

from metrics import calculate_psnr

"""
Assume that your set of n timesteps is T={t_0, t1, ..., t{n-1}}; then, 
what you compute are distances between all pairs of timesteps d(t_k, t_l) with k<l (i.e,. you compute a distance matrix);
the distance is the sum of differences (according to some metrics) of the time step interpolates in between (k<t<l) and the original data. 
This should result in a triangular distance matrix
"""

def calculate_dist_matrix(original_data_gt, interpolated_data):
    """Computes the distance matrix between interpolated timesteps using PSNR as the distance metric"""
    n_timesteps = len(interpolated_data)
    dist_matrix = np.zeros((n_timesteps, n_timesteps))
    # lpips_model = lpips.LPIPS(net='alex').cuda() # vgg
    for i in tqdm(range(n_timesteps)):
        for j in range(i + 1, n_timesteps):
            dist = 0.0
            for k in range(i + 1, j):
                interp = interpolated_data[k]
                orig = original_data_gt[k]
                dist += calculate_psnr(orig, interp) # 25h on SVCG, 9h on Habrok
                # orig = np.expand_dims(orig, axis=0)
                # interp = np.expand_dims(interp, axis=0)
                # orig_tensor = torch.from_numpy(orig).float()
                # interp_tensor = torch.from_numpy(interp).float()
                # dist += lpips_model(orig_tensor.cuda(), interp_tensor.cuda()) # very long
            dist_matrix[i, j] = dist
    return dist_matrix # upper triangular

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








