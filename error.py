import argparse
from curses.ascii import FF
import os
from re import I
import cv2
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as patches
# import skimage.measure
# print("scikit-image version: {}".format(skimage.__version__))
import math
import statistics
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from utils import visualize_ind, visualize_series

# TODO: add frames or images that were GT for interpolation
# show full video

def calculate_psnr(img1, img2):
    """
    Calculate the peak signal-to-noise ratio (PSNR) between two images.
    
    Parameters:
        img1 (numpy array): The first image.
        img2 (numpy array): The second image.
    
    Returns:
        float: The PSNR value between the two images.
    """
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    """
    Calculate the structural similarity index (SSIM) between two images.
    
    Parameters:
        img1 (numpy array): The first image.
        img2 (numpy array): The second image.
    
    Returns:
        float: The SSIM value between the two images.
    """
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_metrics(original_data, interpol_data, factor):
    """
    Calculate the peak signal-to-noise ratio (PSNR) and structural similarity index (SSIM) between two datasets,
    and return the mean values of these metrics.
    
    Parameters:
        original_data (numpy array): The original data.
        interpol_data (numpy array): The interpolated data.
        factor (int): The factor by which the data is being interpolated.
    
    Returns:
        tuple: A tuple containing the mean PSNR and SSIM values between the two datasets.
    """
    psnr_scores_original = []
    ssim_scores_original = []
    psnr_scores_interpol = []
    ssim_scores_interpol = []

    data_range = min(original_data.shape[0], interpol_data.shape[0])
    for i in range(data_range):
        psnr_score = calculate_psnr(original_data[i], interpol_data[i])
        ssim_score = calculate_ssim(original_data[i], interpol_data[i])
        # psnr_scores_interpol.append(psnr_score)
        # ssim_scores_interpol.append(ssim_score)
        if i % factor != 0:
            psnr_scores_interpol.append(psnr_score)
            ssim_scores_interpol.append(ssim_score)
        else:
            psnr_scores_original.append(psnr_score)
            ssim_scores_original.append(ssim_score)
            # print(original_data[i])
            # print(interpol_data[i])
            # not same because inference model is only student

    # input("x")
    # print(psnr_scores_interpol)

    psnr_scores_interpol_mean = statistics.mean(psnr_scores_interpol)
    ssim_scores_interpol_mean = statistics.mean(ssim_scores_interpol)

    return psnr_scores_interpol_mean, ssim_scores_interpol_mean

    # psnr_scores_original_mean = statistics.mean(psnr_scores_original)
    # ssim_scores_original_mean = statistics.mean(ssim_scores_original)

    print("psnr_scores_interpol_mean: %f dB" % psnr_scores_interpol_mean)
    print("ssim_scores_interpol_mean: %f" % ssim_scores_interpol_mean)

    # print("psnr_scores_original_mean: %f dB" % psnr_scores_original_mean)
    # print("ssim_scores_original_mean: %f" % ssim_scores_original_mean)

    total_error_psnr = round(sum(psnr_scores_interpol), 2)
    total_psnr_scores.append(total_error_psnr)
    print("total_error_psnr:", total_error_psnr) # PSNR - higher better
    print("mean_error_psnr:", psnr_scores_interpol_mean) # PSNR - higher better
    mean_psnr_scores.append(round(psnr_scores_interpol_mean, 2))
    mean_ssim_scores.append(round(ssim_scores_interpol_mean, 3))

    # interpol_data = np.array(interpol_data)

    # The higher the PSNR, the better the quality of the compressed, or reconstructed image.
    # The range of SSIM values extends between -1 and +1 and only equals 1 if the two images are identical. 

    selection = False
    if selection:
        # todo: change this!
        threshold = psnr_scores_interpol_mean - psnr_scores_interpol_mean / 10.
        # threshold = psnr_scores_interpol_mean - (psnr_scores_original_mean - psnr_scores_interpol_mean) / 4
        print("threshold: ", threshold)
        selected_timesteps = []
        k = 0
        # print(len(psnr_scores_interpol))
        for i in range(interpol_data.shape[0]):
            # print(psnr_scores_interpol[k])
            if i % factor != 0: # and psnr_scores_interpol[k] < threshold:
                psnr_score = calculate_psnr(original_data[i,...], interpol_data[i,...])
                if psnr_score < threshold:
                    # take the frame
                    selected_timesteps.append(original_data[i, ...])
                    # k += 1
                    # print(psnr_score)

        selected_timesteps = np.array(selected_timesteps)
        print(selected_timesteps.shape)

    # input("x")

    print("total_psnr_scores", total_psnr_scores)
    #  total_psnr_scores [6317.07, 8888.87, 9297.37, 8755.94, 7687.85, 7025.27, 5269.31]
    print("mean_psnr_scores", mean_psnr_scores)
    print("mean_ssim_scores", mean_ssim_scores)

def create_data_for_interpol(factor, filename):
    """
    Creates a new array of data by selecting every 'factor'-th element of the data read from the specified 'filename' file. 
    The data is first normalized so that the maximum value is 255, and the data type is changed to 'int'. 
    The original data and the new data array are then returned.
    
    Parameters:
    factor (int): The interval at which elements are selected from the data.
    filename (str): The name of the file to read the data from.
    
    Returns:
    tuple: A tuple containing the original data array and the new data array.
    """
    # load member
    pkl_file = open(filename, 'rb')
    data = []
    data = pickle.load(pkl_file)
    print(data.shape)
    # print(data)
    # print(np.max(data))
    # print(np.min(data))
    data = data * 255.0 / data.max()
    data = data.astype(int)
    # print(data)
    print("Data is in range %f to %f dB" % (np.min(data), np.max(data))) # a tuple of length 2

    # with open('data.txt', 'w') as f:
    #     for row in data[500:505]:
    #         np.savetxt(f, row)
    # input("x")
    data = data[5:] # skip empty
    if "pipedcylinder2d" in filename or "cylinder2d" in filename:
        print("a subset of the member is selected")
        data = data[1250:1450] # member is too long
    
    if "rectangle2d" in filename:
        data = data[2700:3000]

    # if "pipedcylinder2d" in filename: 
    #     # padding
    #     canvas = np.zeros((data.shape[0], 160, 480))
    #     canvas[:, 0:150, 0:450] = data
    #     data = canvas
    #     print("padded:", data.shape)

    data_for_interpol = []
    for i in range(data.shape[0]):
        if i % factor == 0:
            # print(i)
            data_for_interpol.append(data[i])
    data_for_interpol = np.array(data_for_interpol)
    print("data_for_interpol:", data_for_interpol.shape)

    return data_for_interpol, data

def calculate_diff(original_data, interpol_data, dataset, factor, dir_res):
        original_video = dataset + "_10fps.mp4" # create it once!
        # original_video = video_name
        # interpol_video = "video_10fps_extract2x_2X_20fps.mp4"
        # interpol_video = "video_10fps_extract128x_128X_1280fps.mp4"
        interpol_video = dataset + "_10fps_extract" + str(factor) + "x_" + str(factor) + "X_" + str(factor*10) + "fps.mp4"
        # print("video_name:", interpol_video)

        # vidcap = cv2.VideoCapture(interpol_video)
        # count = 0
        # interpol_data = []

        # while True:
        #     # print("success")
        #     success, frame = vidcap.read()
        #     if not success:
        #         break
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     interpol_data.append(frame)
        #     count += 1

        # print(count)
        # interpol_data = np.array(interpol_data)
        print("interpol_data:", interpol_data.shape)

        diffs = []
        # test
        # interpol_data = np.array(interpol_data)
        # interpol_data = np.zeros((interpol_data.shape[0], interpol_data.shape[1], interpol_data.shape[2]), dtype=np.float32)
        data_range = min(original_data.shape[0], interpol_data.shape[0])
        # print(original_data.type, interpol_data.type)
        for i in range(data_range):
            # diffs.append(cv2.absdiff(np.int32(original_data[i]), np.int32(interpol_data[i])))
            # diffs.append(cv2.absdiff(np.float32(original_data[i]), np.float32(interpol_data[i])))
            diffs.append(cv2.absdiff(original_data[i], interpol_data[i]))
        diffs = np.array(diffs)
        # print(diffs[0])
        # input("x")
        # contrast stretching
        print("Diff", diffs.min(), diffs.max())
        diffs = (diffs - diffs.min()) / (diffs.max() - diffs.min())
        # diffs = (diffs - diffs.min()) / (diffs.max() - diffs.min()) * 255. # no need if 0..1 range
        # print("Diff after stretch", diffs.min(), diffs.max())
        print("Diff min max:", diffs.min(), diffs.max())

        # quantify the error, e.g., giving the mean/dev/max 
        print("Diff mean and std:", np.mean(diffs), np.std(diffs))

        title = "Difference"
        # if "2d" in dataset:
        #     visualize_series(diffs, factor,  dataset, dir_res, title=title, show=False, save=True)

        # visualize(interpol_data[10,...])

        # print(dir(skimage.measure))
        # psnr_score = skimage.measure.peak_signal_noise_ratio(original_data[10,...], interpol_data[10,...])

        calc_metrics = False
        if calc_metrics:
            calculate_metrics(original_data, interpol_data, factor)

        return diffs

# TODO: 
# to see how long it works decently (see when it breaks) and check its suitability for time step reduction):
# incrementally change the number of time steps selected |S|, 
# interpolate between them, 
# sum up the total error for each |S|, 
# and present it in a chart

def create_gt_interpol(dataset, factor):
    # parser = argparse.ArgumentParser(description='Create videos, calculate metrics')
    # parser.add_argument('--dataset', dest='dataset', type=str, default=None)
    # args = parser.parse_args()
    # assert (not args.dataset is None)

    # dataset =  "pipedcylinder2d" # "cylinder2d" # "pipedcylinder2d" # "droplet2d"
    # dataset = args.dataset
    filename = "../Datasets/"
    if dataset == 'rectangle2d':
        filename += "rectangle2d.pkl"
    if dataset == "droplet3d":
        filename += "drop2D/..."
    if dataset == "tangaroa3d":
        filename += "tangaroa3d.pkl" # 300 x 180 x 120 x 201
    if dataset == "droplet2d":
        filename += "drop2D/droplet2d.pkl"
    elif dataset == "pipedcylinder2d":
        filename += "pipedcylinder2d.pkl"
    elif dataset == "cylinder2d":
        filename += "cylinder2d.pkl"
    elif dataset == "FluidSimML2d":
        filename += "FluidSimML_part.pkl"

    print(filename)

    # min_exp = 2 # 1
    # max_exp = 5 # 7
    # for exp in range(min_exp, max_exp+1):

        # factor = 2 ** exp
        # factors.append(factor)
        # print("factor:", factor) # 2 4 8 16 32 64 128

    dir_res = "Results"
    dir_res = os.path.join(dir_res, dataset)
    dir_res = os.path.join(dir_res, str(factor) + "x")
    print("Saving at:", dir_res)

    video_name = dataset + "_10fps_extract" + str(factor) + "x.mp4"
    data_for_interpol, original_data = create_data_for_interpol(factor, filename) # extracted and original
    fps = 10 # 20
    sizeX = data_for_interpol.shape[1]
    sizeY = data_for_interpol.shape[2]
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (sizeY, sizeX), False)
    for i in range(len(data_for_interpol)):
        out.write(data_for_interpol[i].astype('uint8'))
        # out.write(np.invert(data_for_interpol[i].astype('uint8'))) # try invert for droplet2d
    out.release()
    print("Created:", video_name)

    original_data = np.array(original_data)
    print("original_data:", original_data.shape)

    gt = []
    for i in range(original_data.shape[0]):
        # if i % factor != 0:
        gt.append(original_data[i])
    gt = np.array(gt)
    print("GT:", gt.shape)
    title = "Ground_truth"
    if "2d" in dataset:
        visualize_series(gt, factor,  dataset, dir_res, title=title, show=False, save=True)
    # input("GT created...")

    # calc_diff = True # True
    # calc_metrics = False

    # if calc_diff:
    #     interpol_data = calculate_diff(original_data, dataset_video_name, dataset, factor, dir_res)

    return original_data, video_name

calc_metrics = False
if calc_metrics:
    parser = argparse.ArgumentParser(description='Create videos, calculate metrics')
    parser.add_argument('--dataset', dest='dataset', type=str, default=None)
    args = parser.parse_args()
    assert (not args.dataset is None)

    # dataset =  "pipedcylinder2d" # "cylinder2d" # "pipedcylinder2d" # "droplet2d"
    dataset = args.dataset
    filename = "../Datasets/"
    if dataset == "droplet2d":
        filename += "drop2D/droplet2d.pkl"
    elif dataset == "pipedcylinder2d":
        filename += "pipedcylinder2d.pkl"
    elif dataset == "cylinder2d":
        filename += "cylinder2d.pkl"
    elif dataset == "FluidSimML2d":
        filename += "FluidSimML/FluidSimML.pkl" # FluidSimML_part
    print(filename)

    # get original data once
    data_for_interpol, original_data = create_data_for_interpol(8, filename) # extracted and original

    # total_psnr_scores = []
    mean_psnr_scores = []
    mean_ssim_scores = []
    mean_psnr_scores_baseline = []
    mean_ssim_scores_baseline = []

    factors = [2, 4, 8, 16, 32, 64, 128]
    # factors = [2, 4, 8]
    factors_str = list(map(str, factors))

    for factor in factors:
        interpol_video = dataset + "_10fps_extract" + str(factor) + "x_" + str(factor) + "X_" + str(factor*10) + "fps.mp4"
        print("video_name:", interpol_video)
        vidcap = cv2.VideoCapture(interpol_video)
        count = 0
        interpol_data = []
        while True:
            # print("success")
            success, frame = vidcap.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            interpol_data.append(frame)
            count += 1
        # vidcap.release()
        # print(count)
        interpol_data = np.array(interpol_data)
        print("interpol_data:", interpol_data.shape)
        mean_psnr_score, mean_ssim_score = calculate_metrics(original_data.squeeze(), interpol_data.squeeze(), factor)
        mean_psnr_scores.append(round(mean_psnr_score, 2))
        mean_ssim_scores.append(round(mean_ssim_score, 3))

        # linear interpol
        extract_video = dataset + "_10fps_extract" + str(factor) + "x.mp4"
        vidcap = cv2.VideoCapture(extract_video)
        extract_data = []
        while True:
            # print("success")
            success, frame = vidcap.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            extract_data.append(frame)
        # vidcap.release()
        extract_data = np.array(extract_data)
        print("extract_data:", extract_data.shape)

        linear_interpol_data = []
        linear_interpol_data.append(extract_data[0])
        for i in range(extract_data.shape[0] - 1):
            # if i % factor != 0: # interpolate
            for j in range(factor - 1): 
                fusion_map = (j + 1) / factor
                linear_interpol_data.append(fusion_map * extract_data[i] + (1 - fusion_map) * extract_data[i+1])
            # else:
            linear_interpol_data.append(extract_data[i+1])
        linear_interpol_data = np.array(linear_interpol_data)
        print("linear_interpol_data:", linear_interpol_data.shape)

        dir_res = "Results"
        dir_res = os.path.join(dir_res, dataset)
        dir_res = os.path.join(dir_res, str(factor) + "x")
        print("Saving at:", dir_res)
        title = "Baseline"
        if "2d" in dataset:
            visualize_series(linear_interpol_data, factor,  dataset, dir_res, title=title, show=False, save=True)
        # input("x")
        mean_psnr_score, mean_ssim_score = calculate_metrics(original_data.squeeze(), linear_interpol_data.squeeze(), factor)
        mean_psnr_scores_baseline.append(round(mean_psnr_score, 2))
        mean_ssim_scores_baseline.append(round(mean_ssim_score, 3))

    # print(mean_psnr_scores)
    # print(mean_psnr_scores_baseline)

    # mean_psnr_scores = [37.16, 34.86, 31.62, 27.8, 24.8, 22.3, 20.75]
    # mean_ssim_scores = [0.98, 0.97, 0.96, 0.95, 0.92, 0.9, 0.89]
    # factors_str = ''.join(str(factor) for factor in factors)
    factors_str = list(map(str, factors))

    dir_res = "Results"
    dir_res = os.path.join(dir_res, dataset)
    # dir_res = os.path.join(dir_res, str(factor) + "x")
    print("Saving at:", dir_res)

    fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])
    plt.bar(factors_str, mean_psnr_scores, color='b', width = 0.25)
    ax = plt.gca()
    ax.set_xlabel("Interpolation factor")
    ax.set_ylabel("PSNR")
    plt.show()
    # title = "Interpolation score PSNR"
    title = "PSNR"
    # fig.savefig(title)
    fig.savefig(os.path.join(dir_res, title), dpi = 300)

    fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])
    plt.bar(factors_str, mean_ssim_scores, color='g', width = 0.25)
    ax = plt.gca()
    ax.set_xlabel("Interpolation factor")
    ax.set_ylabel("SSIM")
    plt.show()
    title = "SSIM"
    fig.savefig(title)
    fig.savefig(os.path.join(dir_res, title), dpi = 300)

    fig = plt.figure()
    X_axis = np.arange(len(factors_str))
    width = 0.25
    plt.bar(X_axis - width/2., mean_psnr_scores, color='b', width = width, label = 'RIFE')
    plt.bar(X_axis + width/2., mean_psnr_scores_baseline, color='r', width = width, label = 'Linear')
    plt.xticks(X_axis, factors_str)
    ax = plt.gca()
    ax.set_xlabel("Interpolation factor")
    ax.set_ylabel("PSNR")
    plt.legend()
    plt.show()
    # title = "Interpolation score PSNR"
    title = "PSNR_baseline"
    # fig.savefig(title)
    fig.savefig(os.path.join(dir_res, title), dpi = 300)

    fig = plt.figure()
    X_axis = np.arange(len(factors_str))
    width = 0.25
    plt.bar(X_axis - width/2., mean_ssim_scores, color='g', width = width, label = 'RIFE')
    plt.bar(X_axis + width/2., mean_ssim_scores_baseline, color='r', width = width, label = 'Linear')
    plt.xticks(X_axis, factors_str)
    ax = plt.gca()
    ax.set_xlabel("Interpolation factor")
    ax.set_ylabel("SSIM")
    plt.legend()
    plt.show()
    # title = "Interpolation score PSNR"
    title = "SSIM_baseline"
    # fig.savefig(title)
    fig.savefig(os.path.join(dir_res, title), dpi = 300)
