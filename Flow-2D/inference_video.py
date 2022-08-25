# ssh -Y hamid@129.125.75.167
# python3 -m torch.distributed.launch --nproc_per_node=1 train.py --world_size=1 --dataset=droplet2d --mode=train
# python3 inference_video.py --exp=1 --dataset=FluidSimML2d

# rsync -avz --exclude '.git/*' -e 'ssh' /Users/hamidgadirov/Desktop/OpticalFlow/ hamid@129.125.75.167:~/Desktop/OpticalFlow/ 
# rsync -avz -e 'ssh' hamid@129.125.75.167:~/Desktop/OpticalFlow/RIFE/Results/ /Users/hamidgadirov/Desktop/OpticalFlow/RIFE/Results/

from nis import cat
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
from model.pytorch_msssim import ssim_matlab
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pyimof

from utils import visualize_ind, visualize_series, visualize_series_flow, visualize_large
from error import create_gt_interpol, calculate_diff

warnings.filterwarnings("ignore")

def transferAudio(sourceVideo, targetVideo):
    import shutil
    import moviepy.editor
    tempAudioFileName = "./temp/audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:

        # clear old "temp" directory if it exits
        if os.path.isdir("temp"):
            # remove temp directory
            shutil.rmtree("temp")
        # create new "temp" directory
        os.makedirs("temp")
        # extract audio from video
        os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    # combine audio file and new video file
    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))

    if os.path.getsize(targetVideo) == 0: # if ffmpeg failed to merge the video and audio together try converting the audio to aac
        tempAudioFileName = "./temp/audio.m4a"
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        if (os.path.getsize(targetVideo) == 0): # if aac is not supported by selected format
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")

            # remove audio-less video
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    # remove temp directory
    shutil.rmtree("temp")

def interpolate(args, video_name,  dataset, factor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if(args.fp16):
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

    try:
        try:
            from model.oldmodel.RIFE_HDv2 import Model
            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v2.x HD model.")
        except:
            from train_log.RIFE_HDv3 import Model
            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v3.x HD model.")
    except:
        from model.oldmodel.RIFE_HD import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v1.x HD model")
    model.eval()
    model.device()
    # print(model)
    # input("x")

    args.video = video_name
    if not args.video is None:
        videoCapture = cv2.VideoCapture(args.video)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

        videoCapture.release()
        if args.fps is None:
            fpsNotAssigned = True
            args.fps = fps * (2 ** args.exp)
        else:
            fpsNotAssigned = False
        videogen = skvideo.io.vreader(args.video)
        lastframe = next(videogen)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_path_wo_ext, ext = os.path.splitext(args.video)
        print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
        print("args:", args.png, fpsNotAssigned)
        if args.png == False and fpsNotAssigned == True:
            print("The audio will be merged after interpolation process")
        else:
            print("Will not merge audio because using png or fps flag!")
    else:
        videogen = []
        for f in os.listdir(args.img):
            if 'png' in f:
                videogen.append(f)
        tot_frame = len(videogen)
        videogen.sort(key= lambda x:int(x[:-4]))
        lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
        videogen = videogen[1:]
    h, w, _ = lastframe.shape
    vid_out_name = None
    vid_out = None
    if args.png:
        if not os.path.exists('vid_out'):
            os.mkdir('vid_out')
    else:
        if args.output is not None:
            vid_out_name = args.output
        else:
            vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, (2 ** args.exp), int(np.round(args.fps)), args.ext)
        vid_out = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (w, h))

    def clear_write_buffer(user_args, write_buffer):
        cnt = 0
        while True:
            item = write_buffer.get()
            if item is None:
                break
            if user_args.png:
                cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
                cnt += 1
            else:
                vid_out.write(item[:, :, ::-1])

    def build_read_buffer(user_args, read_buffer, videogen):
        try:
            for frame in videogen:
                if not user_args.img is None:
                    frame = cv2.imread(os.path.join(user_args.img, frame))[:, :, ::-1].copy()
                if user_args.montage:
                    frame = frame[:, left: left + w]
                read_buffer.put(frame)
        except:
            pass
        read_buffer.put(None)

    def make_inference(I0, I1, n):
        # global model
        middle, flow, mask = model.inference(I0, I1, args.scale) # get the flow too
        # print(type(middle))
        # print(len(middle))
        # print(type(flow))
        # print("flow:", len(flow))
        # middle_array = np.asarray(middle.cpu())
        # print(middle_array.shape) # (1, 3, 160, 224)

        flow_array = np.asarray(flow)
        # print(flow_array.shape) # (3,) because 3 IFBlocks, check RIFE_HD update()
        # print(flow_array[2].cpu().numpy().shape) # torch.Size([1, 4, 160, 224])
        flow_array_ = flow_array[2].cpu().numpy()
        # visualize_ind(flow_array_[:,2,:,:].squeeze())
        flow_combined.append(flow_array_[:,0:4,:,:].squeeze()) # 0:4 because in x and y and for F_t->0, F_t->1 intermediate flows
        # print("flow_combined", len(flow_combined))

        # mask_array = mask.cpu().numpy()
        # print(mask_array.shape) # (1, 1, 160, 224)
        # visualize_ind(mask_array[0,0,:,:].squeeze())
        
        if n == 1:
            return [middle]
            
        first_half = make_inference(I0, middle, n=n//2) # the floor division // rounds the result down to the nearest whole number
        second_half = make_inference(middle, I1, n=n//2)
        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

    def pad_image(img):
        if(args.fp16):
            return F.pad(img, padding).half()
        else:
            return F.pad(img, padding)

    if args.montage:
        left = w // 4
        w = w // 2
    tmp = max(32, int(32 / args.scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    pbar = tqdm(total=tot_frame)
    if args.montage:
        lastframe = lastframe[:, left: left + w]
    write_buffer = Queue(maxsize=500)
    read_buffer = Queue(maxsize=500)
    _thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
    _thread.start_new_thread(clear_write_buffer, (args, write_buffer))

    I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)
    temp = None # save lastframe when processing static frame

    flow_combined = []

    while True:
        if temp is not None:
            frame = temp
            temp = None
        else:
            frame = read_buffer.get()
        if frame is None:
            break
        I0 = I1
        I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = pad_image(I1)
        I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

        break_flag = False
        if ssim > 0.996:        
            frame = read_buffer.get() # read a new frame
            if frame is None:
                break_flag = True
                frame = lastframe
            else:
                temp = frame
            I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
            I1 = pad_image(I1)
            I1, flow, mask = model.inference(I0, I1, args.scale) # get the flow too
            I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
        
        if ssim < 0.2:
            output = []
            for i in range((2 ** args.exp) - 1):
                output.append(I0)
            '''
            output = []
            step = 1 / (2 ** args.exp)
            alpha = 0
            for i in range((2 ** args.exp) - 1):
                alpha += step
                beta = 1-alpha
                output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
            '''
        else:
            output = make_inference(I0, I1, 2**args.exp-1) if args.exp else []

        if args.montage:
            write_buffer.put(np.concatenate((lastframe, lastframe), 1))
            for mid in output:
                mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                write_buffer.put(np.concatenate((lastframe, mid[:h, :w]), 1))
        else:
            write_buffer.put(lastframe)
            for mid in output:
                mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                write_buffer.put(mid[:h, :w])
        pbar.update(1)
        lastframe = frame
        if break_flag:
            break

    if args.montage:
        write_buffer.put(np.concatenate((lastframe, lastframe), 1))
    else:
        write_buffer.put(lastframe)
    import time
    while(not write_buffer.empty()):
        time.sleep(0.1)
    pbar.close()
    # print("skipping release") # todo
    if not vid_out is None:
        # print("vid_out.release()")
        vid_out.release()

    # flow_combined = []

    # images from video
    vidcap = cv2.VideoCapture(vid_out_name)
    count = 0
    interpol_data = []

    while True:
        success, frame = vidcap.read()
        if not success:
            break
        # todo: avoid this
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # if count % factor != 0: # only for missing frames
        interpol_data.append(frame)
        # count += 1

    # print(count)
    interpol_data = np.array(interpol_data)
    print("Interpolated data:", interpol_data.shape)

    title = "Interpolated_timesteps_" + str(factor) + "x"
    if "2d" in dataset:
        visualize_series(interpol_data, factor, dataset, dir_res, title=title, show=False, save=True)

    # print("Flow vis")
    flow_combined = np.array(flow_combined)
    print("Flow:", flow_combined.shape) # 170 255 294 310 321-5=315

    hsv = np.empty(shape=(flow_combined.shape[0], flow_combined.shape[2], flow_combined.shape[3], 3), dtype=np.uint8)
    # print("hsv:", hsv.shape)

    for i in range(flow_combined.shape[0]):
        hsv[i,:,:,1] = 255
        f1 = flow_combined[i, 0, ...]
        f2 = flow_combined[i, 1, ...]
        mag, ang = cv2.cartToPolar(f1, f2)
        hsv[i, ..., 0] = ang * 180 / np.pi / 2
        hsv[i, ..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # input("x")

    bgr = np.empty(shape=hsv.shape, dtype=np.uint8)
    for i in range(flow_combined.shape[0]):
        bgr[i, ...] = cv2.cvtColor(hsv[i, ...], cv2.COLOR_HSV2BGR)
    print("Flow in bgr:", bgr.shape)

    # droplet2d
    # Interpolated data: (125, 160, 224)                                                                                                                    
    # Flow: (125, 4, 160, 224)                                                                                                                              
    # Flow in bgr: (125, 160, 224, 3)

    # pipedcylinder2d
    # Interpolated data: (100, 150, 450)
    # Flow: (100, 4, 160, 480)
    # Flow in bgr: (100, 160, 480, 3)

    # cylinder2d
    # Interpolated data: (180, 80, 640)
    # Flow: (180, 4, 96, 640)
    # Flow in bgr: (180, 96, 640, 3)

    # this is because of fiss scale levels at IFBlocks in IFNet(), 
    # must be divisible to 32 to have same output dim 

    title = "FlowHSV_01_" + str(factor) + "x"
    if "2d" in dataset:
        visualize_series(bgr, factor, dataset, dir_res, title=title, show=False, save=True)

    # Vector field quiver plot
    flow_u = flow_combined[:, 0, ...]
    flow_v = flow_combined[:, 1, ...]
    title = "Flow_arrows_" + str(factor) + "x"
    if "2d" in dataset:
        visualize_series_flow(interpol_data, flow_u, flow_v, dataset, dir_res, title=title, show=False, save=True)

    print ("flow_combined:", flow_combined.shape)
    print ("interpol_data:", interpol_data.shape)
    return interpol_data, flow_combined

    # move audio to new video file if appropriate
    if args.png == False and fpsNotAssigned == True and not args.video is None:
        try:
            transferAudio(args.video, vid_out_name)
        except:
            print("Audio transfer failed. Interpolated video will have no audio")
            targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
            os.rename(targetNoAudio, vid_out_name)

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
# parser.add_argument('--video', dest='video', type=str, default=None)
parser.add_argument('--dataset', dest='dataset', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--img', dest='img', type=str, default=None)
parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames before processing')
parser.add_argument('--fps', dest='fps', type=int, default=None)
parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
parser.add_argument('--exp', dest='exp', type=int, default=1)
args = parser.parse_args()
assert (not args.dataset is None or not args.img is None)
if args.skip:
    print("skip flag is abandoned, please refer to issue #207.")
if args.UHD and args.scale==1.0:
    args.scale = 0.5
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
if not args.img is None:
    args.png = True

# dataset = "droplet2d" # "cylinder2d" # "pipedcylinder2d" # "droplet2d"
dataset = args.dataset
# if "droplet2d" in args.video:
#     dataset = "droplet2d"
# elif "pipedcylinder2d" in args.video:
#     dataset = "pipedcylinder2d"
# elif "cylinder2d" in args.video:
#     dataset = "cylinder2d"
# elif "FluidSimML2d" in args.video:
#     dataset = "FluidSimML2d"
print("Dataset:", dataset)

min_exp = args.exp # 2 # 1
max_exp = 7 # 7
factors = []

for exp in range(min_exp, max_exp + 1):
    args.exp = exp
    factor = 2 ** exp
    factors.append(factor)
    print("{}x interpolation".format(factor)) # 2 4 8 16 32 64 128
    dir_res = "Results"
    dir_res = os.path.join(dir_res, dataset)
    dir_res = os.path.join(dir_res, str(factor) + "x")
    print("Saving at:", dir_res)

    # GT, create interpol data
    original_data, video_name = create_gt_interpol(dataset, factor)

    # interpol, flow
    print("video_name:", video_name)
    # interpol_data, flow_combined = interpolate(args, video_name, dataset, factor)
    interpol_data, flow_combined = interpolate(args, video_name, dataset, factor)

    # # calc diff, metrics
    diffs = calculate_diff(original_data, interpol_data, dataset, factor, dir_res)

    # add baseline
    # data_to_vis = np.concatenate((np.array(original_data), np.array(interpol_data), np.array(diffs)), axis=0)
    # print(data_to_vis.shape)
    title = "GT_Interpol_Diff_Flow_" + str(factor) + "x"
    if "2d" in dataset:
        visualize_large(original_data, interpol_data, flow_combined, diffs, factor, dataset, dir_res, title=title, show=False, save=True)