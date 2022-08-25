import os
import cv2
import ast
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VimeoDataset(Dataset):
    def __init__(self, dataset_name, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name        
        self.h = 256
        self.w = 448
        self.data_root = '../Datasets/vimeo_triplet'
        self.image_root = os.path.join(self.data_root, 'sequences')
        # self.data_root = '../Datasets/vimeo_interp_test' # 'vimeo_triplet'
        # self.image_root = os.path.join(self.data_root, 'sequences')
        # train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        # train_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        # test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()   
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        # print("load_data")
        cnt = int(len(self.trainlist) * 0.95)
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist[:cnt]
        elif self.dataset_name == 'test':
            self.meta_data = self.testlist
        else:
            self.meta_data = self.trainlist[cnt:]
            

    def aug(self, img0, gt, img1, h, w):
        # print(img0.shape)
        # input("x")
        # ih, iw, _ = img0.shape
        ih, iw, = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w] # :]
        img1 = img1[x:x+h, y:y+w]
        gt = gt[x:x+h, y:y+w]
        return img0, gt, img1

    def getimg(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']

        # Load images
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        # input("get")
        # input(type(img0))

        # additions:	
        # convert to gray
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # to 0..1 range
        img0 = cv2.normalize(img0, img0, 0., 1., cv2.NORM_MINMAX)
        gt = cv2.normalize(gt, gt, 0., 1., cv2.NORM_MINMAX)
        img1 = cv2.normalize(img1, img1, 0., 1., cv2.NORM_MINMAX)
        # input(type(img0))
        # return img0, gt, img1
            
    def __getitem__(self, index):        
        img0, gt, img1 = self.getimg(index)
        print("index:", index)
        if self.dataset_name == 'train':
            img0, gt, img1 = self.aug(img0, gt, img1, 224, 224)
            # print("img0:", img0.shape)
            # if random.uniform(0, 1) < 0.5:
            #     img0 = img0[:, :, ::-1]
            #     img1 = img1[:, :, ::-1]
            #     gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp
        
        # print("Data is in range %f to %f dB" % (np.min(img0), np.max(img0))) #

        # img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        # print("img0:", img0.shape)
        # img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        # gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        img0 = img0[np.newaxis, ...]
        img0 = torch.from_numpy(img0.copy())
        img1 = img1[np.newaxis, ...]
        img1 = torch.from_numpy(img1.copy())
        gt = gt[np.newaxis, ...]
        gt = torch.from_numpy(gt.copy())
        return torch.cat((img0, img1, gt), 0)
