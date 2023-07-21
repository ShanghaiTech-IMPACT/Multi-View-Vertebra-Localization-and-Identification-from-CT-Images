# -*- coding: utf-8 -*-
# @Author: wuhan
# @Date:   2022-10-14 18:49:59
# @Last Modified by:   wuhan
# @Last Modified time: 2023-02-24 13:40:35

import os 
import json
import torch
import numpy as np
import SimpleITK as sitk
import albumentations as A
from genericpath import exists
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as dataset

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

class id_as_seg_dataset(dataset):
    def __init__(self, drr_path=" ", mode="train", data_mode = "training", if_deformation = False, transform = None):
        super().__init__()
        self.drop_list = ["drr_03"]
        self.drr_path = drr_path
        self.mode = mode
        self.data_mode = data_mode
        self.if_deformation = if_deformation
        self.deformation_count = 0
        self.read_list()
        self.data = self.load_list() # READ DATA
        self.transform = transform
        print("deformation is ", self.if_deformation)
        print(f"{self.mode}_data length is : {len(self.data)}")
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        r = 5
        img = sitk.GetArrayFromImage(sitk.ReadImage(self.data[index][0]))
        
        centroids_file = open(self.data[index][1],'r')
        centroids = centroids_file.readlines()
        ids = self.data[index][2]
        ids = ids.split("[")[1]
        ids = ids.split("]")[0]
        ids = ids.split(", ")
        ids = np.array(ids, dtype=int)
        label = np.zeros(img.shape)

        prob = np.random.random()
        if self.if_deformation and prob < 0.5:
            self.deformation_count += 1
            random_state = np.random.RandomState(33)
            alpha=30
            sigma=5
            shape = img.shape
            dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

            x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))#, np.arange(shape[2]))
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))#, np.reshape(z, (-1, 1))
            distored_image = map_coordinates(img, indices, order=1, mode='reflect')
            img = distored_image.reshape(shape)
            for i in range(len(centroids)):
                data = centroids[i].split("[")[1]
                data = data.split("]")[0]
                x,y = data.split(", ")
                x,y = eval(x), eval(y)
                x += dx[int(x)][int(y)]
                y += dy[int(x)][int(y)]
                x,y = int(y), int(x)
                label[(x-r):(x+r+1), (y-r):(y+r+1)] = ids[i] + 1
        else:
            for i in range(len(centroids)):
                data = centroids[i].split("[")[1]
                data = data.split("]")[0]
                x,y = data.split(", ")
                y,x = int(eval(x)), int(eval(y))
                label[(x-r):(x+r+1), (y-r):(y+r+1)] = ids[i] + 1
        if self.transform != None:
            img = ((img/255)  - 0.456)/0.224
            aug = self.transform(image=img,mask=label)
            img = aug['image']
            label = aug['mask']
        else:
            img = ((img/255)  - 0.456)/0.224
        img = img[np.newaxis,:,:]
        img = np.concatenate((img,img,img),axis=0)
        img = torch.FloatTensor(img)
        label = torch.Tensor(label) 
        if index == 0:
            self.deformation_count = 1
        if index == len(self.data) - 1:
            tqdm.write(f'deformation number is {self.deformation_count}')

        return img,label

    def read_list(self):
        ids_all = []
        nii_all =[]
        label_all = []
        txt_name = str(self.mode) + "_id_as_seg.txt"
        if (exists(txt_name)):
            print(f"{txt_name} exists")
            os.remove(txt_name)
        
        folder_list = os.listdir(self.drr_path)
        folder_list.sort()
        for folder in folder_list:
            if self.mode == "train" and folder in self.drop_list:
                continue
            if(folder[:3] != "drr"):
                continue
            txt_file_path = os.path.join(self.drr_path,folder, "ids.txt")
            ids_file = open(txt_file_path, 'r')
            ids = ids_file.readline()
            ids = ids.split("[")[1]
            ids = ids.split("]")[0]
            ids = ids.split(", ")
            for i,id in enumerate(ids):
                ids[i] = eval(id) - 2
            ids_all.append(ids)
            folder_path = os.path.join(self.drr_path,folder,"nii")
            niis = os.listdir(folder_path)
            count = 0
            for nii in niis:
                nii_path = os.path.join(folder_path, nii)
                nii_all.append(nii_path)
                count += 1
            file_all = os.path.join(self.drr_path,folder, "heatmap")
            file_list = os.listdir(file_all)
            file_list.sort()
            for file in file_list:
                if (file[-3:] == "txt"):
                    file_path = os.path.join(file_all, file)
                    label_all.append(file_path)

        txt_file = open(txt_name,'w')
        for i in range(len(nii_all)): 
            txt_file.writelines(nii_all[i] + "*" + str(label_all[i]) + "*" + str(ids_all[i//10]) + "\n")
        txt_file.close()
        print(f'[*] {txt_file} generated!')

    def load_list(self):
        file_name_list = []
        txt_name = self.mode + "_id_as_seg.txt"
        with open(txt_name, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.split("*"))
        return file_name_list 




