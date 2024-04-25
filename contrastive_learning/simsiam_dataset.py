
import os 
import numpy as np
from tqdm import tqdm

from PIL import Image
import SimpleITK as sitk
from genericpath import exists
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as dataset
import torchvision.transforms as T
# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image, ImageFilter

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class drr_dataset(dataset):
    def __init__(self, drr_path=" ", mode="train", mean_std=[[0.456,0.456,0.456], [0.224,0.224,0.224]]):
        super().__init__()
        self.drop_list = ["drr_03"]
        self.drr_path = drr_path
        self.mode = mode
        self.transform = T.Compose([
            T.RandomResizedCrop(224, scale=(0.2, 1.)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
        self.read_list()
        self.data = self.load_list() # READ DATA
        print(f"{self.mode}_data length is : {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def centercrop(self, img, crop_size=256):
        x,y = img.shape
        center_x = int(x/2)
        center_y = int(y/2)
        width = int(crop_size/2)
        return img[center_x-width:center_x+width,center_y-width:center_y+width]
    
    def __getitem__(self, index):
        img = sitk.ReadImage(self.data[index][0])
        # id = np.array(eval(self.data[index][1]),dtype=int)
        # id = np.reshape(id,(1,1))

        img = sitk.GetArrayFromImage(img)
        img = self.centercrop(img)
    

        img = img[:,:,np.newaxis]
        img = np.concatenate((img,img,img),axis=-1)
        img = Image.fromarray(np.uint8(img))
        aug_1 = self.transform(img)
        aug_2 = self.transform(img)

        return aug_1,aug_2

    def read_list(self):
        drr_all = []
        txt_name = str(self.mode) + ".txt"
        if (exists(txt_name)):
            print(f"{txt_name} exists")
            # return 0
            os.remove(txt_name)
        folder_list = os.listdir(self.drr_path)
        folder_list.sort()
        for folder in folder_list:
            if self.mode == "train" and folder in self.drop_list:
                continue
            folder_path = os.path.join(self.drr_path,folder)
            drr_folder = os.path.join(folder_path,"nii")
            drr_files = os.listdir(drr_folder)
            for drr_file in drr_files:
                data = os.path.join(drr_folder, drr_file)
                drr_all.append(data)

        txt_file = open(txt_name,'w')
        
        for i in range(len(drr_all)): 
            id_data = i//10
            txt_file.writelines(drr_all[i] + "*" + str(id_data) + "\n")
        txt_file.close()
        print(f'[*] {txt_file} generated!')

    def load_list(self):
        file_name_list = []
        txt_name = self.mode + ".txt"
        with open(txt_name, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()
                if not lines:
                    break
                file_name_list.append(lines.split("*"))
        return file_name_list 


class multi_view_dataset(dataset):
    def __init__(self, drr_path=" ", mode="train", mean_std=[[0.456,0.456,0.456], [0.224,0.224,0.224]], n_views=10):
        super().__init__()
        self.drop_list = ["drr_03"]
        self.drr_path = drr_path
        self.mode = mode
        self.n_views = n_views
        self.normalize =  T.Compose([T.ToTensor(), T.Normalize(*mean_std)])
        self.transform = T.Compose([
            T.RandomResizedCrop(224, scale=(0.2, 1.)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
        self.read_list()
        self.data = self.load_list() # READ DATA
        print(f"{self.mode}_data length is : {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def centercrop(self, img, crop_size=256):
        x,y = img.shape
        center_x = int(x/2)
        center_y = int(y/2)
        width = int(crop_size/2)
        return img[center_x-width:center_x+width,center_y-width:center_y+width]
    
    def __getitem__(self, index):
        drr_path = os.path.join(self.data[index][0],"nii")
        drr_files = os.listdir(drr_path)

        index_1 = int(eval(self.data[index][1]))
        index_2 = np.random.randint(0,self.n_views,1)[0]
        while index_1 == index_2:
            index_2 = np.random.randint(0,self.n_views,1)[0]
        # print(self.n_views,index_1,index_2)
        img_1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(drr_path,drr_files[index_1])))
        img_2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(drr_path,drr_files[index_2])))

        img_1 = self.centercrop(img_1)
        img_2 = self.centercrop(img_2)
    
        img_1 = img_1[:,:,np.newaxis]
        img_2 = img_2[:,:,np.newaxis]

        img_1 = np.concatenate((img_1,img_1,img_1),axis=-1)
        img_2 = np.concatenate((img_2,img_2,img_2),axis=-1)

        img_1 = Image.fromarray(np.uint8(img_1))
        img_2 = Image.fromarray(np.uint8(img_2))
        
        aug_1_1 = self.transform(img_1)
        aug_1_2 = self.transform(img_1)
        aug_2_1 = self.transform(img_2)
        aug_2_2 = self.transform(img_2)
        img_1 = self.normalize(img_1)
        img_2 = self.normalize(img_2)

        return img_1, img_2, aug_1_1, aug_1_2, aug_2_1, aug_2_2

    def read_list(self):
        drr_all = []
        txt_name = str(self.mode) + "_multi_view.txt"
        if (exists(txt_name)):
            print(f"{txt_name} exists")
            return 0
            os.remove(txt_name)
        folder_list = os.listdir(self.drr_path)
        folder_list.sort()
        for folder in folder_list:
            # if self.mode == "train" and folder in self.drop_list:
            #     continue
            folder_path = os.path.join(self.drr_path,folder)
            drr_folder = os.path.join(folder_path,"nii")
            drr_files = os.listdir(drr_folder)
            for drr_file in drr_files:
                drr_all.append(folder_path)
        txt_file = open(txt_name,'w')
        
        for i in range(len(drr_all)): 
            id_data = i%self.n_views
            txt_file.writelines(str(drr_all[i]) + "*" + str(id_data) + "\n")
        txt_file.close()
        print(f'[*] {txt_file} generated!')

    def load_list(self):
        file_name_list = []
        txt_name = str(self.mode) + "_multi_view.txt"
        with open(txt_name, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.split("*"))
        return file_name_list 


if __name__=="__main__":
    mean_std = [[0.456,0.456,0.456], [0.224,0.224,0.224]]
    mydata = multi_view_dataset(drr_path="/public_bme/data/wuhan/VerSe19/verse19train/enhance_ct_8/10views/enhance_drr_2mm", mode="train", mean_std=mean_std)
    train_dl = DataLoader(mydata, batch_size=1)
    for i, (img) in enumerate(train_dl):
        # print(img.shape)
        img_1, img_2, aug_1_1, aug_1_2, aug_2_1, aug_2_2 = img
        print(len(img),img_1.shape)
        
        break
