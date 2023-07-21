import json
import os 
import torch
import numpy as np
import SimpleITK as sitk
from genericpath import exists
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as dataset

import albumentations as A

def get_json(ctd_path):
    with open(ctd_path) as json_data:
        dict_list = json.load(json_data)
        json_data.close()
    ctd_list = []
    for d in dict_list:
        if 'direction' in d:
            ctd_list.append(tuple(d['direction']))
        elif 'nan' in str(d):            # skipping NaN centroids
            continue
        else:
            ctd_list.append([d['label'], d['X'], d['Y'], d['Z']])
    return ctd_list



class drr_dataset(dataset):
    def __init__(self, drr_path=" ", mode="train", if_identification=False, if_aug=False, n_views=10):
        super().__init__()
        self.drop_list = []#["drr_03"]
        self.drr_path = drr_path
        self.mode = mode
        self.n_views = n_views
        self.if_identification = if_identification
        self.if_aug = if_aug
        print(f'aug is {if_aug}')
        self.read_list()
        self.data = self.load_list() # READ DATA
        print(f"{self.mode}_data length is : {len(self.data)}")


        self.transform = None
        self.transform = A.Compose([
                            A.HorizontalFlip(p=0.2)])

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        img = sitk.ReadImage(self.data[index][0])
        heatmap = sitk.ReadImage(self.data[index][1])        
        ids = self.data[index][2]

        img = sitk.GetArrayFromImage(img)
        heatmap = sitk.GetArrayFromImage(heatmap)
        img = ((img/255) - 0.456)/0.224
        
        if self.transform != None and self.if_aug==True:
            transformed  = self.transform(image = img, mask = heatmap)
            img = transformed["image"]
            heatmap = transformed["mask"]

        img = img[np.newaxis,:,:]
        heatmap = heatmap[np.newaxis,:,:]
        img = np.concatenate((img,img,img),axis=0)

        img = torch.FloatTensor(img)
        heatmap = torch.FloatTensor(heatmap)
        if(self.if_identification):
            return img, heatmap,ids
        else:
            return img, heatmap

    def read_list(self):
        drr_all = []
        heatmap_all = []
        label_file_all = []
        id_all = []
        txt_name = "DRR_localization_" + str(self.mode) + ".txt"
        if (exists(txt_name)):
            print(f"{txt_name} exists")
            return 0
            # os.remove(txt_name)
        
        folder_list = os.listdir(self.drr_path)
        folder_list.sort()
        for folder in folder_list:
            if self.mode == "train" and folder in self.drop_list:
                continue
            folder_path = os.path.join(self.drr_path,folder)
    
            heatmap_folder = os.path.join(folder_path,"heatmap")
            heatmap_files = os.listdir(heatmap_folder)
            heatmap_files.sort()
            for heatmap_file in heatmap_files:
                if heatmap_file[-3:] == ".gz":
                    data = os.path.join(heatmap_folder, heatmap_file)
                    heatmap_all.append(data)

            drr_folder = os.path.join(folder_path,"nii")
            drr_files = os.listdir(drr_folder)
            drr_files.sort()
            for drr_file in drr_files:
                if drr_file[-3:] == ".gz":
                    # print(data)
                    data = os.path.join(drr_folder, drr_file)
                    drr_all.append(data)

            ids = open(os.path.join(folder_path, "ids.txt"),'r')
            data = ids.readline()[1:-1]
            ids.close()
            id_all.append(data)


        txt_file = open(txt_name,'w')
        print(len(drr_all),len(heatmap_all),np.array(id_all).shape)
        for i in range(len(drr_all)): 
            id_data = i//self.n_views
            # print(id_data)s
            txt_file.writelines(drr_all[i] + "*" + heatmap_all[i] + "*" + str(id_all[id_data]) + "\n")
        txt_file.close()
        print(f'[*] {txt_file} generated!')

    def load_list(self):
        file_name_list = []
        txt_name = "DRR_localization_" + self.mode + ".txt"
        with open(txt_name, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()
                if not lines:
                    break
                file_name_list.append(lines.split("*"))
        return file_name_list 



