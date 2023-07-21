import os
import math
import glob
import numpy as np
import SimpleITK as sitk


import sys
sys.path.append(".")
from dataset.drr_dataset import get_json
def create_Gaussian(Gaussian_size = 7, sigma = 3):
    sum = 0
    kernel = np.zeros((Gaussian_size,Gaussian_size))
    k = int(Gaussian_size / 2) + 1
    for x in range(Gaussian_size):
        for y in range(Gaussian_size):
            kernel[x,y] = (1/(2*math.pi*(sigma**2)))*np.exp((-(x-k)**2 - (y-k)**2) / (2 * sigma))
            sum += kernel[x,y]
    for x in range(Gaussian_size):
        for y in range(Gaussian_size):
            kernel[x,y] = kernel[x,y]/sum 
    k1 = 100/np.max(kernel)
    # print(k)
    for x in range(Gaussian_size):
        for y in range(Gaussian_size):
            kernel[x,y] = kernel[x,y] * k1
    return kernel

def create_heatmap(heatmap_size ,centroids, Gaussian_size = 7, sigma = 3):
    heatmap_x,heatmap_y = heatmap_size
    kernel = create_Gaussian(Gaussian_size, sigma)
    heatmap = np.zeros((heatmap_x,heatmap_y))
    for centroid in centroids:
        # get the label xyz
        y,x = centroid
        x,y = int(x), int(y)
        x1,x2 = int(x-Gaussian_size/2),int(x+Gaussian_size/2)
        y1,y2 = int(y-Gaussian_size/2),int(y+Gaussian_size/2)
        if(heatmap[x1:x2, y1:y2].shape != (Gaussian_size,Gaussian_size)):
            print("kernel is from:")
            print(str(x1) + " to " + str(x2) + "," + str(heatmap_x))
            print(str(y1) + " to " + str(y2) + "," + str(heatmap_y))
            print(heatmap[x1-1:x2, y1-1:y2].shape)
        heatmap[x1:x2, y1:y2] += kernel
    return heatmap

if __name__=="__main__":
    data_path = "./Data/VerSe/VerSe19/verse19train/"
    json_path = data_path + "json"
    json_all = os.listdir(json_path)
    ct_base_path = data_path + "raw_ct/rawdata/"
    ct_list = os.listdir(ct_base_path)
    ct_list.sort()
    drr_path = data_path + "enhance_ct_3/enhance_drr/"
    drr_folders = os.listdir(drr_path)
    drr_folders.sort()
    for n, drr_folder in enumerate(drr_folders):
        # 1
        base_path = os.path.join(drr_path, drr_folder)
        isocenter_path = os.path.join(base_path, "isocenter.txt")
        isocenter_file = open(isocenter_path, 'r')
        isocenter = isocenter_file.readline().split(",")

        new_isocenter = []
        for a in isocenter:
            a = str(a)
            a = eval(a)
            new_isocenter.append(a)
        isocenter = new_isocenter
        print(isocenter)

        ct_path = os.path.join(ct_base_path,ct_list[n])
        ct_size = sitk.ReadImage(ct_path).GetSize()
        ct_center = [int(ct_size[0]/2) + 1, int(ct_size[1]/2) + 1, int(ct_size[2]/2) + 1]
        print(ct_center)
        #check

        # 2
        matrix_all = []
        matrix_list = glob.glob(os.path.join(base_path,'*.txt'))
        matrix_list.sort()
        matrix_list.remove(isocenter_path)
        id_path = os.path.join(base_path, "ids.txt")
        if id_path in matrix_list:
            matrix_list.remove(id_path)
        for matrix_file_name in matrix_list:
            print(matrix_file_name)
            matrix = []
            matrix_file = open(matrix_file_name)
            matrix_data = matrix_file.readlines()
            for i,data in enumerate(matrix_data):
                if(i > 0 and i < 4):
                    sign = []
                    # 符号位：3，22，41，60
                    if(data[3]=="-"):
                        sign.append(-1*eval(data[4:18]))
                    else:
                        sign.append(eval(data[4:18]))
                    if(data[22]=="-"):
                        sign.append(-1*eval(data[23:37]))
                    else:
                        sign.append(data[23:37])
                    if(data[41]=="-"):
                        sign.append(-1*eval(data[42:56]))
                    else:
                        sign.append(eval(data[42:56]))
                    if(data[60]=="-"):
                        sign.append(-1*eval(data[61:75]))
                    else:
                        sign.append(eval(data[61:75]))
                    # print(sign)
                    matrix.append(sign)
            matrix_all.append(matrix)
        # check

        # 3
        json_file = get_json(os.path.join(json_path,json_all[n]))
        centroids = []
        ids = [] 
        for json_data in json_file:
            if(isinstance(json_data[0], int)):
                centroids.append(json_data)
                ids.append(json_data[0])
        # print(centroids)
        id_file = open(os.path.join(base_path, 'ids.txt'), 'w')
        id_file.writelines(str(ids))
        id_file.close()
        # check

        # 4
        centroids_drr_all = []
        for p,matrix in enumerate(matrix_all):
            centroids_drr = []
            tmp = isocenter[:]
            tmp.append(1)
            isocenter_3d = []
            isocenter_3d.append(tmp)
            # print(isocenter_3d)
            isocenter_3d = np.array(isocenter_3d,dtype=np.float32).T
            matrix = np.array(matrix,dtype=np.float32)
            output = (matrix @ isocenter_3d)
            # print(matrix.shape, isocenter_3d.shape,output.shape)
            isocenter_2d = (output/output[2]).T[0][:-1]
            # print(isocenter_2d)
            for j in range(len(centroids)):
                tmp = centroids[j][1:] # ct coordinates
                tmp = np.array(tmp) - np.array(ct_center) + np.array(isocenter)
                tmp = list(tmp)
                tmp.append(1)
                centroid_3d = []
                centroid_3d.append(tmp)
                centroid_3d = np.array(centroid_3d).T
                output_centroid = (matrix @ centroid_3d)
                centroid_2d = (output_centroid/output_centroid[2]).T[0][:-1]
                dis = centroid_2d - isocenter_2d
                drr_coor = [768+dis[0], 768+dis[1]]
                centroids_drr.append(drr_coor)
                # print(f'drr_{p+1} vert_{j+1} coor is {drr_coor}, 2d coor is {centroid_2d}')
            centroids_drr_all.append(centroids_drr)
            # break
        #check
        drr_path_nii = os.path.join(base_path, "nii")
        # print(drr_path_nii)
        drr_nii_all = os.listdir(drr_path_nii)
        for m,centroids_drr in enumerate(centroids_drr_all):
            txt_name = os.path.join(base_path, "heatmap",drr_folder+ "_" + str(m) + ".txt")
            txt_file = open(txt_name, 'w')
            for c in centroids_drr:
                txt_file.write(str(c)+"/n")
            txt_file.close()
            target_base_path = os.path.join("./VerSe/VerSe19/verse19train/raw_ct/drr//",drr_folder,"heatmap") 
            os.makedirs(target_base_path, exist_ok=True)
            save_name = os.path.join(target_base_path, drr_folder+ "_" + str(m) + ".txt")
            os.rename(txt_name, save_name)
            print(f'{save_name} done.')
            # drr = sitk.ReadImage(os.path.join(drr_path_nii,drr_nii_all[m]))
            # direction = drr.GetDirection()
            # spacing = drr.GetSpacing() 
            # origin = drr.GetOrigin()

            # heatmap = create_heatmap([1024,1024],centroids_drr,11,7)
            # saveImg = sitk.GetImageFromArray(heatmap)
            # saveImg.SetOrigin(origin)
            # saveImg.SetSpacing(spacing)
            # saveImg.SetDirection(direction)
            # heatmap_path = os.path.join(base_path, "heatmap")
            # os.makedirs(heatmap_path, exist_ok=True)
            # save_name = os.path.join(heatmap_path,drr_folder + "_" + str(m) + ".nii.gz")
            # print(m,drr_folder,save_name,centroids_drr)
            # sitk.WriteImage(saveImg,save_name)
            # break