import os
import json
import heapq
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk

import torch
import torchvision.models as models

import SimpleITK as sitk
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.DPCA import Get_Local_Density,Get_Each_Center_Distance,Choose_Cluster_Centers


class log_writer():
    def __init__(self, txt_name = ""):
        super().__init__()
        self.name = txt_name
    def write(self,info):
        writer = open(self.name,"a+")
        data = info + "\n"
        writer.writelines(data)
        print(info)
        writer.close()

# 2D function
def find_centroids(result, valid_thresh = 3.0, cut_distance = 7.0, density = 10.0, distance = 7.0):
    centroids = []
    result = np.array(result)

    index_pts = (result > valid_thresh)
    coord = np.array(np.nonzero((index_pts == 1)))
    points_num = coord.shape[1]

    coord_dis_row = np.repeat(coord[:, np.newaxis, :], points_num, axis = 1)
    coord_dis_col = np.repeat(coord[:, :, np.newaxis], points_num, axis = 2)
    points_distance = np.sqrt(np.sum((coord_dis_col - coord_dis_row) ** 2, axis=0))

    points_data = np.where(result>valid_thresh)
    x,y = points_data
    points_data = list(zip(x,y))

    points_density = Get_Local_Density(points_num, points_distance, cut_distance)
    center_distance = Get_Each_Center_Distance(points_num, points_distance, points_density)
    labels, center_index = Choose_Cluster_Centers(points_num, points_density, center_distance,density, distance)
    for i in range(len(center_index)):
        index = center_index[i]
        centroids.append(points_data[index])
    return centroids



def read_drr_nii(nii):
    nii = sitk.GetArrayFromImage(sitk.ReadImage(nii))
    nii = ((nii/255)  - 0.456)/0.224
    nii = nii[np.newaxis,:,:]
    nii = np.concatenate((nii,nii,nii),axis=0)
    nii = nii[np.newaxis,:,:]
    return nii

# 3D function
import numpy as np
import torch
import numpy as np
def intersect(P0,P1):
    P0 = np.array(P0)
    P1 = np.array(P1)
    n = (P1)/np.linalg.norm(P1,axis=1)[:,np.newaxis] # normalized
    projs = np.eye(n.shape[1]) - n[:,:,np.newaxis]*n[:,np.newaxis]  # I - n*n.T

    R = projs.sum(axis=0)
    q = (projs @ P0[:,:,np.newaxis]).sum(axis=0)
    p = np.linalg.lstsq(R,q,rcond=None)[0]
    return p

def get_two_matrix(txt_path):
    txt_file = open(txt_path, 'r')
    txt_data = txt_file.readlines()
    Extrinsic = []
    Intrinsic = []
    for i,data in enumerate(txt_data):
        tmp = data.split("   ")
        if(i>7 and i<12):
            row = []
            # print(i,tmp)
            tmp.remove("")
            for d in tmp:
                row.append(eval(d))
            Extrinsic.append(row)
        if(i>12 and i<16):
            row = []
            # print(i,tmp)
            tmp.remove("")
            for d in tmp:
                row.append(eval(d))
            Intrinsic.append(row)
    return torch.Tensor(Extrinsic), torch.Tensor(Intrinsic)

def get_rays(H, W, c2i, w2c):
    i, j = torch.meshgrid(torch.linspace(0, H - 1, H).to(w2c.device),
                          torch.linspace(0, W - 1, W).to(w2c.device))
    i = i.t()
    j = j.t()

    cx, cy = (H - 1) / 2, (W - 1) / 2

    i2c = torch.linalg.pinv(c2i)

    c2w = torch.linalg.inv(w2c)
    ori = c2w @ torch.Tensor([0, 0, 0, 1]).to(w2c.device)
    ori = ori[:3]

    coords = torch.stack([i - cx,
                          j - cy,
                          torch.ones_like(i)], dim=1)
    rays_d = i2c @ coords
    rays_d = c2w @ rays_d
    rays_d = torch.moveaxis(rays_d, 1, -1)
    rays_d = rays_d[:, :, :3]
    rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)

    rays_o = ori.expand(rays_d.shape)
    return rays_o.numpy(), rays_d.numpy()

def read_json(json_path):
    with open(json_path) as json_data:
        data = json.load(json_data)
        json_data.close()
    return data



def read_model(params_path):
    model = None
    if params_path:
        model = models.segmentation.fcn_resnet50(num_classes=25, weights = None)
        pretrained_dict = torch.load(params_path)['net'] #logs/on_enhance_8/new_coding/fcnresnet50_with_pretrain/10epoch_40_with_pretrain_backbone_39.pth
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.train()
    return model

def infer_voting(base_path, drr_path, params_path, mode, save_path, save_name, 
                  valid_thresh, cut_distance, density, distance,
                  n_views = 10, if_read_infer=False, if_read_gt=False ):
    txt_file = log_writer(save_path+"results.txt")
    
    model = read_model(params_path)
    model.cuda()

    matrix_name_list = ['0000.txt', '0001.txt', '0002.txt', '0003.txt', '0004.txt', '0005.txt', '0006.txt', '0007.txt', '0008.txt', '0009.txt',
                        '0010.txt', '0011.txt', '0012.txt', '0013.txt', '0014.txt', '0015.txt', '0016.txt', '0017.txt', '0018.txt', '0019.txt']

    drr_list = os.listdir(drr_path)
    drr_list.sort()

    ct_path = base_path + "enhance_ct/"
    ct_list = os.listdir(ct_path)
    ct_list.sort()
    ct_size = []
    for ct_file in ct_list:
        ct_file_path = os.path.join(ct_path, ct_file)
        ct = sitk.ReadImage(ct_file_path)
        ct_size.append(ct.GetSize())
    
    if if_read_gt:
        infer_centroids_2d = read_json(save_path+mode + "_2d_centroids_gt_"+ str(n_views) +"views.json")
    else:
        if if_read_infer: 
            infer_centroids_2d = read_json(save_path+mode+"_2d_centroids.json")

    r = 5
    class_num = 25
    pred_all = {}
    for j in tqdm(range(len(drr_list))):
        drr_folder = drr_list[j]
        isocenter_file = open(os.path.join(drr_path, drr_folder, "isocenter.txt"), 'r')
        isocenter = isocenter_file.readline().split(",")
        new_isocenter = []
        for a in isocenter:
            a = str(a)
            a = eval(a)
            new_isocenter.append(a)
        isocenter = new_isocenter

        
        drr_folder_path = os.path.join(drr_path, drr_folder)
        
        drr_nii_path = os.path.join(drr_folder_path, "nii")
        niis = os.listdir(drr_nii_path)
        niis.sort()
        if not if_read_gt and not if_read_infer:
            infer_nii_path = os.path.join(drr_folder_path, "infer")
            infer_niis = os.listdir(infer_nii_path)
            infer_niis.sort()

        centroids_predictions_2d = []
        centroids_predictions_2d_len = []
        # localization on 10 views drr
        for i in range(len(niis)):
            if if_read_infer:
                centroids_prediction = infer_centroids_2d[drr_folder][i]
            else:
                infer_path = os.path.join(infer_nii_path, infer_niis[i])
                infer_nii = sitk.GetArrayFromImage(sitk.ReadImage(infer_path))
                centroids_prediction = find_centroids(infer_nii, valid_thresh, cut_distance, density, distance)
            centroids_predictions_2d.append(centroids_prediction)
            centroids_predictions_2d_len.append(len(centroids_prediction))
        tmp = np.bincount(centroids_predictions_2d_len)[1:]
        prediction_len = 0
        for num,e in enumerate(tmp):
            if e == np.max(tmp):
                prediction_len = num + 1
        valid_view_num = 0
        record = {}
        # id each vert in 10 drr images
        tmp_prob = []
        sequence_loss = []
        for i in range(len(niis)):
            valid_view_num += 1
            nii_path = os.path.join(drr_nii_path, niis[i])
            img = read_drr_nii(nii_path)
            img = torch.FloatTensor(img)
            img = img.cuda()
            with torch.no_grad():
                output = model(img)['out']
            output = output.squeeze()
            output = torch.softmax(output, dim=0)
            for ii in range(centroids_predictions_2d_len[i]):
                target = torch.zeros(img.shape[-2:])
                x,y = centroids_predictions_2d[i][ii]
                x += 1
                y += 1
                target[(x-r):(x+r+1), (y-r):(y+r+1)] = 1
                index = (target==1)
                for iii in range(output.shape[0]):
                    if iii == 0:
                        tmp = output[iii][index]
                        tmp = tmp.unsqueeze(0)
                    else:
                        tmp = torch.cat((tmp, output[iii][index].unsqueeze(0)), dim=0)
                if ii == 0:
                    probability = torch.mean(tmp, dim=1)
                    probability = probability.unsqueeze(0)
                else:
                    probability = torch.cat((probability, torch.mean(tmp, dim=1).unsqueeze(0)), dim = 0)
            tmp_prob.append(probability)
            sequence_loss_opt = probability
            for p in range(centroids_predictions_2d_len[i]):
                for q in range(class_num):
                    if p > 0 and q > 1:
                        sequence_loss_opt[p][q] = max(sequence_loss_opt[p-1][q-2]*0.1 , sequence_loss_opt[p-1][q-1]*0.8 , sequence_loss_opt[p-1][q]*0.1 ) + sequence_loss_opt[p][q]
            sequence_loss.append(torch.max(sequence_loss_opt).item()/((prediction_len)*0.8))
        # use sequence loss to find topk and use them as weights:
        a = []
        valid_prob = []
        for i in range(len(niis)):
            if(centroids_predictions_2d_len[i] == prediction_len): # matched drr
                a.append(sequence_loss[i])
                valid_prob.append(tmp_prob[i])
        a = np.array(a)
        topk = len(a)

        topk_index = heapq.nlargest(topk, range(len(a)), a.__getitem__)
        sum_all = 0
        for top in range(topk):
            sum_all += sequence_loss[topk_index[top]]

        final_prob = valid_prob[topk_index[0]]*a[topk_index[0]]/sum_all
        for num,index in enumerate(topk_index):
            if num == 0:
                final_prob = valid_prob[index]*a[index]/sum_all
            else:
                final_prob += valid_prob[index]*a[index]/sum_all

        probability = final_prob

        start = 0
        for p in range(prediction_len):
            for q in range(class_num):
                if p > 0 and q > 0:
                    probability[p][q] = max(probability[p-1][q-2]*0.1 , probability[p-1][q-1]*0.8 , probability[p-1][q]*0.1 ) + probability[p][q]
        start = probability[-1].argmax()-prediction_len+2
        if start<=0:
            start = 1
        pred_list = [i for i in range(start, start + prediction_len)]


        # mapping
        Origin,direction = [],[]
        for i in range(n_views):
            data = matrix_name_list[i]
            data_path = os.path.join(drr_folder_path, data)
            Extrinsic,Intrinsic = get_two_matrix(data_path)
            tmp_o,tmp_d = get_rays(1024,1024,Intrinsic,Extrinsic)
            o,d = [], []
            for ii in range(len(centroids_predictions_2d[i])):
                x,y = centroids_predictions_2d[i][ii]
                x += 1
                y += 1
                o.append(tmp_o[x,y])
                d.append(tmp_d[x,y])
            Origin.append(o)
            direction.append(d)
        centroids_3d = []
        for i in range(prediction_len):
            p0,p1 = [], []
            for ii in range(len(niis)):
                if(centroids_predictions_2d_len[ii] == prediction_len): # matched drr
                    p0.append(Origin[ii][i])
                    p1.append(direction[ii][i])
            p0 = np.array(p0)
            p1 = np.array(p1)
            point = intersect(p0,p1)
            centroids_3d.append((point.T - np.array(isocenter) + np.array(ct_size[j])/2)[0].tolist())
        pred_all[drr_folder] = list(zip(pred_list, centroids_3d))

    save_json_file = save_path + "/" + save_name + "_" + mode+"_voting.json"
    with open(save_json_file,"w") as f:
        json.dump(pred_all,f)
    print(save_json_file + " saved.")

    with open(save_path + mode + "_all.json") as json_data:
        gt_dic_list = json.load(json_data)
        json_data.close()
    with open(save_json_file) as json_data:
        pred_dic_list = json.load(json_data)
        json_data.close()

    distance_error = 0
    count_all, count_right = 0,0
    count = 0
    miss_vert_count = {}
    for k in gt_dic_list.keys():
        pred_centroids = []
        pred_ids = []
        gt_centroids = []
        gt_ids = []
        for i in range(len(gt_dic_list[k])):
            gt_ids.append(gt_dic_list[k][i][0])
            gt_centroids.append(gt_dic_list[k][i][1])
        for i in range(len(pred_dic_list[k])):
            pred_ids.append(pred_dic_list[k][i][0])
            pred_centroids.append(pred_dic_list[k][i][1])
        pred_len = len(pred_dic_list[k])
        gt_len = len(gt_dic_list[k])
        if gt_len - pred_len > 0:
            miss_vert_count[k] = gt_len - pred_len
        distance_matrix = np.zeros((pred_len, gt_len))
        for i in range(pred_len):
            for j in range(gt_len):
                distance_matrix[i][j] = np.sqrt(np.sum(np.square(np.array(pred_centroids[i])+1 - np.array(gt_centroids[j]))))
        index = distance_matrix.argmin(axis=1)
        for i in range(len(index)):
            if distance_matrix[i][index[i]] < 20 :
                distance_error += distance_matrix[i][index[i]]
                count_all += 1
                count_right += (pred_ids[i]==gt_ids[index[i]])
                if pred_ids[i]!=gt_ids[index[i]]:
                    print(f'{k} pred is {pred_ids[i]}, gt is {gt_ids[index[i]]}')
            else:
                count += 1
                count_all += 1
    
    for i,k in enumerate(miss_vert_count.keys()):
        if miss_vert_count[k] > 0:
            print(k,miss_vert_count[k])
    txt_file.write(f'[*] {mode.zfill(10)}_voting: id rate is {100*count_right/(count_all):.2f}%, distance error is {distance_error/count_all:.4f} + {abs(sum(miss_vert_count.values())*1000)/(count_all):.4f} = {(distance_error + abs(sum(miss_vert_count.values())*1000))/(count_all):.4f} mm, over dis num: {count}, mis num {sum(miss_vert_count.values())}.')




