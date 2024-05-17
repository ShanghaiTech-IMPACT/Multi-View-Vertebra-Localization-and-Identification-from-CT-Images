import os
import cv2
import glob
import json
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

def plastimatch_drr(input_file, output_dir):
    # parameters
    N_views = 10
    sad = 500
    sid = 1000

    image = sitk.ReadImage(input_file)
    image_arr = sitk.GetArrayFromImage(image)

    volume_resolution = np.array(image.GetSize()) #CT volume像素尺寸 单位pixel
    volume_spacing = np.array(image.GetSpacing()) #CT volume spacing 单位mm/pixel
    volume_origin = np.array(image.GetOrigin())
    volume_phy = volume_spacing * (volume_resolution) #CT volume物理尺寸 单位mm

    proj_phy = volume_phy * sid / sad #projection的物理尺寸
    proj_phy = proj_phy[-2:]
    proj_spacing = volume_spacing * sid / sad #projection spacing 单位mm/pixel
    proj_spacing = proj_spacing[-2:]
    proj_resolution = np.round(proj_phy / proj_spacing).astype(int)  #CT volume投影的像素尺寸 可以认为=volume_resolution

    proj_spacing = proj_phy / proj_resolution # 微小改动spacing

    isocenter = volume_origin + volume_phy / 2

    params = {
        # 'angle_per_view': angle_per_view,
        'N_views': N_views,
        'sad': sad,
        'sid': sid,
        'volume_resolution': volume_resolution.tolist(),
        'volume_spacing': volume_spacing.tolist(),
        'volume_origin': volume_origin.tolist(),
        'volume_phy': volume_phy.tolist(),
        'proj_resolution': [1024,1024],#proj_resolution.tolist(),
        'proj_spacing': [2,2],#proj_spacing.tolist(),
        'proj_phy': [2048,2048],#proj_phy.tolist(),
        'isocenter': isocenter.tolist(),
    }
    command = f'plastimatch drr -a {params["N_views"]} --sad {params["sad"]} --sid {params["sid"]}  -r "{params["proj_resolution"][0]} {params["proj_resolution"][1]}" -z "{params["proj_phy"][0]} {params["proj_phy"][1]}" -o "{params["isocenter"][0]} {params["isocenter"][1]} {params["isocenter"][2]}" -I {input_file} -O {output_dir} -t raw --autoscale --autoscale-range "0 255"'
    print(command)
    os.system(f'{command}')
    params['command'] = command
    print('Generating nii files...')
    raw_files = glob.glob(os.path.join(output_dir, '*.raw'))
    nii_path = os.path.join(output_dir, 'nii')
    png_path = os.path.join(output_dir, 'png')
    os.makedirs(nii_path, exist_ok=True)
    os.makedirs(png_path, exist_ok=True)
    for file in tqdm(raw_files):

        raw2nii_png(src=file, dst_nii=os.path.join(nii_path, os.path.basename(file).split('.')[0] + '.nii.gz'),
                dst_png=os.path.join(png_path, os.path.basename(file).split('.')[0] + '.png'),
                H=params["proj_resolution"][0], W=params["proj_resolution"][1], phyH=params["proj_phy"][0], phyW=params["proj_phy"][1])

    isocenter_file = open(os.path.join(output_dir, 'isocenter.txt'), 'w')
    isocenter_file.writelines(str(params['isocenter'])[1:-1])
    isocenter_file.close()
    with open(os.path.join(output_dir, 'transforms.json'), 'w') as f:
        json.dump(params, f, indent=4)

def raw2nii_png(src, dst_nii, dst_png, H, W, phyH, phyW):
    raw_arr = np.fromfile(src, dtype='float32')
    raw_arr = raw_arr.reshape(W, H)
    cv2.imwrite(dst_png,raw_arr)
    image = sitk.GetImageFromArray(raw_arr)
    spacingH, spacingW = phyH / H, phyW / W
    image.SetSpacing([spacingH, spacingW, 1])
    sitk.WriteImage(image, dst_nii)


if __name__=="__main__":
    mode_all = ["train"] #  
    for mode in mode_all:
        base_path = "D:/Data/VerSe/VerSe19/verse19" + mode + "/enhance_ct_8/"

        air_ct_path = os.path.join(base_path, "ct")
        air_ct_list = os.listdir(air_ct_path)
        air_ct_list.sort()

        for i,air_ct in enumerate(air_ct_list):
            ct_path = os.path.join(air_ct_path, air_ct)
            save_path = os.path.join(base_path,"enhance_drr","drr_"+air_ct[-9:-7]+"/")
            os.makedirs(save_path, exist_ok=True)
            plastimatch_drr(ct_path, save_path)
            break
    