import SimpleITK as sitk
from pathlib import Path
import numpy as np
import nibabel as nib
import nibabel.processing as nip
import nibabel.orientations as nio
import os
import json

import sys
sys.path.append('../')
# from data.ct_transform_drr import *


def ct_reorient(img, axcodes_to=('L', 'P', 'S')):
    """Reorients the nifti from its original orientation to another specified orientation

    Parameters:
    ----------
    img: nibabel image
    axcodes_to: a tuple of 3 characters specifying the desired orientation

    Returns:
    ----------
    newimg: The reoriented nibabel image

    """
    aff = img.affine
    arr = np.asanyarray(img.dataobj, dtype=img.dataobj.dtype)
    ornt_fr = nio.io_orientation(aff)
    ornt_to = nio.axcodes2ornt(axcodes_to)
    ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)
    arr = nio.apply_orientation(arr, ornt_trans)
    aff_trans = nio.inv_ornt_aff(ornt_trans, arr.shape)
    newaff = np.matmul(aff, aff_trans)
    newimg = nib.Nifti1Image(arr, newaff)
    print("[*] Image reoriented from", nio.ornt2axcodes(ornt_fr), "to", axcodes_to)
    return newimg

def ct_resample(img, voxel_spacing=(1, 1, 1), order=3):
    """Resamples the nifti from its original spacing to another specified spacing

    Parameters:
    ----------
    img: nibabel image
    voxel_spacing: a tuple of 3 integers specifying the desired new spacing
    order: the order of interpolation

    Returns:
    ----------
    new_img: The resampled nibabel image

    """
    # resample to new voxel spacing based on the current x-y-z-orientation
    aff = img.affine
    shp = img.shape
    zms = img.header.get_zooms()
    # Calculate new shape
    new_shp = tuple(np.rint([
        shp[0] * zms[0] / voxel_spacing[0],
        shp[1] * zms[1] / voxel_spacing[1],
        shp[2] * zms[2] / voxel_spacing[2]
    ]).astype(int))
    new_aff = nib.affines.rescale_affine(aff, shp, voxel_spacing, new_shp)
    new_img = nip.resample_from_to(img, (new_shp, new_aff), order=order, cval=-1024)
    print("[*] Image resampled to voxel size:", voxel_spacing)
    return new_img

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

def centroids_to_dict(ctd_list):
    """Converts the centroid list to a dictionary of centroids

    Parameters:
    ----------
    ctd_list: the centroid list

    Returns:
    ----------
    dict_list: a dictionart of centroids having the format dict[vertebra] = ['X':x, 'Y':y, 'Z': z]

    """
    dict_list = []
    for v in ctd_list:
        if any('nan' in str(v_item) for v_item in v): continue  # skipping invalid NaN values
        v_dict = {}
        if isinstance(v, tuple):
            v_dict['direction'] = v
        else:
            v_dict['label'] = int(v[0])
            v_dict['X'] = v[1]
            v_dict['Y'] = v[2]
            v_dict['Z'] = v[3]
        dict_list.append(v_dict)
    return dict_list


def centroids_save(ctd_list, out_path):
    """Saves the centroid list to json file

    Parameters:
    ----------
    ctd_list: the centroid list
    out_path: the full desired save path

    """
    if len(ctd_list) < 2:
        print("[#] Centroids empty, not saved:", out_path)
        return
    json_object = centroids_to_dict(ctd_list)

    # Problem with python 3 and int64 serialisation.
    def convert(o):
        if isinstance(o, np.int64):
            return int(o)
        raise TypeError

    with open(out_path, 'w') as f:
        json.dump(json_object, f, default=convert)
    print("[*] Centroids saved:", out_path)


def centroids_rescale(ctd_list, img, voxel_spacing=(1, 1, 1)):
    """rescale centroid coordinates to new spacing in current x-y-z-orientation

    Parameters:
    ----------
    ctd_list: list of centroids
    img: nibabel image
    voxel_spacing: desired spacing

    Returns:
    ----------
    out_list: rescaled list of centroids

    """
    ornt_img = nio.io_orientation(img.affine)
    ornt_ctd = nio.axcodes2ornt(ctd_list[0])
    if np.array_equal(ornt_img, ornt_ctd):
        zms = img.header.get_zooms()
    else:
        ornt_trans = nio.ornt_transform(ornt_img, ornt_ctd)
        aff_trans = nio.inv_ornt_aff(ornt_trans, img.dataobj.shape)
        new_aff = np.matmul(img.affine, aff_trans)
        zms = nib.affines.voxel_sizes(new_aff)
    ctd_arr = np.transpose(np.asarray(ctd_list[1:]))
    v_list = ctd_arr[0].astype(int).tolist()  # vertebral labels
    ctd_arr = ctd_arr[1:]
    ctd_arr[0] = np.around(ctd_arr[0] * zms[0] / voxel_spacing[0], decimals=1)
    ctd_arr[1] = np.around(ctd_arr[1] * zms[1] / voxel_spacing[1], decimals=1)
    ctd_arr[2] = np.around(ctd_arr[2] * zms[2] / voxel_spacing[2], decimals=1)
    out_list = [ctd_list[0]]
    ctd_list = np.transpose(ctd_arr).tolist()
    for v, ctd in zip(v_list, ctd_list):
        out_list.append([v] + ctd)
    print("[*] Rescaled centroid coordinates to spacing (x, y, z) =", voxel_spacing, "mm")
    return out_list

def centroids_reorient(ctd_list, img, decimals=1):
    """reorient centroids to image orientation

    Parameters:
    ----------
    ctd_list: list of centroids
    img: nibabel image
    decimals: rounding decimal digits

    Returns:
    ----------
    out_list: reoriented list of centroids

    """
    ctd_arr = np.transpose(np.asarray(ctd_list[1:]))
    if len(ctd_arr) == 0:
        print("[#] No centroids present")
        return ctd_list
    v_list = ctd_arr[0].astype(int).tolist()  # vertebral labels
    ctd_arr = ctd_arr[1:]
    ornt_fr = nio.axcodes2ornt(ctd_list[0])  # original centroid orientation
    axcodes_to = nio.aff2axcodes(img.affine)
    ornt_to = nio.axcodes2ornt(axcodes_to)
    trans = nio.ornt_transform(ornt_fr, ornt_to).astype(int)
    perm = trans[:, 0].tolist()
    shp = np.asarray(img.dataobj.shape)
    ctd_arr[perm] = ctd_arr.copy()
    for ax in trans:
        if ax[1] == -1:
            size = shp[ax[0]]
            ctd_arr[ax[0]] = np.around(size - ctd_arr[ax[0]], decimals)
    out_list = [axcodes_to]
    ctd_list = np.transpose(ctd_arr).tolist()
    for v, ctd in zip(v_list, ctd_list):
        out_list.append([v] + ctd)
    print("[*] Centroids reoriented from", nio.ornt2axcodes(ornt_fr), "to", axcodes_to)
    return out_list


if __name__ == "__main__":
    mode_all = ["train", "test","validation"] #    "train", "test",
    for mode in mode_all:
        base_path = "D:/Data/VerSe/VerSe19/verse19" + mode + "/"
        ct_path = base_path + "raw_ct/rawdata/"
        mask_path = base_path + "derivatives/"
        json_path = base_path + "json/"
        cts = os.listdir(ct_path)
        cts.sort()
        masks = os.listdir(mask_path)
        masks.sort()
        jsons = os.listdir(json_path)
        jsons.sort()
        for i,ct in enumerate(cts):
            centroids = []
            if(ct == ".DS_Store"):
                continue
            print(f'---------- {i} {ct} ----------')
            # CT
            ct_name = os.path.join(ct_path,ct)
            ct_data = nib.load(ct_name)
            ct_res = ct_resample(ct_data)
            ct_reo = ct_reorient(ct_res)
            # nib.Nifti1Image(img,img_affine).to_filename(‘xxxxx.nii.gz’)
            nib.save(ct_reo, ct_name)
            # Json
            json_name = os.path.join(json_path,ct[:-7]+".json")
            json_data = get_json(json_name)
            json_res = centroids_rescale(json_data, ct_data)
            json_reo = centroids_reorient(json_res, ct_reo)
            centroids_save(json_reo, json_name)
        print("mask transform start : ")
        for mask in masks:
            if(mask == ".DS_Store"):
                continue
            print(f'----------{mask} ----------')
            mask_name = os.path.join(mask_path, mask)
            mask = nib.load(mask_name)
            mask_res = ct_resample(mask,(1,1,1),0)
            mask_reo = ct_reorient(mask_res)
            nib.save(mask_reo, mask_name)