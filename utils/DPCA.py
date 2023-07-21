# -*- coding: utf-8 -*-
# @Author: wuhan
# @Date:   2022-03-08 17:41:26
# @Last Modified by:   wuhan
# @Last Modified time: 2022-04-13 15:38:34

import matplotlib.pyplot as plt

import sys
import numpy as np
import random as rd
import SimpleITK as sitk
# 计算各点之间的距离
def Get_Distance(points_data, points_num):
    points_list = points_data
    points_distance = np.zeros((points_num, points_num))
    for i in range(points_num):
        for j in range(points_num):
            x_t = points_list[i][0] - points_list[j][0]
            y_t = points_list[i][1] - points_list[j][1]
            # z_t = points_list[i][2] - points_list[j][2]
            points_distance[i][j] = np.sqrt(np.square(x_t) + np.square(y_t)) #+ np.square(z_t))
    return points_distance


# 计算各点的局部密度
def Get_Local_Density(points_num, points_distance, cut_distance, if_exp = True):
    points_density = np.zeros(points_num)
    for i in range(points_num):
        density = 0
        # 使用 Gaussian kernel 连续型密度
        for j in range(points_num):
            if 0 < points_distance[i][j] < cut_distance:
                if if_exp:
                    density += np.exp(-np.square(points_distance[i][j] / cut_distance))
                else:
                    density += 1
        points_density[i] = density
    # 返回时转化为 list
    return points_density.tolist()


# 计算各点的聚类中心距离
def Get_Each_Center_Distance(points_num, points_distance, points_density):
    # 初始化聚类中心距离列表
    center_distance = []

    # 找到每个密度大于自身的点, 计算距离并保存索引
    for i in range(points_num):
        higher_index = []
        for j in range(points_num):
            if points_density[i] < points_density[j]:
                higher_index.append(j)

        # 如果 higher_index 非空, 说明当前点不是最大局部密度点
        if higher_index:
            each_center_distance = Get_Min_Center_Distance(higher_index, points_distance, i)
            center_distance.append(each_center_distance)
        # 如果 higher_index 为空, 说明当前点是最大局部密度点
        else:
            each_center_distance = Get_Max_Center_Distance(points_distance, points_density, i)
            center_distance.append(each_center_distance)
    return center_distance


# 对于非最大局部密度点 x
def Get_Min_Center_Distance(higher_index, points_distance, i):
    # 在密度大于 x 的数据点中, 找到与 x 最小的距离, 作为 x 的聚类中心距离
    distance = []
    for j in higher_index:
        distance.append(points_distance[i][j])
    return min(distance)


# 对于最大局部密度点 x
def Get_Max_Center_Distance(points_distance, points_density, i):
    # 在其他所有数据点中, 找到与 x 最大的距离, 作为 x 的聚类中心距离
    max_density = max(points_density)
    return max(points_distance[i])


# 计算截断距离 dc, 参数 t 范围 (0, 1)
def Get_CutOff_Distance(points_num, points_distance, t = 0.02):
    # 获取所有非零距离
    distance = []
    for i in range(points_num):
        for j in range(i + 1):
            if points_distance[i][j] > 0:
                distance.append(points_distance[i][j])

    # 对距离列表升序排列, 选择 t * 100% 的距离作为 dc
    distance.sort()
    # 返回第 M * t (四舍五入) 段距离, 其中 M = 1/2 * n * (n - 1)
    index = round(t * 0.5 * points_num * (points_num - 1))
    return distance[index]


# 确定聚类中心点
def Choose_Cluster_Centers(points_num, points_density, center_distance, density=90, distance=13):
    # 算法不足: 中心点根据决策图人为选取, 主观性过强, 程序无法自动进行
    # 手动调参方式: 选取密度大于总体 rate_1, 中心距离大于总体 rate_2 的点作为中心点
    # points_density_bound = rate_1 * (max(points_density) - min(points_density)) + min(points_density)
    # center_distance_bound = rate_2 * (max(center_distance) - min(center_distance)) + min(center_distance)
    # 构造聚类标签
    labels = [0] * points_num
    center_index = []
    count = 0
    # 对中心点进行标记
    for i in range(points_num):
        if points_density[i] > density and center_distance[i] > distance:
            count += 1
            labels[i] = count
            center_index.append(i)
    return labels, center_index


# 根据聚类中心点进行密度聚类
def Clustering(points_num, points_distance, points_density, labels):

    # 由高密度点到低密度点依次进行
    points_density_sorted = sorted(points_density, reverse = True)

    # 对每个点而言, 寻找比自己密度高的点中的最近点
    for i in range(points_num):
        density_1 = points_density_sorted[i]
        index_1 = points_density.index(density_1)

        # 如果目标点还未归类为某一簇
        if labels[index_1] == 0:
            min_distance = sys.maxsize
            nearest_index = 0

            for j in range(i):
                density_2 = points_density_sorted[j]
                index_2 = points_density.index(density_2)

                # 寻找距离最近的高密度点
                if 0 < points_distance[index_1][index_2] < min_distance:
                    min_distance = points_distance[index_1][index_2]
                    nearest_index = index_2

            # 归属于距离最近的高密度点所在的簇
            if nearest_index > 0:
                labels[index_1] = labels[nearest_index]

    return labels


# 读取文件数据 (txt 文件)
def ReadFile(path):
    # 文件中的每一行数据: 横坐标 x, 纵坐标 y, 样例聚类编号 i
    data = np.loadtxt(path, dtype = float, delimiter = ',')

    # 将坐标存入 points_data 矩阵
    points_data = data[:, 0:2]
    # 将样例聚类编号存入 example_labels 数组
    example_labels = data[:, 2]
    return points_data, example_labels


# 生成随机颜色
def RandColor():
    color_num = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color_code = ""
    for i in range(6):
        color_code += color_num[rd.randint(0, 14)]
    return "#" + color_code


# 绘制可视化图像
def Draw_Figure(points_num, points_data, points_density, center_distance, labels, center_index):
    # # 正常显示中文标签
    # matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    # # 正常显示负号
    # matplotlib.rcParams['axes.unicode_minus'] = False

    # 随机调色
    colors = []
    for i in range(len(center_index) + 1):
        rand_color = RandColor()
        colors.append(rand_color)

    color_labels = []
    for i in range(points_num):
        index = labels[i]
        color_labels.append(colors[index])

    # # 图1: 数据点分布图
    # fig1 = plt.figure(figsize = (8, 6))
    # plt.scatter(points_data[:, 0], points_data[:, 1], c = color_labels)
    # for i in center_index:
    #     plt.text(points_data[:, 0][i], points_data[:, 1][i], labels[i])
    # plt.title(u'数据点分布图')
    # plt.xlabel('x');  plt.ylabel('y')
    # plt.savefig(r'数据点分布图.png')
    # plt.show()

    # 图2: 决策图
    fig2 = plt.figure(figsize = (8, 6))
    plt.scatter(points_density, center_distance, c = color_labels)
    for i in center_index:
        plt.text(points_density[i], center_distance[i], labels[i])
    plt.title(u'graph')
    plt.xlabel("density");  plt.ylabel("distance")
    # plt.savefig(r'决策图')
    plt.show()


# # 参数设置
# if __name__ == '__main__':
#     sigmoid = True
#     if sigmoid:
#         output_path = "data_sample/visualization/sigmoid/output.nii.gz"
#         output_ct = sitk.ReadImage(output_path)
#         output_array = sitk.GetArrayFromImage(output_ct)
#         x,y,z = output_array.shape
#         count = 0
#         coordinates = []
#         for i in range(x):
#             for j in range(y):
#                 for m in range(z):
#                     if(output_array[i][j][m]==1):
#                         count+=1
#                         coordinates.append([i,j,m])
#         print(f'point num is {x*y*z} and valid point num is {count}')
        
#         points_data = coordinates
#         points_num = np.shape(points_data)[0]
#         print(points_num)
#         points_distance = Get_Distance(points_data, points_num)
#         print(points_distance.shape)
        
#         cut_distance = Get_CutOff_Distance(points_num, points_distance, t=0.1)
#         print(cut_distance)
        
#         cut_distance = 7
#         rate_1, rate_2 = 0.01, 0.01
#         points_density = Get_Local_Density(points_num, points_distance, cut_distance)
#         center_distance = Get_Each_Center_Distance(points_num, points_distance, points_density)
#         labels, center_index = Choose_Cluster_Centers(points_num, points_density, center_distance, rate_1, rate_2)
#         print("centroid data is :")
#         for index in center_index:
#             print(coordinates[index])
        
#         labels = Clustering(points_num, points_distance, points_density, labels)
#         Draw_Figure(points_num, points_data, points_density, center_distance, labels, center_index)
#     else:
#         output_path = "data_sample/visualization/no_sigmoid/output.nii.gz"
#         output_ct = sitk.ReadImage(output_path)
#         output_array = sitk.GetArrayFromImage(output_ct)
#         x,y,z = output_array.shape
#         count = 0
#         coordinates = []
#         for i in range(x):
#             for j in range(y):
#                 for m in range(z):
#                     if(output_array[i][j][m]>10):
#                         count+=1
#                         coordinates.append([i,j,m])
#         print(f'point num is {x*y*z} and valid point num is {count}')
        
#         points_data = coordinates
#         points_num = np.shape(points_data)[0]
#         print(points_num)
#         points_distance = Get_Distance(points_data, points_num)
#         print(points_distance.shape)
        
        # cut_distance = Get_CutOff_Distance(points_num, points_distance, t=0.1)
        # print(cut_distance)
        
        # cut_distance = 7
        # rate_1, rate_2 = 0.01, 0.01
        # points_density = Get_Local_Density(points_num, points_distance, cut_distance)
        # center_distance = Get_Each_Center_Distance(points_num, points_distance, points_density)
        # labels, center_index = Choose_Cluster_Centers(points_num, points_density, center_distance, rate_1, rate_2)
        # print("centroid data is :")
        # for index in center_index:
        #     print(coordinates[index])
        
        # labels = Clustering(points_num, points_distance, points_density, labels)
        # Draw_Figure(points_num, points_data, points_density, center_distance, labels, center_index)
    



