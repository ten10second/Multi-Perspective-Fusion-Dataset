
import cv2 as cv
from PIL import Image
import numpy as np
from pathlib import Path
import os
import re 
import shutil
import argparse
import numpy
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from lib.tracking_utils.utils import remove_files

# parser = argparse.ArgumentParser()
# parser.add_argument('--view', default='bev')
# args = parser.parse_args()

def extract_int_from_string(string):
    number = ""
    for i in string:
        if i.isdigit():
            number += i
    
    return str(number)


# frame_pic_path = args.view + '_objective'
# id_frame_path = 'inference' + '/' + args.view + '_id_frame'
# gallary_path = 'inference/' + args.view + '_gallary'

##############  03.21 frame_pic --> id_frame ############

def making_the_bev_id_frame_folder(perspective, video_index):
    
    frame_pic_path = '../' + perspective + '_objective'
    id_frame_path =  '../' + 'inference' + '/' + perspective + '_id_frame'

    bev_in_frame_subfolder = os.path.join(id_frame_path, 'video_'+ video_index)  # inference/fpv_id_frame/video_1
    os.mkdir(bev_in_frame_subfolder) if not os.path.isdir(bev_in_frame_subfolder) else remove_files(bev_in_frame_subfolder)

    id_list = []

    for path, dirs, files in os.walk(frame_pic_path):

        id_list.extend(files)
        
    id_list = list(set(id_list))


    # 判断 id_frame_path文件夹是否为空 如果有历史文件 先清空 
    # if os.path.exists(id_frame_path):
    #     print('start clean the historical files in the id_frame folder')
    #     for subfolder in os.listdir(id_frame_path):
    #         subfolder_path = os.path.join(id_frame_path, subfolder)
    #         shutil.rmtree(subfolder_path) 
    # print('Cleaned historical files')   

    print('Making new files in the id_frame folder')
    for ped_id in id_list:

        jpg_path = frame_pic_path + '/' + ped_id    # random path 
        ped_id = Path(jpg_path).stem   # return string ped_number
        ped_id_path = bev_in_frame_subfolder + '/'  + ped_id
        os.mkdir(ped_id_path)


    for frame in os.listdir(frame_pic_path):
        
        for id_jpg in id_list:

            source_jpg_path = Path(frame_pic_path + '/' + frame + '/' + id_jpg)

            # print(source_jpg_path)
            ped_id = Path(source_jpg_path).stem    # return string ped_number

            if source_jpg_path.exists():

                source_image = cv.imread(str(source_jpg_path))
                image_size = source_image.shape
                image_w = image_size[1]
                image_h = image_size[0]

                if image_h * image_w > 6000 and (0.3 < image_w / image_h < 0.7):
                                
                    target_jpg_path =  Path(bev_in_frame_subfolder + '/' + ped_id + '/' + frame + '.jpg')
                    shutil.copy2(source_jpg_path, target_jpg_path)

    print('STEP1: Generated the ' + perspective + '_id_frame folder')


def making_the_fpv_id_frame_folder(perspective, video_index):
    
    frame_pic_path = '../' + perspective + '_objective'
    id_frame_path =  '../' + 'inference' + '/' + perspective + '_id_frame'   # inference/fpv_id_frame
    
    fpv_in_frame_subfolder = os.path.join(id_frame_path, 'video_'+video_index)  # inference/fpv_id_frame/video_1
    os.mkdir(fpv_in_frame_subfolder) if not os.path.isdir(fpv_in_frame_subfolder) else remove_files(fpv_in_frame_subfolder)

    id_list = []

    for path, dirs, files in os.walk(frame_pic_path):

        id_list.extend(files)
        
    id_list = list(set(id_list))
    print(id_list)

    # make target file

    # 判断 id_frame_path文件夹是否为空 如果有历史文件 先清空 ( FPV 视角不需要动态清空fpv_id_frame)
    # if os.path.exists(id_frame_path):
    #     print('start clean the historical files in the id_frame folder')
    #     for subfolder in os.listdir(id_frame_path):
    #         subfolder_path = os.path.join(id_frame_path, subfolder)
    #         shutil.rmtree(subfolder_path) 
    # print('Cleaned historical files')   

    print('Making new files in the id_frame folder')
    for ped_id in id_list:

        jpg_path = frame_pic_path + '/' + ped_id    # random path   
        ped_id = Path(jpg_path).stem   # return string ped_number
        ped_id_path =  fpv_in_frame_subfolder + '/'  + ped_id
        os.mkdir(ped_id_path)

    for frame in os.listdir(frame_pic_path):
        
        for id_jpg in id_list:
            source_jpg_path = Path(frame_pic_path + '/' + frame + '/' + id_jpg)

            # print(source_jpg_path)
            ped_id = Path(source_jpg_path).stem    # return string ped_number

            if source_jpg_path.exists():

                source_image = cv.imread(str(source_jpg_path))
                image_size = source_image.shape
                image_w = image_size[1]
                image_h = image_size[0]

                if image_h * image_w > 5000 and (0.3 < image_w / image_h < 0.7) :
                                
                    target_jpg_path =  Path(fpv_in_frame_subfolder + '/' + ped_id + '/' + frame + '.jpg')
                    shutil.copy2(source_jpg_path, target_jpg_path)

    print('STEP1: Generated the ' + perspective + '_id_frame folder')


####################### re-name the image 04.03  ####################

def rename(perspective, video_index):

    # perspective = 'bev' 
    # video_index = '1', '2','3' ...
    id_frame_path =  '../' + 'inference' + '/' + perspective + '_id_frame'
        
    video_folder = id_frame_path + '/' + 'video_' + video_index
    for ped_file in os.listdir(video_folder):
        ped_file_path = video_folder + '/' + ped_file   # inference/fpv_id_frame/video_1/ped3
        ped_id_number = extract_int_from_string(ped_file)  # ped3 --> 3

        for imagename in os.listdir(ped_file_path):  # 
            frame_number = extract_int_from_string(imagename)                       
            if imagename.endswith('.jpg'):
                if perspective == 'fpv':
                    new_filename = ped_id_number.zfill(4) + '_' + 'c2s' + video_index + '_' + frame_number.zfill(6) + '_' + '01' + '.jpg'
                # print(os.path.join(folder_path, new_filename))
                if perspective == 'bev':
                    new_filename = ped_id_number.zfill(4) + '_' + 'c1s' + video_index + '_' + frame_number.zfill(6) + '_' + '01' + '.jpg'
    
                os.rename(os.path.join(ped_file_path, imagename), os.path.join(ped_file_path, new_filename))

    print('STEP2: Finished Rename')


######################## 04.01 make the bev_gallary file ###############

def make_bev_gallary_file(perspective, video_index):
    
    frame_pic_path = '../' + perspective + '_objective'
    id_frame_path = '../' + 'inference' + '/' + perspective + '_id_frame'    # inference/bev_id_frame
    gallary_path = '../' + 'inference/' + perspective + '_gallary'          # inference/bev_gallary

    subfolder_bev_id_frame = id_frame_path + '/' + 'video_' + video_index   # inference/bev_id_frame/video_1

    subfolder_in_bev_gallary = os.path.join(gallary_path, 'video_'+ video_index)  # inference/bev_gallary/video_1
    os.mkdir(subfolder_in_bev_gallary) if not os.path.isdir(subfolder_in_bev_gallary) else remove_files(subfolder_in_bev_gallary)

    # gallary中的文件不用清空 每个片段连续处理后集中储存在gallary文件夹
    # if os.path.exists(gallary_path):
    #     print('start clean the historical files in the gallary')
    #     for subfolder in os.listdir(gallary_path):
    #         subfolder_path = os.path.join(gallary_path, subfolder)
    #         os.remove(subfolder_path)

    # print('Cleaned historical gallary folder') 

    for ped_id in os.listdir(subfolder_bev_id_frame):
        ped_file_path = subfolder_bev_id_frame + '/' + ped_id  #  inference/bev_id_frame/video_1/ped1
        
        picked_size = 0
        bev_gallary_img_source = None
        bev_gallary_img_path = subfolder_in_bev_gallary + '/'
        
        if len(os.listdir(ped_file_path)) != 0:
            for img_name in os.listdir(ped_file_path):
                img_path = ped_file_path + '/' + img_name   # inference/bev_id_frame/video_1/ped1/xxx.jpg
                
                source_image = cv.imread(img_path)
                image_size = source_image.shape
                
                if image_size[0] * image_size[1] > picked_size:
                    picked_size = image_size[0] * image_size[1]
                    bev_gallary_img_source = img_path
                    bev_img_name = img_name
        
            shutil.copy2(bev_gallary_img_source, bev_gallary_img_path + bev_img_name)

    print('STEP3: Make the ' + perspective + '_gallery')


### 05.26 confirm the center_person_id 
def confirm_center_person(video_index):

    # 筛选出现帧数最多的id 
    
    bev_id_frame = '../' + 'inference' +'/' + 'bev_id_frame'
    subfolder_bev_id_frame = bev_id_frame + '/' + 'viedo_' + video_index
    
    center_ped = ' '
    max_frame = 0

    for ped_folder in os.listdir(subfolder_bev_id_frame):
        ped_folder_path = os.path.join(subfolder_bev_id_frame, ped_folder)   # inference/bev_id_frame/ped2
        ped = ped_folder_path.split('/')[-1]   # ped2 
        frame_num =  len([f for f in os.listdir(ped_folder_path)if os.path.isfile(os.path.join(ped_folder_path, f))])
        if frame_num > max_frame:
            max_frame = frame_num
            center_ped = ped

    return center_ped

############## 04.01 make the fpv_gallary file ###############

def make_fpv_gallary_file(perspective, video_index):

    frame_pic_path = '../' + perspective + '_objective'
    id_frame_path = '../' + 'inference' + '/' + perspective + '_id_frame' 
    gallary_path = '../' + 'inference/' + perspective + '_gallary'

    subfolder_fpv_id_frame = id_frame_path + '/' + 'video_' + video_index 
    subfolder_in_fpv_gallary = os.path.join(gallary_path, 'video_'+ video_index)  # inference/bev_gallary/video_1
    os.mkdir(subfolder_in_fpv_gallary) if not os.path.isdir(subfolder_in_fpv_gallary) else remove_files(subfolder_in_fpv_gallary)


    # 清空fpv_gallary 文件夹 
    # if os.path.exists(gallary_path):
    #     print('start clean the historical files in the gallary')
    #     for fpv_image in os.listdir(gallary_path):
    #         fpv_image_path = os.path.join(gallary_path, fpv_image)
    #         os.remove(fpv_image_path)

    # print('Cleaned historical gallary folder') 

    for ped_id in os.listdir(subfolder_fpv_id_frame):
        ped_file_path = subfolder_fpv_id_frame + '/' + ped_id

        gallary_img_path = subfolder_in_fpv_gallary + '/'
        frame_list = []

        if len(os.listdir(ped_file_path)) != 0:
            for img_name in os.listdir(ped_file_path):
                frame_list.append(img_name)

            if len(frame_list) > 2:
                gallary_img = frame_list[int( len(frame_list) / 2)]          
                fpv_gallary_img_source = ped_file_path + '/' + gallary_img

                shutil.copy2(fpv_gallary_img_source, gallary_img_path + gallary_img)            

    print('STEP3: Make the ' + perspective + '_gallery')

#  统计文件夹内所有文件的数量(单层文件夹)   
def counting_file(folder_path):
    
    for root,dirs,files in os.walk(folder_path):
        count = len(files)

    return count

def get_intrinsic_parameters(chess_folder_path):

    images = []
    for img_name in os.listdir(chess_folder_path):
        images.append(os.path.join(chess_folder_path, img_name))
    # set the dimension of the chessboard
    chessboardDimension = (8, 6)

    # 创建空列表存储像素坐标和世界坐标
    imagePointsArray = [] # 2d points in image plane.
    objectPointsArray = [] # 3d points in real world space.

    # prepare object points 
    objectPoints = np.zeros((chessboardDimension[0] * chessboardDimension[1],3), np.float32)
    objectPoints[:,:2] = np.mgrid[0:chessboardDimension[0], 0:chessboardDimension[1]].T.reshape(-1,2)

    # 对每一张图像进行处理
    for imageName in images:

        img = cv.imread(imageName)
        grayImage = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img_size = grayImage.shape[::-1]
        
        # 检测棋盘格图像上的角点，并获取它们的像素坐标
        ret,corners = cv.findChessboardCorners(grayImage,chessboardDimension,None)
        if ret == True:
            imagePointsArray.append(corners)
            objectPointsArray.append(objectPoints)

    ret,camera_matrix,distortion_coefficients,rvecs,tvecs = cv.calibrateCamera(objectPointsArray,imagePointsArray,img_size,None,None)

    return ret,camera_matrix,distortion_coefficients

def get_points_array(d):
    with open(d, 'r') as f:
        next(f)
        points_arr = [list(map(int, l.rstrip().split())) for l in f]
        points_arr = np.array(points_arr[:4], dtype=np.float64)
        
    pixel_points = points_arr[:, 3:]
    world_points = points_arr[:, :3]
    
    pixel_points = pixel_points[:, np.newaxis, :]
    world_points = world_points[:, np.newaxis, :]

    return pixel_points, world_points

def get_extrinsic_parameters(width, height, mtx, dist, calibration_text):
    
    pixel_points, world_points = get_points_array(calibration_text)
    # retval, rvec, tvec = cv.solvePnP(world_points, pixel_points, mtx, dist, None, None, True,
    #                                   cv.SOLVEPNP_DLS)
    _, rvec, tvec, inliers = cv.solvePnPRansac(world_points, pixel_points, mtx, dist)

    R,_ = cv.Rodrigues(rvec)
    
    return  rvec, tvec, R

def pixel_to_world(camera_intrinsics, r, t, img_points):

    K_inv = np.mat(camera_intrinsics).I
    R_inv = np.asmatrix(r).I
    R_inv_T = np.dot(R_inv, np.asmatrix(t))
    world_points = []
    coords = np.zeros((3, 1), dtype=np.float64)
    for img_point in img_points:
        coords[0] = img_point[0]
        coords[1] = img_point[1]
        coords[2] = 1.0
        cam_point = np.dot(K_inv, coords)
        cam_R_inv = np.dot(R_inv, cam_point)
        scale = R_inv_T[2][0] / cam_R_inv[2][0]
        scale_world = np.multiply(scale, cam_R_inv)
        world_point = np.asmatrix(scale_world) - np.asmatrix(R_inv_T)
        pt = np.zeros((3, 1), dtype=np.float64)
        pt[0] = world_point[0]
        pt[1] = world_point[1]
        pt[2] = 0
        world_points.append(pt.T.tolist())

    return world_points


def smooth(x, y):
    # input(x, window, k)

    window_length = len(x) if len(x) % 2 == 1 else len(x) - 1
    x_hat = savgol_filter(x, window_length, 3)
    y_hat = savgol_filter(y, window_length, 3)

    return x_hat, y_hat

##############  03.21 drop ############ 
# 以下100行代码是真的是脑子不太行的产物（两个月后0515有感

# 计算ped文件夹中图像的相似度 drop相似度低的图像()

# def img_correlation(path1, path2):

#     img1 = cv.imread(path1,0)
#     img2 = cv.imread(path2,0)

#     hist1 = cv.calcHist([img1],[0],None,[256],[0,256])
#     hist2 = cv.calcHist([img2],[0],None,[256],[0,256])

#     cv.normalize(hist1,hist1,0,255,cv.NORM_MINMAX)
#     cv.normalize(hist2,hist2,0,255,cv.NORM_MINMAX)

#     correl = cv.compareHist(hist1,hist2,cv.HISTCMP_CORREL)

#     return correl   # 返回两张图像的相似度值


# waiting_pool_file = 'inference/output/waiting_pool'
# threshold = 0.9 
# for ped in os.listdir(id_frame_path):

#     ped_in_frame_path = id_frame_path + '/' + ped
#     path_list = os.listdir(ped_in_frame_path)
#     path_list = sorted(path_list, key=lambda x: int(re.search(r'\d+', x).group()))

#     correl_list = [] 
#     for i in range(len(path_list)-1):
        
#         frame1 = path_list[i]
#         frame2 = path_list[i+1]
            
#         path1 = ped_in_frame_path + '/' + frame1 
#         path2 = ped_in_frame_path + '/' + frame2

#         correl = img_correlation(path1, path2)

#         # 判断相似度的下降情况, 相似度下降过高 drop

#         if len(correl_list) > 0:
#             last_correl = correl_list[-1]

#             if (last_correl - correl) / last_correl > 0.1:
#                 # planA
#                 # break
                
#                 # planB 将匹配度不高的ped-frame 放入waiting pool中，等待进行再次匹配
#                 frame_waiting_list = path_list[i+1:]

#                 for waiting_frame in frame_waiting_list:

#                     source_pic_path = ped_in_frame_path + '/' + waiting_frame
#                     target_pic_path = waiting_pool_file + '/' + 'waiting' + ped + '_' + waiting_frame
#                     shutil.copy2(source_pic_path, target_pic_path)
#                     os.remove(source_pic_path)  # 删掉原文件

#         correl_list.append(correl)
            
##############  03.21  对waiting_pool中照片与id_frame中的行人进行匹配 ############
### to-do 匹配率太低的话直接新建一个ped文件夹##################

# for waiting_ped in os.listdir(waiting_pool_file):
#     waiting_ped_path = waiting_pool_file + '/' + waiting_ped

#     file_name = Path(waiting_ped_path).stem
#     frame = file_name.split('_')[-1]
#     frame = re.findall('\d+', frame)[0]

#     match_score = 0

#     for ped in os.listdir(id_frame_path):

#         ped_refe_img_file = id_frame_path + '/' + ped
#         match_score_list = []

#         for reference_img in os.listdir(ped_refe_img_file):
            
#             reference_img_path = ped_refe_img_file + '/' + reference_img
    
#             if img_correlation(waiting_ped_path, reference_img_path) > 0.8:   # 阈值设置为0.8
#                 match_score_list.append(img_correlation(waiting_ped_path,reference_img_path))
            
#         ped_match_score = np.average(match_score_list)
#         if ped_match_score > match_score:
#             match_score = max(match_score, ped_match_score)
#             match_ped = ped

#     if match_score != 0:
#         waiting_target_path =  id_frame_path + '/' + match_ped + '/' + 'frame' + frame + '.jpg'
#         # shutil.copy2(waiting_ped_path, waiting_target_path)
#         # os.remove(waiting_ped_path)

#     print(waiting_ped, match_ped, match_score, '\n')   
#     print('----------------------------------------')
