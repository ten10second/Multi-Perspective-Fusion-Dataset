from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import re
import os.path as osp
import pandas as pd
import numpy as np
import time
import cv2
from lib.opts import opts
from lib.tracking_utils.utils import mkdir_if_missing, clean_if_occupied, remove_files
from lib.tracking_utils.log import logger
import lib.datasets.dataset.jde as datasets
from track import eval_seq, del_files

import sys
sys.path.append('peds_correlation.py')
import peds_correlation    # tool script 

logger.setLevel(logging.INFO)

# 根据分辨率调整
def calculate_center_x(row):
    return row['x_1']  + row['width'] / 2

def calculate_center_y(row):
    return row['y_1'] + row['height'] / 2

def bbox2trajectory(opt, date_info):
    bev_result_path = opt.bev_results

    for bev_text in os.listdir(bev_result_path):
        bev_bbox = os.path.join(bev_result_path, bev_text)    #../results/bev_results/2023_04_03_bev_bbox_1.txt
        video_index = bev_bbox[:-4].split('_')[-1]   #'1'

        columns_name = ['frame', 'id', 'x_1', 'y_1', 'width', 'height', '0', '1', '2', '3']
        pixel_data = pd.read_csv(bev_bbox, sep=',', names = columns_name)
        pixel_data = pixel_data.iloc[:, :6]

        pixel_data['center_x'] = pixel_data.apply(calculate_center_x, axis=1)
        pixel_data['center_y'] = pixel_data.apply(calculate_center_y, axis=1)
        pixel_data.drop(['x_1', 'y_1', 'width', 'height'], axis=1, inplace=True)
        pixel_trajectory_path = bev_result_path + '/'+ date_info + '_bev-pixel-trajectory_' + video_index + '.txt'

        pixel_data.to_csv(pixel_trajectory_path, sep='\t', index=False, header=False)
        os.remove(bev_bbox)       # delete the bbox info

def img_correlation(path1, path2):

    img1 = cv2.imread(path1,0)
    img2 = cv2.imread(path2,0)

    hist1 = cv2.calcHist([img1],[0],None,[256],[0,256])
    hist2 = cv2.calcHist([img2],[0],None,[256],[0,256])

    cv2.normalize(hist1,hist1,0,255,cv2.NORM_MINMAX)
    cv2.normalize(hist2,hist2,0,255,cv2.NORM_MINMAX)

    correl = cv2.compareHist(hist1,hist2,cv2.HISTCMP_CORREL)

    return correl   # 返回两张图像的相似度值

def similar(path):
    
    back_image = '../center_person_image/back_photo.jpg'
    front_image = '../center_person_image/front_photo.jpg'

    if img_correlation(path, back_image) > 0.8 or img_correlation(path, front_image) > 0.8:
        return True
    else:
        return False


def bev_demo(opt):
    
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    # opt.input_folder: ../videos/23.04.03
    bev_folder_path = opt.input_folder + '/bev_folder'   # ../videos/23.04.03/bev_folder
    bev_gallary_path = '../inference/bev_gallary'

    # 清空 '../results/bev_results' 文件夹下的文件
    del_files(opt.bev_results)

    for cut in os.listdir(bev_folder_path):

        opt.input_video = bev_folder_path + '/' + cut
        #---------------------------------------#
        # opt.input_video path : ../videos/23.04.03/bev_folder/2023_04_03_sony_1.mp4
        video_number = opt.input_video.split('/')[-1]    # 2023_04_03_sony_1.mp4
        video_info, suffix = video_number.split('.')   # 2023_04_03_sony_1
        video_index = video_info.split('_')[-1]       # 1
        date_info = video_info.split('_')[0:3]         
        date_info = '_'.join(date_info)               # 2023_04_03
        perspective_info = video_info.split('_')[-2]     # sony / action2
        if perspective_info == 'sony':
            bev_perspective = 'bev'
            results_txt_folder = opt.bev_results  # ../results/bev_results
        #---------------------------------------#
        result_name = date_info + '_' + bev_perspective + '_bbox_' + video_index + '.txt'
        result_path = os.path.join(results_txt_folder, result_name)

        dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
        frame_rate = dataloader.frame_rate
        frame_dir = None if opt.output_format == 'text' else osp.join(result_root, video_index +'_video')
        eval_seq(opt, bev_perspective, dataloader, 'mot', result_path,
                save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
                use_cuda=opt.gpus!=[-1])

        print('-------Data Processing--------')

        peds_correlation.making_the_bev_id_frame_folder(bev_perspective, video_index)
        print('---------Rename the photo in the folder----------')
        # 需要配合 目标检测 不然每次都会单独修改图片名称 
        peds_correlation.rename(bev_perspective, video_index)
        
        print('---------Make the bev-gallary----------')
        peds_correlation.make_bev_gallary_file(bev_perspective, video_index)
        bev_ped_number = peds_correlation.counting_file(bev_gallary_path)

        print('『Save center person info』')
        print('Information Details:')
        print('video_info: {}'.format(video_number))
        print('view: {}'.format(bev_perspective))
        print('video_index: {}'.format(video_info.split('_')[-1]))
        print('pedestrian number of BEV view: {}'.format(bev_ped_number))
        
        if opt.output_format == 'video':
            output_video_path = osp.join(result_root, video_number)
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, video_index +'_frame'), output_video_path)
            os.system(cmd_str)

    return date_info

def fpv_demo(opt):
    
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting first-person view tracking...')
    # opt.input_folder: ../videos/23.04.03
    fpv_folder_path = opt.input_folder + '/fpv_folder'   # ../videos/23.04.03/fpv_folder
    fpv_gallary_path = '../inference/fpv_gallary'

    del_files(opt.fpv_results)

    for cut in os.listdir(fpv_folder_path):

        fpv_input_video = fpv_folder_path + '/' + cut
        print(fpv_input_video)
        #---------------------------------------#
        # fpv_input_video path : ../videos/23.04.03/fpv_folder/2023_04_03_action2_1.mp4
        video_number = fpv_input_video.split('/')[-1]    # 2023_04_03_action2_1.mp4
        video_info, suffix = video_number.split('.')   # 2023_04_03_action2_1
        video_index = video_info.split('_')[-1]       # '1'
        date_info = video_info.split('_')[0:3]         
        date_info = '_'.join(date_info)               # 2023_04_03
        perspective_info = video_info.split('_')[-2]     # sony / action2
        if perspective_info == 'action2':
            fpv_perspective = 'fpv'
            results_txt_folder = opt.fpv_results
        #---------------------------------------#
        result_name = date_info + '_' + fpv_perspective + '_bbox_' + video_index + '.txt'
        result_path = os.path.join(results_txt_folder, result_name)

        dataloader = datasets.LoadVideo(fpv_input_video, opt.img_size)
        frame_rate = dataloader.frame_rate
        frame_dir = None if opt.output_format == 'text' else osp.join(result_root, video_index +'_video')
        eval_seq(opt, fpv_perspective, dataloader, 'mot', result_path,
                save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
                use_cuda=opt.gpus!=[-1])

        print('-------Data Processing(make the fpv_id_frame_file)--------')

        peds_correlation.making_the_fpv_id_frame_folder(fpv_perspective, video_index)
        print('---------Rename the photo in the folder----------')
        # 需要配合 目标检测 不然每次都会单独修改图片名称 
        peds_correlation.rename(fpv_perspective, video_index)
        
        print('---------Make the fpv-gallary----------')
        peds_correlation.make_fpv_gallary_file(fpv_perspective, video_index)
        fpv_ped_number = peds_correlation.counting_file(fpv_gallary_path)
        print('Information Details:')
        print('video_info: {}'.format(video_number))
        print('view: {}'.format(fpv_perspective))
        print('video_index: {}'.format(video_info.split('_')[-1]))
        print('pedestrian number of first-person view: {}'.format(fpv_ped_number))
        
        if opt.output_format == 'video':
            output_video_path = osp.join(result_root, video_number)
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, video_index +'_frame'), output_video_path)
            os.system(cmd_str)


def pixel2world(opt, sample_rate):

    video_frame = 25
    date_info = opt.input_folder.split('/')[-1]  # '2023_04_03'
    chess_folder_path = opt.input_folder + '/' + date_info + '_intrinsic_calibration'
    print(chess_folder_path)
    extrinsic_calib_video = opt.input_folder + '/' + date_info + '_extrinsic_calibration' + '/' + date_info + '_extrinsic_calibration' + '.mp4'
    extrinsic_calib_path = opt.input_folder + '/' + date_info + '_extrinsic_calibration' + '/' + date_info + '_extrinsic_calibration' + '.txt'
    cap = cv2.VideoCapture(extrinsic_calib_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, mtx, dist = peds_correlation.get_intrinsic_parameters(chess_folder_path)
    rvecs, tvecs, R = peds_correlation.get_extrinsic_parameters(width, height, mtx, dist, 
                                                                calibration_text=extrinsic_calib_path)
    camera_parameter = {
                        "R": R,
                        "T": tvecs,
                        "camera_intrinsic" : mtx
    }
    camera_intrinsic = camera_parameter["camera_intrinsic"]
    R = camera_parameter["R"]
    t = np.asmatrix(camera_parameter["T"])

    bev_results = opt.bev_results  # ../results/bev_results

    for text in os.listdir(bev_results):
        pixel_trajectory_path = os.path.join(bev_results, text)  # 'results/bev_results/2023_04_03_bev-pixel-trajectory_1.txt' 
        video_index = pixel_trajectory_path[:-4].split('_')[-1]   #'1'
        world_trajectory_name = date_info + '_bev-world-trajectory_' + video_index + '.txt'
        
        with open(pixel_trajectory_path, 'r') as f:
            Pixel_Data = f.readlines()

        with open(os.path.join(bev_results, world_trajectory_name), 'w') as f:
             
             for row_pixel_data in Pixel_Data:
                img_points = []
                n = video_frame / sample_rate

                row_pixel_data = row_pixel_data.rstrip('\n').split('\t')
                row_pixel_data = list(map(float, row_pixel_data))
                frame = row_pixel_data[0]
                if frame % n == 0:
                    ped_id = row_pixel_data[1]
                    px = row_pixel_data[2]
                    py = row_pixel_data[3]

                    img_points.append([px, py]) 

                result = peds_correlation.pixel_to_world(camera_intrinsic, R, t, img_points)

                for item in result:
                    for word_corr in item:
                        word_x = word_corr[0] / 100  # cm --> m 
                        word_y = word_corr[1] / 100

                        f.write(str(int(frame)) + ' ' + str(int(ped_id)) + ' ' + str(word_x) + ' ' + str(word_y) + '\n')

        os.remove(pixel_trajectory_path)   # 删除像素坐标系文件 保留转化后的世界坐标系文件
    print('『 The pixel coordinate file has been deleted 』')


def smooth_trajectory(opt):

    bev_pos_folder = opt.bev_results     # ../results/bev_results
    for pos_file in os.listdir(bev_pos_folder):
        pos_file_path = os.path.join(bev_pos_folder, pos_file)

        with open(pos_file_path, 'r') as f:
            arr = [list(map(float, l.rstrip().split())) for l in f]
            arr = np.array(arr, dtype=np.float32)
            for ped_index in np.unique(arr[:, 1]):
                condition = arr[:, 1] == ped_index
                index = np.where(condition)[0]
                condition_arr = arr[condition]
                if len(condition_arr) > 5:
                    condition_arr_x = condition_arr[:, 2]  # x
                    condition_arr_y = condition_arr[:, 3]  # y
                    condition_arr_x, condition_arr_y = peds_correlation.smooth(condition_arr_x, condition_arr_y)
                    for i in range(len(index)):
                        arr[index[i], 2] = condition_arr_x[i]
                        arr[index[i], 3] = condition_arr_y[i]
    
    print('『 Smoothing Process has been completed 』')
    

def make_center_person_file(opt, center_person_dict):
    
    bev_pos_folder = opt.bev_results     # ../results/bev_results
    for pos_file in os.listdir(bev_pos_folder):
        pos_file_path =  os.path.join(bev_pos_folder, pos_file)  # '../results/bev_results/2023_04_03_bev_bbox_1.txt'
        video_index = pos_file_path[:-4].split('_')[-1]  # '1'
        date_pattern = r"(\d{4})_(\d{2})_(\d{2})"
        match = re.search(date_pattern, pos_file_path)

        if match:
            date_info = match.group(1) + '_' + match.group(2) + '_' + match.group(3)

        center_ped = center_person_dict[video_index]  # 'ped2'
        int_ped_id = int(re.search('\d+', center_ped).group())   #取出在字符串'ped244'中的int 244
        center_person_trajectory_filename = date_info + '_bev_' + video_index + '_' + center_ped + '.txt'   # 2023_04_03_bev_1_ped2.txt
        
        with open(pos_file_path, 'r') as f:
            arr = [list(map(float, l.rstrip().split())) for l in f]
            arr = np.array(arr, dtype=np.float32)
            condition = arr[:, 1] == int_ped_id
            condition_arr = arr[condition]
            center_ped_trajectory_path = os.path.join(bev_pos_folder, center_person_trajectory_filename)   # ../results/bev_results/ped2
            # 检查文件是否存在 如果存在先清空
            if os.path.exists(center_ped_trajectory_path):
                os.remove(center_ped_trajectory_path)
            np.savetxt(center_ped_trajectory_path, condition_arr, fmt='%d %d %.4f %.4f')
                            
    print('『 Filtered the center-person trajectory file 』')

def sort_folders_by_file_count(path):
    # 根据video中的子文件夹数量 return the candidate_list
    folders = []
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            folders.append(folder)
    folders.sort(key=lambda x: len(os.listdir(os.path.join(path, x))), reverse=True)  # [ped8, ped2]
    
    return folders

def get_movement_distance(path, ped_id):
    
    # 输入轨迹文件的路径 'results/bev_results/2023_04_03_bev-world-trajectory_1.txt' 
    # ped_id  --> 'ped2'
    int_ped_id = int(re.search('\d+', ped_id).group())
    with open(path, 'r') as f:
        arr = [list(map(float, l.rstrip().split())) for l in f]
        arr = np.array(arr, dtype=np.float32)
        condition = arr[:, 1] == int_ped_id
        condition_arr = arr[condition]

        if len(condition_arr) > 3:

            # 判断candidate-center-person 是否符合移动距离的条件
            start_pos = condition_arr[0][-2:]
            destination_pos = condition_arr[-1][-2:]
            movement_distance = np.sqrt(np.sum(np.square(start_pos - destination_pos)))     
    
            return movement_distance
        else:
            return 0

def pick_center_person(opt, candidate_list, date_info, video_index):

    bev_pos_folder = opt.bev_results     # ../results/bev_results
    bev_id_frame = '../' + 'inference' +'/' + 'bev_id_frame'
    bev_gallary = '../inference/bev_gallary/' + 'video_' + video_index  # '../inference/bev_id_frame/video_5'
    center_person = ''
    candidate_person_id = candidate_list[0]  # ped53

    trajectory_path = bev_pos_folder + '/' + date_info + '_' + 'bev-world-trajectory' + '_' + video_index + '.txt'
    # candidate_person_image  '../inference/bev_id_frame/ped6 
    for bev_gallary_image in os.listdir(bev_gallary):
        _ = bev_gallary_image.split('.')[0]
        gallary_ped_id = _.split('_')[0]  # '0053'
        
        if int(re.search('\d+', candidate_person_id).group()) == int(gallary_ped_id):
            candidate_person_path = bev_gallary + '/' + bev_gallary_image

    if get_movement_distance(trajectory_path, candidate_person_id) > 10 and similar(candidate_person_path):
        center_person = candidate_person_id
    else:
        candidate_list.remove(candidate_person_id)
        if len(candidate_list) > 1:
            center_person = pick_center_person(opt, candidate_list, date_info,video_index)
        else: 
            center_person = candidate_list[-1]

    return center_person

def confirm_center_person(opt, date_info):
    
    bev_pos_folder = opt.bev_results    # ../results/bev_results
    bev_id_frame = '../' + 'inference' +'/' + 'bev_id_frame'
    
    center_person_dict = {}

    for text in os.listdir(bev_pos_folder):

        trajectory_path = os.path.join(bev_pos_folder, text)  # 'results/bev_results/2023_04_03_bev-world-trajectory_1.txt' 
        video_index = trajectory_path[:-4].split('_')[-1]   #'1'  
        subfolder_bev_id_frame = bev_id_frame + '/' + 'video_' + video_index   # '../inference/bev_id_frame/video_1'
        candidate_list = sort_folders_by_file_count(subfolder_bev_id_frame)
        center_person = pick_center_person(opt, candidate_list, date_info, video_index)
        center_person_dict[video_index] = center_person

    return center_person_dict
        

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()

    date_info = bev_demo(opt)        # center_id: [ped2, ped45, ...]
    fpv_demo(opt)
    time.sleep(5)

    print('---------pixel2pixel_trajectory---------')
    bbox2trajectory(opt, date_info) 
    time.sleep(5)

    # pixel --> world  only for bev
    print('----------pixel2world with sample----------')
    pixel2world(opt, sample_rate=2.5)
    time.sleep(5)

    # sample & smooth
    print('-------------smooth--------------')
    smooth_trajectory(opt)
    time.sleep(5)

    print('---------confirm the center person id------------')
    center_person_dict = confirm_center_person(opt, date_info)
    print(center_person_dict)


    print('-----Make the center-person trajectory file-----')
    make_center_person_file(opt, center_person_dict)
    print('\n' * 3)