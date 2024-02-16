
import re
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--date_info', default='2023_04_03')
parser.add_argument('--ped_id', default='ped62')
parser.add_argument('--video_index', default='6')

args = parser.parse_args()

def make_patch(date_info, ped_id, video_index):

    bev_gallary = '../inference/bev_gallary/' + 'video_' + video_index 
    bev_pos_folder = '../results/bev_results'
    
    pos_file_path = bev_pos_folder + '/' + date_info + '_' + 'bev-world-trajectory' + '_' + video_index + '.txt'
    center_person_trajectory_filename = date_info + '_bev_' + video_index + '_' + ped_id + '.txt'   # 2023_04_03_bev_1_ped2
    
    int_ped_id = int(re.search('\d+', ped_id).group())
    
    with open(pos_file_path, 'r') as f:
        arr = [list(map(float, l.rstrip().split())) for l in f]
        arr = np.array(arr, dtype=np.float32)
        condition = arr[:, 1] == int_ped_id
        condition_arr = arr[condition]
        center_ped_trajectory_path = os.path.join(bev_pos_folder, center_person_trajectory_filename)   # ../results/bev_results/ped2

        np.savetxt(center_ped_trajectory_path, condition_arr, fmt='%d %d %.4f %.4f')

     
if __name__ == '__main__':
    
    make_patch(args.date_info, args.ped_id, args.video_index)