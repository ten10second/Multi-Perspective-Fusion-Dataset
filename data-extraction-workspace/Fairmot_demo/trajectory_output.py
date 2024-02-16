import os
from pathlib import Path
import pandas as pd
import numpy as np

# 根据分辨率调整
def calculate_center_x(row):
    return row['x_1']  + row['width'] / 2 

def calculate_center_y(row):
    return row['y_1'] + row['height'] / 2 

def bbox2trajectory(result_path):
    # txt_path = 'demos/bev_results.txt'

    columns_name = ['frame', 'id', 'x_1', 'y_1', 'width', 'height', '0', '1', '2', '3']
    pixel_data = pd.read_csv(result_path, sep=',', names = columns_name)
    pixel_data = pixel_data.iloc[:, :6]

    pixel_data['center_x'] = pixel_data.apply(calculate_center_x, axis=1)
    pixel_data['center_y'] = pixel_data.apply(calculate_center_y, axis=1)
    pixel_data.drop(['x_1', 'y_1', 'width', 'height'], axis=1, inplace=True)

    pixel_data.to_csv('pixel_trajectory.txt', sep='\t', index=False, header=False)


# center_person = pixel_data.loc[pixel_data['id'] == 5]

############plot##########################


# center_person[['frame', 'id', 'center_x', 'center_y']].to_csv('id_5.txt', sep='\t', index=False)

# import matplotlib.pyplot as plt
# import pandas as pd

# plt.plot(center_person['center_x'], center_person['center_y'])
# plt.show()

############## 像素坐标转世界坐标 ###############
extrinsic_text = 'videos/23.04.03/2023_04_03_extrinsic calibration/2023_4_3_extrinsic calibration.txt'
def read_extrinsic_text(d):
    with open(d, 'r') as f:
        next(f)
        arr = [line.strip() for line in f]  

    
    pixel_point = np.empty([6, 2], dtype=np.float32)
    world_point = np.empty([6, 3], dtype=np.float32)
    return arr

res = read_extrinsic_text(extrinsic_text)
print(res)