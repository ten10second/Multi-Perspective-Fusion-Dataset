import cv2 as cv
import os

bev_gallary_path = 'inference/bev_gallary'
bev_id_frame_path = 'inference/bev_id_frame'
fpv_gallary_path = 'inference/fpv_gallary'
fpv_in_frame_path = 'inference/fpv_id_frame'


# 计算图像的相似度

def calculate_correlation(img_path1, img_path2):

    img1 = cv.imread(img_path1,0)
    img2 = cv.imread(img_path2,0)

    hist1 = cv.calcHist([img1],[0],None,[256],[0,256])
    hist2 = cv.calcHist([img2],[0],None,[256],[0,256])

    cv.normalize(hist1,hist1,0,255,cv.NORM_MINMAX)
    cv.normalize(hist2,hist2,0,255,cv.NORM_MINMAX)

    correl = cv.compareHist(hist1,hist2,cv.HISTCMP_CORREL)

    return correl

# for fpv_ped in os.listdir(fpv_gallary_path):
    
#     fpv_gallary_image_path = fpv_gallary_path + '/' + fpv_ped
#     max_corr = 0
#     target_ped = ""  
    
#     for bev_ped_id in os.listdir(bev_id_frame_path):
#         correlation_list = []
#         bev_id_file = bev_id_frame_path + '/' + bev_ped_id
#         for bev_frame in os.listdir(bev_id_file):
#             bev_frame_path = bev_id_file + '/' + bev_frame
            
#             correlation_list.append(calculate_correlation(fpv_gallary_image_path, bev_frame_path))

#         bev_id_corr = sum(correlation_list) / len(correlation_list)

#         if bev_id_corr > max_corr:
#             target_ped = bev_ped_id
#             max_corr = bev_id_corr
    
#     print(fpv_ped, target_ped)

target_ped = ""
for bev_id in os.listdir(bev_gallary_path):
    bev_ped_path =  bev_gallary_path + '/' + bev_id
    max_corr = 0.

    for fpv_id in os.listdir(fpv_gallary_path):
        fpv_ped_path = fpv_gallary_path + '/' + fpv_id

        corr = calculate_correlation(bev_ped_path, fpv_ped_path)
        if corr > max_corr:
            target_ped = fpv_id
            max_corr = corr
        
    print(bev_id, target_ped, max_corr)
    print('--------------------------')
    # if bev_id_corr > max_corr:
    #     target_ped = bev_id
    #     max_corr = bev_id_corr

    # print(fpv_id, target_ped, max_corr)
        
        
        
    