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


def excute_clean_process():

    print('Clean for NEXT TURN')
    # bev_gallary 
    remove_files('../inference/bev_gallary')
    remove_files('../inference/fpv_gallary')
    # clean the id_frame folder in inference folder
    remove_files('../inference/bev_id_frame')
    remove_files('../inference/fpv_id_frame')
    remove_files('../bev_objective')
    remove_files('../fpv_objective')
    # clean the results files
    clean_if_occupied('../results/bev_results')
    clean_if_occupied('../results/fpv_results')

if __name__ == '__main__':
    
    excute_clean_process()