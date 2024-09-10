import sys
sys.path.append('..')
import os
import math
import time
import random
import logging
import torch
import pickle
import pandas as pd
from ast import literal_eval
import numpy as np
import multiprocessing as mp
from functools import partial
# from config import load_params
from  tqdm import tqdm
import yaml
from datetime import datetime
def clean_and_output_data():
    path = '../TrajData/'
    dir = path + str(params['dataset'])
    traj_list = os.listdir(dir)
    min_lon = params['minlon']
    max_lon = params['maxlon']
    min_lat = params['minlat']
    max_lat = params['maxlat']
    maxseqlen = params['maxseqlen']
    minseqlen = params['minseqlen']
    for traj_filename in tqdm(traj_list, desc='Processing Grid'):

        filename = dir + '/' + traj_filename
        newdirname = '../TrajData/' + str(params['dataset']) + '_out'
        if not os.path.exists(newdirname):
            try:
                os.makedirs(newdirname)
            except OSError as e:
                print(f"Error: Failed to create directory '{newdirname}'. {e}")


        newfilename = newdirname +'/'+ traj_filename
        newlines = str()
        with open(filename,'r',errors='ignore') as f:

            lines = f.readlines()
            cnt = 0
            if len(lines)<minseqlen:
                continue
            for line in lines:

                temp = line.strip().split(' ')
                if len(temp) < 3:
                    continue
                lon = float(temp[1])
                lat = float(temp[0])
                time = int(temp[2])


                if lon <= min_lon or lon >= max_lon or lat <= min_lat or lat >= max_lat:
                    newlines = ''
                    break
                else:

                    newlines = newlines + str(lat) + ' ' + str(lon) + ' ' + str(time) + '\n'
                    cnt = cnt + 1

        if newlines != ''and cnt>=minseqlen:
            with open(newfilename,'w') as newfile:
                newfile.write(newlines)


import os
import random
import time
import pandas as pd

# For Geolife
root_path = r'../TrajData/Geolife Trajectories 1.3/Data/'
out_path = r'../TrajData/Geolife/'
file_list = []
dir_list = []


def get_file_path(root_path, file_list, dir_list):
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(root_path, dir_file)
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            get_file_path(dir_file_path, file_list, dir_list)
        else:
            file_list.append(dir_file_path)


get_file_path(root_path, file_list, dir_list)

random.shuffle(file_list)

write_name = 0

for fl in file_list:
    if write_name % 100 == 0:
        print('preprocessing ', write_name)
    f = open(fl)
    fw = open(out_path + str(write_name), 'w')
    c = 0
    line_count = 0
    for line in f:
        if c < 6:
            c = c + 1
            continue
        temp = line.strip().split(',')
        if len(temp) < 7:
            continue
        fw.write(temp[0] + ' ' + temp[1] + ' ' + str(
            int(time.mktime(time.strptime(temp[5] + ' ' + temp[6], '%Y-%m-%d %H:%M:%S')))) + '\n')
        line_count = line_count + 1
    f.close()
    fw.close()
    if line_count <= 30:
        os.remove(out_path + str(write_name))
        write_name = write_name - 1
    write_name = write_name + 1



dataset = 'Geolife'
params = yaml.safe_load(open('../Setting.yaml'))[dataset]
clean_and_output_data()

