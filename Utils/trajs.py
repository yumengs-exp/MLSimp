
# import torch
# import torch.nn as nn
import torch.nn.functional as F
import os
from itertools import tee
import numpy as np
import math
from Utils.data_utils import lonlat2meters_np
def load_trajlist(dataset):
    path = './TrajData/'
    dir = path + str(dataset)
    file_list = os.listdir(dir)
    return file_list

def to_traj(dataset,file):
    path = './TrajData/'
    dir = path + str(dataset)
    filename = dir+'/'+file
    traj = []
    f = open(filename)
    for line in f:
        temp = line.strip().split(' ')
        if len(temp) < 3:
            continue
        traj.append([float(temp[0]), float(temp[1]), int(float(temp[2]))])
    f.close()
    return traj

def load_allset(dataset):
    orig_trajset = []
    orig_t = []
    path = './TrajData/'
    dir = path+str(dataset)
    file_list = os.listdir(dir)
    max_lon,max_lat=0.0,0.0
    min_lon,min_lat = 500.0,500.0
    t_max = 0
    i=0
    for file in file_list:
        print(i)
        i = i+1
        traj = []
        t=[]
        filename = dir+'/'+str(file)
        f = open(filename)
        for line in f:
            temp = line.strip().split(' ')
            if len(temp) < 3:
                continue
            lon = float(temp[1])
            lat = float(temp[0])
            time = int(temp[2])
            if max_lon<lon:
                max_lon=lon
            if max_lat<lat:
                max_lat=lat
            if min_lon>lon:
                min_lon = lon
            if min_lat>lat:
                min_lat = lat
            traj.append([lon, lat])
            t.append(time)
        f.close()
        orig_trajset.append(traj)
        # 将绝对时间转为相对时间
        t_nor = [i-t[0] for i in t]
        orig_t.append(t_nor)
        t = t[-1]-t[0]
        if t_max<t:
            t_max = t
        # print(t)

    lat_range = [min_lat, max_lat]
    lon_range = [min_lon, max_lon]
    print(lat_range)
    print(lon_range)
    print(t_max)

    return orig_trajset,orig_t, [min_lat, max_lat], [min_lon, max_lon],t_max


def _load(self, dataset, trajlist):
    ori_traj_set = []
    path = './TrajData/'+dataset
    for num in trajlist:
        ori_traj_set.append(F.to_traj(path + str(num)))

# if __name__=='__main__':
#     load_allset('Geolife')



def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)




def l2_distance(lon1, lat1, lon2, lat2):
    return math.sqrt( (lon2 - lon1) ** 2 + (lat2 - lat1) ** 2 )
def generate_original_features(src, x_max, y_max, x_min, y_min):
    # src = [length, 2]
    # src = [src[:2] f]

    src = np.array(src)[:,:2]
    # src_1, src_2  = lonlat2meters_np(src[:,0],src[:,1])
    # src = np.array(list(zip(src_1,src_2)))
    tgt = []
    lens = []
    for p1, p2 in pairwise(src):
        lens.append(l2_distance(p1[0], p1[1], p2[0], p2[1]))
    lens = np.array(lens)

    for i in range(1, len(src) - 1):
        dist = (lens[i - 1] + lens[i]) / 2
        # dist = dist / (Config.trajcl_local_mask_sidelen / 1.414)  # float_ceil(sqrt(2))

        radian = math.pi - math.atan2(src[i - 1][0] - src[i][0], src[i - 1][1] - src[i][1]) \
                 + math.atan2(src[i + 1][0] - src[i][0], src[i + 1][1] - src[i][1])
        radian = 1 - abs(radian) / math.pi

        x = (src[i][0] - x_min) / (x_max - x_min)
        y = (src[i][1] - y_min) / (y_max - y_min)
        tgt.append([x, y, dist, radian])
        # masks.append(mask)

    x = (src[0][0] - x_min) / (x_max - x_min)
    y = (src[0][1] - y_min) / (y_max - y_min)
    tgt.insert(0, [x, y, 0.0, 0.0])


    x = (src[-1][0] - x_min) / (x_max - x_min)
    y = (src[-1][1] - y_min) / (y_max - y_min)
    tgt.append([x, y, 0.0, 0.0])

    # tgt = [length, 4]
    return tgt