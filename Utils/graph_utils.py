# import _pickle as cPickle
# from Utils.dataset import load_allset
# from Utils.data_utils import points2meter,lonlat2meters
# import math
import networkx as nx
# import yaml
# from Utils.trajs import load_trajlist,to_traj
import os
# import numpy as np
# from sklearn.neighbors import KDTree
from tqdm import tqdm
def graph_constructor(grid,dataset):
    # load dataset

    G =nx.Graph()
    grid_size = grid.grid_size
    G.add_nodes_from(range(grid_size))


    path = './TrajData/'
    dir = path + str(dataset)+ '_out'
    traj_list = os.listdir(dir)

    for traj_filename in tqdm(traj_list, desc='Processing Graph'):
        filename = dir + '/' + traj_filename
        f = open(filename)
        pre = None

        linecnt=0
        for line in f:
            linecnt +=1
            temp = line.strip().split(' ')
            if len(temp) < 3:
                continue
            lon = float(temp[1])
            lat = float(temp[0])

            gridID = grid.gps2idx(lon, lat)

            if gridID != 'UNK':
                if pre != None:
                    if G.has_edge(pre, gridID):
                        G[pre][gridID]['weight'] += 1
                    else:
                        G.add_weighted_edges_from([(pre, gridID, 1)])
                else:
                    pre = gridID
            if linecnt>1000:
                break
        f.close()
    return G


def save_G(G, file):
    nx.write_gpickle(G, file)

def load_G(file):
    return nx.read_gpickle(file)