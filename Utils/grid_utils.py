import pickle
# from Utils.dataset import load_allset
from Utils.data_utils import points2meter,lonlat2meters,lonlat2meters_np
import math
# import networkx as nx
# import yaml
# from Utils.trajs import load_trajlist,to_traj
import os
import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm
class Grid(object):
    def __init__(self,minlon,minlat,maxlon,maxlat,xstep,ystep, dataset= 'Geolife', minfreq=500, max_grid_size=10000, k=1, grid_start=6):
        self.minlon = minlon
        self.minlat = minlat
        self.maxlon = maxlon
        self.maxlat = maxlat
        self.xstep = xstep
        self.ystep = ystep

        self.minx, self.miny = lonlat2meters(minlon, minlat)
        self.maxx, self.maxy = lonlat2meters(maxlon, maxlat)
        numx = round(self.maxx - self.minx, 6) / xstep
        self.numx = int(math.ceil(numx))
        numy = round(self.maxy - self.miny, 6) / ystep
        self.numy = int(math.ceil(numy))

        gridmap_path = "./Grid&Graph/" + dataset + '/gridmap_' + str(xstep) + '_' + str(ystep) + '_' + str(minfreq) + '_' + str(max_grid_size) + '.pickle'
        # self.save_gridmap(None, None, gridmap_path)
        if not os.path.exists(gridmap_path):
            #gridmap initialize
            grid_size, gridmap = self.get_gridmap(minfreq,max_grid_size,k,grid_start,dataset)
            self.save_gridmap(grid_size,gridmap,gridmap_path)
        else:
            grid_size, gridmap = self.load_gridmap(gridmap_path)

        self.grid_size = grid_size
        self.gridmap = gridmap

        return
    def save_gridmap(self,grid_size, gridmap,file):
        grid_dic = {
            'grid_size':grid_size,
            'gridmap':gridmap
        }
        with open(file,'wb') as f:
            pickle.dump(grid_dic,f)

    def load_gridmap(self,file):
        with open(file,'rb') as f:
            grid_dic = pickle.load(f)
        return grid_dic['grid_size'], grid_dic['gridmap']





    def get_gridmap(self, minfreq, max_grid_size, k, grid_start,dataset):


        grids = dict()
        hotgrid = []
        num_out_range= 0

        path = './TrajData/'
        dir = path + str(dataset) + '_out'
        traj_list = os.listdir(dir)


        for traj_filename in tqdm(traj_list, desc='Processing Grid'):

            filename = dir + '/' + traj_filename
            f = open(filename)
            linecnt =0
            for line in f:
                linecnt+=1
                temp = line.strip().split(' ')
                if len(temp) < 3:
                    continue
                lon = float(temp[1])
                lat = float(temp[0])
                if self.out_of_range(lon,lat):
                    num_out_range  += 1
                else:
                    grid = self.gps2grid(lon,lat)
                    if grid in grids.keys():
                        grids[grid]+=1
                    else:
                        grids[grid]=1
                if linecnt>1000:
                    break
            f.close()


        max_grid_size = min(max_grid_size, len(grids.keys()))
        cnt = 0
        grids=sorted(grids.items(), key=lambda d: d[1],reverse=True)
        for grid in grids:
            grid_idx=grid[0]
            freq = grid[1]
            if cnt >=max_grid_size:
                break
            elif freq> minfreq:
                cnt +=1
                hotgrid.append(grid_idx)





        hotgrid2idx = dict([(grid, i - 1 + grid_start)  for (i, grid) in enumerate(hotgrid)])
        grid_size = grid_start+len(hotgrid)


        data = np.zeros([len(hotgrid), 2])
        i = 0
        for grid_ in hotgrid:
            x, y = self.grid2coord(grid_)
            data[i, :] = [x, y]
            i += 1
        hotgrid_kdtree = KDTree(data)

        def knearestHotgrids(grid, k):
            # @assert region.built == true "Build index for region first"
            coord = self.grid2coord(grid)
            #     idx搜索到点的索引，dists返回的是coord到这些近邻点的欧氏距离
            dists, idxs = hotgrid_kdtree.query(np.array([[coord[0], coord[1]]]), k)
            # 取对应下标表示的cell id,因为从上层可以得出此处的k一直为1，所以偷懒写法
            res = hotgrid[idxs[0].tolist()[0]]
            return res, dists

        def nearestHotgrid(grid):
            # @assert region.built == true "Build index for region first"
            hotgrid, _ = knearestHotgrids(grid, 1)
            return hotgrid


        gridmap=[]
        for i in range(self.numx*self.numy):
            if i in hotgrid:
                gridmap.append(hotgrid2idx[i])
            else:
                i_hotgrid = nearestHotgrid(i)
                gridmap.append(hotgrid2idx[i_hotgrid])
        return grid_size, gridmap












    def out_of_range(self,lon, lat):
        return not (self.minlon <= lon < self.maxlon and self.minlat <= lat < self.maxlat)

    def coord2grid(self,x,y):
        xoffset = round(x - self.minx, 6) / self.xstep
        yoffset = round(y - self.miny, 6) / self.ystep
        xoffset = int(math.floor(xoffset))
        yoffset = int(math.floor(yoffset))
        return yoffset * self.numx + xoffset
    def gps2grid(self,lon,lat):
        x,y = lonlat2meters(lon,lat)
        return self.coord2grid(x,y)

    def grid2coord(self,grid):
        yoffset = grid // self.numx
        xoffset = grid % self.numx
        y = self.miny + (yoffset + 0.5) * self.ystep
        x = self.minx + (xoffset + 0.5) * self.xstep
        return x, y

    def gps2idx(self,lon, lat):
        if self.out_of_range(lon, lat):
            return "UNK"
        return self.grid2idx(self.gps2grid(lon, lat))

    def grid2idx(self, grid):
        return self.gridmap[grid]


    # def traj2idxseq(self,traj):
    #     seq = []
    #     for i in range(len(traj)):
    #         lon,lat = traj[i]
    #         x = self.gps2idx(lon,lat)
    #         if x =='UNK':
    #             continue
    #         seq.append(x)
    #     return seq

    def traj2idxseq(self,traj:np.array):
        x, y = lonlat2meters_np(traj[:,:,0], traj[:,:,1])

        # def coord2grid(self, x, y):
        xoffset = np.round(x - self.minx, 6) / self.xstep
        yoffset = np.round(y - self.miny, 6) / self.ystep
        xoffset = xoffset.astype(int)
        yoffset = yoffset.astype(int)
        grids = yoffset * self.numx + xoffset
        grid_id = np.array(self.gridmap)
        return grid_id[grids]










# class Graph(object):
#     #图权重
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.G = nx.Graph()
#
#
#         # self.dataset= dataset
#
#
#         # if load == False:
#         #     self.graph_constructor()
#         # self.delta = delta
#         # self.lat_range = lat_range
#         # self.lon_range = lon_range
#         # self._init_grid_hash_function()
#
#     def graph_constructor(self,grid):
#         # load dataset
#
#         grid_size = grid.grid_size
#         self.G.add_node(range(grid_size))
#
#         path = './TrajData/'
#         dir = path + str(self.dataset)
#         traj_list = os.listdir(dir)
#
#
#         for traj_filename in traj_list:
#             filename = dir + '/' + traj_filename
#             f = open(filename)
#             pre = None
#             for line in f:
#                 temp = line.strip().split(' ')
#                 if len(temp) < 3:
#                     continue
#                 lon = float(temp[1])
#                 lat = float(temp[0])
#                 time = int(temp[2])
#
#                 gridID = grid.gps2idx(lon, lat)
#
#                 if gridID != 'UNK':
#                     if pre != None:
#                         if self.G.has_edge(pre,gridID):
#                             self.G[pre][gridID]['weight'] +=1
#                         else:
#                             self.G.add_weighted_edges_from([(pre,gridID,1)])
#             f.close()
#         self.save_G(grid.xstep,grid.ystep,grid.minfreq,grid.max_grid_size,grid.k,grid.grid_start)
#     def graph_loader(self,grid):
#         self.load_G(grid.xstep,grid.ystep,grid.minfreq,grid.max_grid_size,grid.k,grid.grid_start)
#     def save_G(self,xstep,ystep,minfreq,max_grid_size,k,grid_start):
#         file = './TrajSimp/Grid&Graph/' + self.dataset + '/graph_' + str(xstep) + '_' + str(ystep) + '_' + \
#                str(minfreq) + '_' + str(max_grid_size) + '_' + str(k) + '_' + str(grid_start) + '.txt'
#         nx.write_gpickle(self.G,file)
#     def load_G(self,xstep,ystep,minfreq,max_grid_size,k,grid_start):
#         file = './TrajSimp/Grid&Graph/' + self.dataset + '/graph_' + str(xstep) + '_' + str(ystep) + '_' + \
#                str(minfreq) + '_' + str(max_grid_size) + '_' + str(k) + '_' + str(grid_start) + '.txt'
#         self.G = nx.read_gpickle(file)
#
#         orig_trajset, self.lat_range, self.lon_range = load_allset(self.dataset)
#         self.mXmin, self.mXmax, self.mYmin, self.mYmax = points2meter([self.lat_range[0], self.lat_range[1], self.lon_range[0], self.lon_range[1]])
#
#         # init node
#         X_num = math.ceil((self.mXmax -self.mXmin) / self.len_grid)
#         Y_num = math.ceil((self.mYmax -self.mYmin) / self.len_grid)
#         self.node_num = X_num * Y_num
#         self.G.add_nodes_from(range(self.node_num))
#
#         # trajectory 2 meter
#         meter_trajset = [points2meter(traj) for traj in orig_trajset]
#         for traj in meter_trajset:
#             last = -1
#             for point in traj:
#                 x_grid = int((point[0] - self.mXmin) / self.len_grid)
#                 y_grid = int((point[1] - self.mYmin) / self.len_grid)
#                 index = (y_grid) * (len(X_num)) + x_grid
#                 if last != -1:
#                     self.G.add_edge(last,index)
#                 last = index
#         self.G.remove_edges_from(nx.selfloop_edges(self.G))
#
#         #save graph
#         yamlfile = './GraphData/'+str(self.dataset)+'/info.yaml'
#         graphinfo = {
#             'node_num': self.node_num,
#             'edge_num': self.G.number_of_edges(),
#             'lat_range': self.lat_range,
#             'lon_range': self.lon_range,
#             'Xmin': self.mXmin,
#             'Xmax': self.mXmax,
#             'Ymin': self.mYmin,
#             'Ymax': self.mYmax,
#         }
#         with open(yamlfile, 'w') as outfile:yaml.dump(graphinfo, outfile, default_flow_style=False)
#
#         adjfile = './GraphData/' + str(self.dataset) + '/adj'
#         nx.write_adjlist(self.G,adjfile)
#
#
#
#
#     def load_graph(self):
#         infofile = './GraphData/' + str(self.dataset) + '/info.yaml'
#         adjfile = './GraphData/' + str(self.dataset) + '/adj'
#         self.G = nx.read_adjlist(adjfile)
#         info = yaml.safe_load(open(infofile))
#         self.node_num = info['node_num']
#         self.lat_range = info['lat_range']
#         self.lon_range = info['lon_range']
#         self.mXmin = info['Xmin']
#         self.mXmax = info['Xmax']
#         self.mYmin = info['Ymin']
#         self.mYmax = info['Ymax']
#         #查一下点
#
#     def traj2grid_seq(self, trajs = [], isCoordinate = False):
#         grid_traj = []
#         for r in trajs:
#             x_grid, y_grid, index = self.get_grid_index((r[2],r[1]))
#             grid_traj.append(index)
#
#         privious = None
#         hash_traj = []
#         for index, i in enumerate(grid_traj):
#             if privious==None:
#                 privious = i
#                 if isCoordinate == False:
#                     hash_traj.append(i)
#                 elif isCoordinate == True:
#                     hash_traj.append(trajs[index][1:])
#             else:
#                 if i==privious:
#                     pass
#                 else:
#                     if isCoordinate == False:
#                         hash_traj.append(i)
#                     elif isCoordinate == True:
#                         hash_traj.append(trajs[index][1:])
#                     privious = i
#         return hash_traj
# #
# #
# #     def _init_grid_hash_function(self):
# #         dXMax, dXMin, dYMax, dYMin = self.lon_range[1], self.lon_range[0], self.lat_range[1], self.lat_range[0]
# #         x = self._frange(dXMin,dXMax, self.delta)
# #         y = self._frange(dYMin,dYMax, self.delta)
# #         self.x = x
# #         self.y = y
# #
# #     def _frange(self, start, end=None, inc=None):
# #         "A range function, that does accept float increments..."
# #         if end == None:
# #             end = start + 0.0
# #             start = 0.0
# #         if inc == None:
# #             inc = 1.0
# #         L = []
# #         while 1:
# #             next = start + len(L) * inc
# #             if inc > 0 and next >= end:
# #                 break
# #             elif inc < 0 and next <= end:
# #                 break
# #             L.append(next)
# #         return L
# #
# #     def get_grid_index(self, tuple):
# #         test_tuple = tuple
# #         test_x,test_y = test_tuple[0],test_tuple[1]
# #         x_grid = int((test_x-self.lon_range[0])/self.delta)
# #         y_grid = int((test_y-self.lat_range[0])/self.delta)
# #         index = (y_grid)*(len(self.x)) + x_grid
# #         return x_grid,y_grid, index
# #
# #     def traj2grid_seq(self, trajs = [], isCoordinate = False):
# #         grid_traj = []
# #         for r in trajs:
# #             x_grid, y_grid, index = self.get_grid_index((r[2],r[1]))
# #             grid_traj.append(index)
# #
# #         privious = None
# #         hash_traj = []
# #         for index, i in enumerate(grid_traj):
# #             if privious==None:
# #                 privious = i
# #                 if isCoordinate == False:
# #                     hash_traj.append(i)
# #                 elif isCoordinate == True:
# #                     hash_traj.append(trajs[index][1:])
# #             else:
# #                 if i==privious:
# #                     pass
# #                 else:
# #                     if isCoordinate == False:
# #                         hash_traj.append(i)
# #                     elif isCoordinate == True:
# #                         hash_traj.append(trajs[index][1:])
# #                     privious = i
# #         return hash_traj
# #
# #     def _traj2grid_preprocess(self, traj_feature_map, isCoordinate =False):
# #         trajs_hash = []
# #         trajs_keys = traj_feature_map.keys()
# #         for traj_key in trajs_keys:
# #             traj = traj_feature_map[traj_key]
# #             trajs_hash.append(self.traj2grid_seq(traj, isCoordinate))
# #         return trajs_hash
# #
# #     def preprocess(self, traj_feature_map, isCoordinate = False):
# #         if not isCoordinate:
# #             traj_grids = self._traj2grid_preprocess(traj_feature_map)
# #             print('gird trajectory nums {}'.format(len(traj_grids)))
# #
# #             useful_grids = {}
# #             count = 0
# #             max_len = 0
# #             for i, traj in enumerate(traj_grids):
# #                 if len(traj) > max_len: max_len = len(traj)
# #                 count += len(traj)
# #                 for grid in traj:
# #                     if useful_grids.has_key(grid):
# #                         useful_grids[grid][1] += 1
# #                     else:
# #                         useful_grids[grid] = [len(useful_grids) + 1, 1]
# #             print(len(useful_grids.keys()))
# #             print(count, max_len)
# #             return traj_grids, useful_grids, max_len
# #         elif isCoordinate:
# #             traj_grids = self._traj2grid_preprocess(traj_feature_map, isCoordinate=isCoordinate)
# #             max_len = 0
# #             useful_grids = {}
# #             for i, traj in enumerate(traj_grids):
# #                 if len(traj) > max_len:
# #                     max_len = len(traj)
# #             return traj_grids, useful_grids, max_len
# #
# #     def read_StrToBytes(self, path,):
# #         return self.fileobj.read(path).encode()
# #
# #     def write_BytesToStr(self, size=-1):
# #         return self.fileobj.readline(size).encode()
# #
# # def trajectory_feature_generation(path ='./data/toy_trajs',
# #                                   lat_range = lat_range,
# #                                   lon_range = lon_range,
# #                                   min_length=50):
# #     fname = path.split('/')[-1].split('_')[0]  # porto
# #
# #     trajs = cPickle.loads(open(path, 'rb+').read(), encoding='bytes')
# #     traj_index = {}
# #     max_len = 0
# #     preprocessor = Preprocesser(delta = 0.001, lat_range = lat_range, lon_range = lon_range)
# #     for i, traj in enumerate(trajs):
# #         new_traj = []
# #         coor_traj = []
# #         if (len(traj)>min_length):
# #             inrange = True
# #             for p in traj:
# #                 lon, lat = p[0], p[1]
# #                 if not ((lat > lat_range[0]) & (lat < lat_range[1]) & (lon > lon_range[0]) & (lon < lon_range[1])):
# #                     inrange = False
# #                 new_traj.append([0, p[1], p[0]])
# #
# #             if inrange:
# #                 coor_traj = preprocessor.traj2grid_seq(new_traj, isCoordinate=True)
# #                 if len(coor_traj)==0:
# #                     print(len(coor_traj))
# #                 if ((len(coor_traj) >10) & (len(coor_traj)<150)):
# #                     if len(traj) > max_len:
# #                         max_len = len(traj)
# #                     traj_index[i] = new_traj
# #         if len(traj_index) >= 10000:
# #             break
# #
# #     print("轨迹序列最长为{}".format(max_len))
# #     print("一共有{}条轨迹".format(len(traj_index.keys())))
# #
# #     cPickle.dump(traj_index, open('./features/{}_traj_index'.format(str(fname)),'wb'))
# #
# #     trajs, useful_grids, max_len = preprocessor.preprocess(traj_index, isCoordinate=True)
# #     cPickle.dump((trajs,[],max_len), open('./features/{}_traj_coord'.format(str(fname)), 'wb'))
# #
# #     all_trajs_grids_xy = []
# #     min_x, min_y, max_x, max_y = 2000, 2000, 0, 0
# #     for i in trajs:
# #         for j in i:
# #             x, y, index = preprocessor.get_grid_index((j[1], j[0]))
# #             if x < min_x:
# #                 min_x = x
# #             if x > max_x:
# #                 max_x = x
# #             if y < min_y:
# #                 min_y = y
# #             if y > max_y:
# #                 max_y = y
# #     print(min_x, min_y, max_x, max_y)
# #
# #     for i in trajs:
# #         traj_grid_xy = []
# #         for j in i:
# #             x, y, index = preprocessor.get_grid_index((j[1], j[0]))
# #             x = x - min_x
# #             y = y - min_y
# #             grids_xy = [y, x]
# #             traj_grid_xy.append(grids_xy)
# #         all_trajs_grids_xy.append(traj_grid_xy)
# #
# #     print(all_trajs_grids_xy[0])
# #     print(len(all_trajs_grids_xy))
# #
# #     cPickle.dump((all_trajs_grids_xy,[],max_len), open('./features/{}_traj_grid'.format(fname), 'wb'))
# #     return './features/{}_traj_coord'.format(fname), fname