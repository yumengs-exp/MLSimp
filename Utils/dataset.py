import time

import numpy as np

import os
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from Utils.trajs import generate_original_features
import torch
from torch.nn.utils.rnn import pad_sequence
import pickle
from Model.tbert import TBERT, TBERTEmbedding, PositionalEncoding, TemporalEncoding
from Utils.grid_utils import Grid
from itertools import zip_longest
import torch.nn.functional as F


def createTrainVal(dataset, ntrain=2000, nval=1000,neval=2000,T_num=1000):

    path = '../TrajData/'
    dir = path + str(dataset) + '_out'
    traj_list = sorted(os.listdir(dir))  
    trajs = []
    cnt = 0



    for traj_filename in tqdm(traj_list, desc='Loading Trajectories'):

        filename = dir + '/' + traj_filename

        with open(filename, 'r') as f:

            lines = f.readlines()
            if len(lines) >=T_num:
                for i in range(len(lines)//T_num):
                    traj = []
                    for line in lines[i*T_num:i*T_num+T_num]:
                        temp = line.strip().split(' ')
                        if len(temp) < 3:
                            continue
                        lon = float(temp[1])
                        lat = float(temp[0])
                        time = int(temp[2])
                        traj.append([lon, lat, time])
                    trajs.append(traj)
                cnt += len(lines)//T_num



    trainsrc =  open(path + str(dataset) + '_TVT/train.pkl','wb')
    valsrc = open(path + str(dataset) + '_TVT/val.pkl','wb')
    testsrc = open(path + str(dataset) + '_TVT/test.pkl','wb')

    pickle.dump(trajs[0:ntrain],trainsrc)
    pickle.dump(trajs[ntrain:ntrain+nval],valsrc)
    pickle.dump(trajs[ntrain+nval:ntrain+nval+neval],testsrc)

    trainsrc.close()
    valsrc.close()
    testsrc.close()

def load_trajs(dataset,filename='train.pkl'):
    path = './TrajData/'
    tvt_dir = path + str(dataset) + '_TVT/'
    with open(tvt_dir+filename, "rb") as f:
        # 使用pickle加载文件中的数据
        trajs = pickle.load(f)
    return trajs
class TrajDataset(Dataset):
    def __init__(self,data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def read_traindataset(dataset,grid, max_length, overlap, s=None, amount=None):
    if s == None:
        trajs = load_trajs(dataset, 'train.pkl')
        trajs2 = load_trajs(dataset, 'val.pkl')
        trajs3 = load_trajs(dataset,'test.pkl')
        trajs = trajs+trajs2+trajs3
    else:
        trajs = load_trajs(dataset,'train.pkl')[s:s+amount]
    return BertDataset(trajs,grid,max_length,overlap)


def read_traintrajs(dataset, s=None, amount=None):
    if s == None:
        trajs = load_trajs(dataset, 'train.pkl')
        trajs2 = load_trajs(dataset, 'val.pkl')
        trajs3 = load_trajs(dataset,'test.pkl')
        trajs = trajs+trajs2+trajs3
    else:
        trajs = load_trajs(dataset,'train.pkl')[s:s+amount]
    return trajs

def read_trajs(path):
    with open(path, "rb") as f:
        trajs = pickle.load(f)
    return trajs



class BertDataset(Dataset):
    def __init__(self,data,grid,max_length,overlap):

        data_lonlat = np.array(data)[:,:,:2]
        data_time = np.array(data)[:,:,2]
        self.data_time = data_time
        self.data = grid.traj2idxseq(data_lonlat)
        self.max_length = max_length
        self.overlap = overlap

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        #mask
        #seg
        #
        seg = []
        segvalid = []
        data_index = []
        seg_time = []
        start = 0
        end=-1
        data= self.data[index]
        time = self.data_time[index]
        while end!=len(data):

            end = min(start + self.max_length, len(data))

            if start == 0:
                valid_start = 0
            else:
                valid_start = self.overlap
            if end == len(data):
                valid_end = end - start
            else:
                valid_end = end - self.overlap
            segvalid.append([valid_start,valid_end])
            seg.append(data[start:end])
            seg_time.append(time[start:end])
            data_index.append(index)
            start += self.max_length - 2 * self.overlap


        return seg,segvalid,seg_time, data_index



class GraphSimpDataset(Dataset):
    def __init__(self, data, params):
        # traj转化成gnnemb，然后分段，
        # init grid
        self.device = params['device']
        print('Init Grid')
        xstep = params['step']
        ystep = params['step']
        grid = Grid(params['minlon'], params['minlat'], params['maxlon'], params['maxlat'], xstep, ystep, params['dataset'],
                    params['minfreq'], params['max_grid_size'], params['k'], params['grid_start'])
        print(f'Grid initialization successful.')
        data_lonlat = np.array(data)[:, :, :2]
        data_time = np.array(data)[:, :, 2]
        self.data_time = data_time
        self.data = grid.traj2idxseq(data_lonlat)
        self.max_length = params['max_length']
        self.overlap =  params['overlap']
        self.pretrain(params['Bert'])
        del self.data, self.data_time, self.max_length, self.overlap
    def __len__(self):
        return len(self.trajs_feature)

    def pretrain(self, params):

        encoding_type = params['encoding_type']
        tbert_num_layers = params['tbert_num_layers']
        tbert_num_heads = params['tbert_num_heads']
        embed_size = params['embed_size']
        max_seq_len = params['max_seq_len']
        hidden_size = embed_size * 4
        device = params['device']
        grid_size = params['grid_size']
        pretrain_model = params['pretain_path']

        encoding_layer = PositionalEncoding(embed_size, max_seq_len)
        if encoding_type == 'temporal':
            encoding_layer = TemporalEncoding(embed_size)
        tbert_embedding = TBERTEmbedding(encoding_layer, embed_size, grid_size)
        tbert_model = TBERT(tbert_embedding, hidden_size, num_layers=tbert_num_layers, num_heads=tbert_num_heads)
        tbert_model.load_state_dict(torch.load(pretrain_model, map_location=device))
        tbert_model = tbert_model.to(device)

        self.trajs_feature = []
        self.trajs_edge_index = []
        self.trajs_point_node_index = []
        self.trajs_seg_node_index = []
        self.trajs_emb = []
        self.trajs_neighbor=[]
        self.amplify_labels = None

        self.pretrain_time = 0

        for i in range(len(self.data)):
            point_emb_list = []
            seg_emb_list = []
            [segs, segs_valid, segs_time, _] = self.segment(i)
            segs = np.transpose(np.array(list(zip_longest(*segs, fillvalue=0))))
            segs_time = np.transpose(np.array(list(zip_longest(*segs_time, fillvalue=0))))
            segs = torch.tensor(segs).long().to(device)
            segs_time = torch.tensor(segs_time).float().to(device)
            start_time = time.time()
            tbert_out = tbert_model(segs, timestamp=segs_time)
            end_time = time.time()
            self.pretrain_time += (end_time - start_time)
            for j in range(segs.size(0)):
                point_emb = tbert_out[j][segs_valid[j][0]:segs_valid[j][1]]
                seg_emb = torch.mean(point_emb, dim=0)
                point_emb_list.append(point_emb)
                seg_emb_list.append(seg_emb)
            traj_point_emb = torch.cat(point_emb_list, dim=0)
            traj_seg_emb = torch.cat(seg_emb_list, dim=0).view(len(seg_emb_list),embed_size)
            traj_emb = torch.mean(traj_point_emb,dim=0)

            sim = F.cosine_similarity(traj_point_emb.unsqueeze(1), traj_point_emb.unsqueeze(0), dim=2)
            value,indices = torch.topk(sim,11,dim=-1)


            feature = torch.cat((traj_point_emb, traj_seg_emb), dim=0)
            point_node_index = torch.tensor(range(traj_point_emb.size(0))).long().to(device)
            seg_node_index = torch.tensor(range(traj_point_emb.size(0), traj_point_emb.size(0) + traj_seg_emb.size(0))).long().to(device)
            edge_index = torch.cartesian_prod(point_node_index, seg_node_index).t().contiguous().to(device)
            self.trajs_neighbor.append(indices[:,1:].T)
            self.trajs_feature.append(feature)
            self.trajs_edge_index.append(edge_index)
            self.trajs_point_node_index.append(point_node_index)
            self.trajs_seg_node_index.append(seg_node_index)

            self.trajs_emb.append(traj_emb)



    def update_simp(self,amplify_labels):

        self.amplify_labels = torch.zeros((len(self.trajs_point_node_index),(self.trajs_point_node_index[0]).size(0))).to(self.device)
        for i,amplify_label in enumerate(amplify_labels):
           for index in amplify_label:
               self.amplify_labels[i][index]=1


    def segment(self, index):
        seg = []
        segvalid = []
        data_index = []
        seg_time = []
        start = 0
        end = -1
        data = self.data[index]
        time = self.data_time[index]
        while end != len(data):

            end = min(start + self.max_length, len(data))

            if start == 0:
                valid_start = 0
            else:
                valid_start = self.overlap
            if end == len(data):
                valid_end = end - start
            else:
                valid_end = end - start - self.overlap
            segvalid.append([valid_start, valid_end])
            seg.append(data[start:end])
            seg_time.append(time[start:end])
            data_index.append(index)
            start += self.max_length - 2 * self.overlap
        return seg, segvalid, seg_time, data_index

    def __getitem__(self, index):
        if self.amplify_labels==None:
            return  self.trajs_feature[index],self.trajs_edge_index[index],self.trajs_point_node_index[index],self.trajs_seg_node_index[index],self.trajs_emb[index],self.trajs_neighbor[index],0
        else:
            return self.trajs_feature[index],self.trajs_edge_index[index],self.trajs_point_node_index[index],self.trajs_seg_node_index[index],self.trajs_emb[index],self.trajs_neighbor[index], self.amplify_labels[index]

class TBERTDataset:
    def __init__(self, bertdataset):
        self.loc_index = []
        self.ts = []
        for i in range(len(bertdataset)):
            [seg,_,seg_time,_]= bertdataset[i]
            self.loc_index+=seg
            self.ts+=seg_time

    def gen_sequence(self,min_len=0,select_days=None, include_delta=False):
        seq_set = []
        for i in range(len(self.loc_index)):
            one_set = [self.loc_index[i], self.ts[i], len(self.ts[i])]
            seq_set.append(one_set)
            # if user_index >= 10: break
        return seq_set

def GraphSimpcollate(batch):


    batch = list(zip(*batch))


    return batch

class DiffuSimpDataset(Dataset):
    def __init__(self,trajs,simp_trajs_idx):
        self.trajs = trajs
        self.simp_trajs_idx = simp_trajs_idx

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, item):
        return self.trajs[item], self.simp_trajs_idx[item]
def DiffuSimpcollate(batch,x_max, y_max, x_min, y_min,device):
    trajs, simp_trajs_idx = list(zip(*batch))
    trajs_emb = []
    for traj in trajs:
        traj_emb = generate_original_features(traj,x_max, y_max, x_min, y_min)
        trajs_emb.append(traj_emb)
    trajs_emb = torch.Tensor(trajs_emb).to(device)
    trajs_padding = pad_sequence(trajs_emb, batch_first=True).to(device)
    trajs_len = torch.LongTensor(list(map(len, trajs))).to(device)
    max_trajs_len = trajs_len.max().item()
    padding_mask = torch.arange(max_trajs_len).to(device)[None, :] >= trajs_len[:, None]


    simp_trajs=[]

    for i in range(len(trajs)):

        simp_traj_idx = np.concatenate([[0],simp_trajs_idx[i].cpu().numpy(),[-1]])
        simp_traj = trajs_emb[i][simp_traj_idx]
        simp_trajs.append(simp_traj)


    simp_trajs_padding = pad_sequence(simp_trajs, batch_first=True).to(device)
    simp_trajs_len = torch.LongTensor(list(map(len, simp_trajs))).to(device)
    max_simp_trajs_len = simp_trajs_len.max().item()
    simp_padding_mask = torch.arange(max_simp_trajs_len).to(device)[None, :] >= simp_trajs_len[:, None]

    labels = []
    labels_mask = []
    for i in range(len(trajs)):
        simp_traj_idx = np.concatenate([[0], simp_trajs_idx[i].cpu().numpy(), [len(trajs[i])-1]])


        labels.append(torch.LongTensor(simp_traj_idx).to(device) )

    labels = torch.stack(labels,dim=0)



    return trajs_padding, padding_mask, simp_trajs_padding, simp_padding_mask,labels,labels_mask


if __name__ == '__main__':
    createTrainVal('Geolife')