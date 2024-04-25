import argparse
import torch
import os

import numpy as np

import yaml

from Utils.dataset import read_trajs,GraphSimpDataset

from Model.GraphSimp import simp
import Utils.query_utils_val as F

import pickle

import path
import time

def range_query(DB, DB_TREE, SimpDB, simDB_TREE, query1, query2, gt_path,  Xmin,Ymin, Tmin,dataset,q_type,distri):
    RES = F.range_query_operator(DB_TREE, simDB_TREE, query1 + query2, False, Xmin, Ymin, Tmin,dataset,q_type,distri)
    print(f'range query effectiveness ({distri} distribution) f1 = {RES}')

def knn_edr(DB, DB_TREE, sim_DB, simDB_TREE, query1,query2,gt_path,Xmin,Ymin,Tmin,dataset,q_type,distri):
    edr_name_data = f'_knn_query_edr_{distri}'
    if os.path.exists(gt_path + edr_name_data):
        [GroundQuerySet, interval] = pickle.load(open(gt_path + edr_name_data, 'rb'), encoding='bytes')
    else:
        GroundQuerySet, interval = F.knn_edr_query_offline(DB, DB_TREE, query1 + query2, Xmin, Ymin, Tmin)
        pickle.dump([GroundQuerySet, interval], open(gt_path + edr_name_data, 'wb'), protocol=2)

    RES = F.knn_edr_query_online(GroundQuerySet, interval, simDB_TREE, sim_DB, Xmin=Xmin, Ymin=Ymin, Tmin=Tmin,dataset=dataset,q_type=q_type,distri=distri)
    print(f'knn edr query effectiveness ({distri} distribution) f1 = {RES}')


def cluster(DB, DB_TREE, SimpDB, simDB_TREE, query1, query2, gt_path,Xmin,Ymin,Tmin,dataset,q_type,distri):
    clu_name = f'_cluster_query_{distri}'
    if os.path.exists(gt_path + clu_name):
        [traj_clus, query1, query2] = pickle.load(open(gt_path + clu_name, 'rb'), encoding='bytes')
    else:
        traj_clus = F.clustering_offline(DB, DB_TREE, query1 + query2, Xmin, Ymin, Tmin)
        pickle.dump([traj_clus, query1, query2], open(gt_path + clu_name, 'wb'), protocol=2)

    RES = F.clustering_online(traj_clus, SimpDB, simDB_TREE, query1 + query2, Xmin, Ymin, Tmin,dataset,q_type,distri)
    print(f'clustering effectiveness ({distri} distribution) f1 = {RES}')

def join_query(DB, DB_TREE, SimpDB, simDB_TREE, query1, query2, gt_path,  Xmin,Ymin, Tmin,dataset,q_type,distri):
    RES = F.join_query_operator(DB, SimpDB, DB_TREE, simDB_TREE, query1 + query2, Xmin, Ymin, Tmin,dataset=dataset,q_type=q_type,distri=distri)
    print(f'join(similarity) query effectiveness ({distri} distribution) f1 = {RES}')

def validation():
    print('Loading data....')
    with open('Setting.yaml','r') as file:
        params = yaml.safe_load(file)[dataset]
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    device = params['device']
    ad_param =  params['ad_ratio']
    crs = params['cr']
    test_path = './TrajData/'
    test_file_path = test_path + str(dataset) + '_TVT/test.pkl'
    trajs = np.array(read_trajs(test_file_path))

    Xmin = float(params['Xmin'])
    Ymin = float(params['Ymin'])
    Tmin = int(params['Tmin'])

    DB_TREE = F.build_or_load_Rtree(trajs,f'./Val/{dataset}/rtree_simp')
    query_path = f'./Val/{dataset}/{q_type}/adj_query_{distri}'
    if os.path.exists(query_path):
        [query1,query2] = pickle.load(open(query_path, 'rb'), encoding='bytes')
    else:
        if distri =='data':
            DB_DISTRI, ID2Grid, DB_DISTRI_trajID = F.get_distribution_feature_data(trajs,Xmin=Xmin, Ymin=Ymin,Tmin=Tmin,dataset=dataset,q_type=q_type)
            gen_distri, query1, query2 = F.get_query_workload_data(DB_DISTRI)
        if distri =='gau':
            DB_DISTRI, ID2Grid, DB_DISTRI_trajID = F.get_distribution_feature_gau(trajs,Xmin=Xmin, Ymin=Ymin,Tmin=Tmin,dataset=dataset,q_type=q_type)
            gen_distri, query1, query2 = F.get_query_workload_gau(DB_DISTRI)
        pickle.dump([query1, query2], open(query_path, 'wb'), protocol=2)
    adjust = F.range_query_adjust(DB_TREE,query1+query2,Xmin=Xmin, Ymin=Ymin,Tmin=Tmin,dataset=dataset,q_type=q_type,distri=distri)
    del DB_TREE

    graph_test_dataset = GraphSimpDataset(trajs, params)

    adjust_tensor = torch.zeros((len(trajs),len(trajs[0]))).to(device)
    for i in adjust:
        for j in i:
            adjust_tensor[j[0],j[1]]= 1
    adjust_tensor /= adjust_tensor.sum()

    print('Simplifying....')
    simptime_start = time.time()
    trajs_score = simp(graph_test_dataset, params)
    trajs_score /=trajs_score.sum()
    simptime_end = time.time()
    simptime = simptime_end-simptime_start

    final_score = (1-ad_param) * trajs_score+ ad_param * adjust_tensor
    final_score = final_score.cpu().detach().numpy()

    sampletime_start = time.time()
    sum = int(cr * (final_score.shape[0] * final_score.shape[1]))
    mask = np.zeros((final_score.shape[0], final_score.shape[1]), dtype=bool)
    sample_indices = np.random.choice(final_score.shape[0] * final_score.shape[1], size=sum, p=final_score.flatten(),replace=False)
    mask.flat[sample_indices] = True
    mask = mask.reshape((final_score.shape[0], final_score.shape[1]))
    mask[:, 0] = True
    mask[:, -1] = True
    simp_trajs = []
    for i in range(trajs.shape[0]):
        simp_traj = trajs[i][mask[i]].tolist()
        for point in simp_traj:
            point[2] = int(point[2])
        simp_trajs.append(simp_traj)
    sampletime_end = time.time()
    sampletime = sampletime_end - sampletime_start

    pretrain_time = graph_test_dataset.pretrain_time
    print(f'simplification time: {sampletime + simptime + pretrain_time}')
    fname = f"distri_{distri}_cr_{cr}.pickle"
    result_path = path.Path(f'./SimpTraj/{dataset}/{q_type}/{fname}')
    with open(result_path, 'wb') as f:
        pickle.dump(simp_trajs, f)

    print('Testing....')
    DB = read_trajs(test_file_path)
    gt_path = path.Path(f'./Val/{dataset}/{q_type}/test')

    DB_TREE = F.build_Rtree(DB, f'./Val/{dataset}/rtree_test')
    query_path = f'./Val/{dataset}/{q_type}/test_query_{distri}'



    if os.path.exists(query_path):
        [query1,query2] = pickle.load(open(query_path, 'rb'), encoding='bytes')
    else:
        if distri =='data':
            DB_DISTRI, ID2Grid, DB_DISTRI_trajID = F.get_distribution_feature_data(trajs,Xmin=Xmin, Ymin=Ymin,Tmin=Tmin,dataset=dataset,q_type=q_type)
            gen_distri, query1, query2 = F.get_query_workload_data(DB_DISTRI)
        if distri =='gau':
            DB_DISTRI, ID2Grid, DB_DISTRI_trajID = F.get_distribution_feature_gau(trajs,Xmin=Xmin, Ymin=Ymin,Tmin=Tmin,dataset=dataset,q_type=q_type)
            gen_distri, query1, query2 = F.get_query_workload_gau(DB_DISTRI)

        pickle.dump([query1, query2], open(query_path, 'wb'), protocol=2)


    with open(result_path, 'rb') as f:
        SimpDB = pickle.load(f)
    simDB_TREE = F.build_Rtree(SimpDB, f"./Val/{dataset}/{q_type}/simp_rtree_distri_{distri}_cr_{cr}")


    if q_type == 'range':
        range_query(DB, DB_TREE, SimpDB, simDB_TREE, query1, query2, gt_path,  Xmin=Xmin,Ymin=Ymin, Tmin=Tmin,dataset=dataset,q_type=q_type,distri=distri)
    elif q_type == 'knn':
        knn_edr(DB, DB_TREE, SimpDB, simDB_TREE, query1, query2, gt_path, Xmin=Xmin, Ymin=Ymin,Tmin=Tmin,dataset=dataset,q_type=q_type,distri=distri)
    elif q_type == 'cluster':
        cluster(DB, DB_TREE, SimpDB, simDB_TREE, query1, query2, gt_path,Xmin=Xmin, Ymin=Ymin,Tmin=Tmin,dataset=dataset,q_type=q_type,distri=distri)
    elif q_type == 'join':
        join_query(DB, DB_TREE, SimpDB, simDB_TREE, query1, query2, gt_path, Xmin=Xmin, Ymin=Ymin,Tmin=Tmin,dataset=dataset,q_type=q_type,distri=distri)
    del simDB_TREE
    del DB_TREE




if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Test parameters.')

    parser.add_argument('--q_type', choices=['range', 'knn', 'cluster', 'join'], default='knn',help='Type of query')
    parser.add_argument('--dataset', choices=['Geolife'], default='Geolife',help='dataset')
    parser.add_argument('--distri', choices=['data', 'gau'], default='gau',help='Type of query distribution')
    parser.add_argument('--cr', choices=[0.0025,0.003,0.0035,0.004,0.0045,0.01,0.02], default=0.0025, help='Compression ratio')

    args = parser.parse_args()


    dataset = args.dataset
    q_type = args.q_type
    distri = args.distri
    cr = args.cr

    validation()



