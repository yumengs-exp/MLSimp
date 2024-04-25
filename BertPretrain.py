
import pickle
# from julia.api import Julia
# jl = Julia(compiled_modules=False)
# from config import load_params
import torch
import time
import os

import numpy as np
import datetime
import path
import shutil
from logger import get_logger
from Utils.grid_utils import Grid
from Utils.graph_utils import load_G,save_G,graph_constructor
import yaml
from Model.node2vec import node2vec_pretrain
from Utils.dataset import read_traindataset,TBERTDataset
from Model.tbert import TBERT, train_tbert, TBERTEmbedding, PositionalEncoding, TemporalEncoding, MaskedLM
import torch.nn  as nn


if __name__ == '__main__':

    dataset = 'Geolife'


    with open('Setting.yaml','r') as file:
        params = yaml.safe_load(file)[dataset]
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


    #### configure output directory
    model_name  = 'TrajSimp'
    dirname = f'{datetime.datetime.now()}'.replace(' ', '_').replace(':', '.')
    out_dir = path.Path(f'./{params["out_dir"]}/{model_name}_{dataset}_log/{dirname}')

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.makedirs_p()

    ### configure logger
    baselogger = get_logger('base logger', f'{out_dir}/logging.log', params['nostdout'])
    losslogger = get_logger('loss logger', f'{out_dir}/loss.log',  stdout=False)
    resultlogger = get_logger('result logger', f'{out_dir}/result.log', params['nostdout'])

    baselogger.info(params)
    losslogger.info(params)
    resultlogger.info(params)


    #init grid
    print('Init Grid')
    xstep = params['step']
    ystep = params['step']
    grid = Grid(params['minlon'], params['minlat'], params['maxlon'], params['maxlat'], xstep, ystep, dataset, params['minfreq'], params['max_grid_size'], params['k'], params['grid_start'])
    print(f'Grid initialization successful.')



    # init emb
    print('Init grid embbedings...')
    emb_file = './Grid&Graph/' + str(params['dataset']) + '/node2vec_emb_' + str(xstep) + '_' + str(ystep) + '_' + str(params['minfreq']) + '_' + str(params['max_grid_size'])
    if params['load_emb']:
        assert (os.path.exists(emb_file)), ' emb file does not exist'
        grid_emb_matrix = torch.load(emb_file)
    else:
        graph_file = './Grid&Graph/' + params['dataset'] + '/graph_' + str(grid.xstep) + '_' + str(grid.ystep) + '_' + str(params['minfreq']) + '_' + str(params['max_grid_size'])
        if os.path.exists(graph_file):
            G = load_G(graph_file)
        else:
            G = graph_constructor(grid,dataset)
            save_G(G, graph_file)
        grid_emb_matrix = node2vec_pretrain(G,params['gridemb_dim'],params['walk_length'],params['context_size'],params['walks_per_node'],params['gridemb_lr'],params['num_workers'],params['gridemb_epochs'])
        torch.save(grid_emb_matrix, emb_file)
    print(f'Grid\'s embeddings initialization successful.')


    train_dataset = read_traindataset(dataset,grid, params['max_length'], params['overlap'])
    pretrain_dataset = TBERTDataset(train_dataset)
    encoding_type = 'temporal'
    tbert_num_layers =  4
    tbert_num_heads = 8
    tbert_mask_prop =  0.2
    tbert_detach = False
    tbert_objective = 'mlm'
    tbert_static =  False
    embed_size = params['gridemb_dim']
    max_seq_len = params['max_length']
    hidden_size = embed_size *4
    init_param = False
    embed_epoch = 2000
    device = 'cuda:0'

    encoding_layer = PositionalEncoding(embed_size, max_seq_len)
    if encoding_type == 'temporal':
        encoding_layer = TemporalEncoding(embed_size)

    obj_models = [MaskedLM(embed_size, grid.grid_size)]

    obj_models = nn.ModuleList(obj_models)

    tbert_embedding = TBERTEmbedding(encoding_layer, embed_size, grid.grid_size, grid_emb_matrix)
    tbert_model = TBERT(tbert_embedding, hidden_size, num_layers=tbert_num_layers, num_heads=tbert_num_heads,
                      init_param=init_param, detach=tbert_detach)


    pretrain_model = './ModelSave/' + str(params['dataset']) + '/pretrain/Bert_' + str(xstep) + '_' + str(ystep) + '_' + str(params['minfreq']) + '_' + str(params['max_grid_size'])
    pretrain_objmodel = './ModelSave/' + str(params['dataset']) + '/pretrain/BertObj_' + str(xstep) + '_' + str(ystep) + '_' + str(params['minfreq']) + '_' + str(params['max_grid_size'])
    embed_layer = train_tbert(pretrain_dataset, tbert_model, obj_models, mask_prop=tbert_mask_prop,
                             num_epoch=embed_epoch, batch_size=64, device=device, save_path=pretrain_model,obj_save_path = pretrain_objmodel)






