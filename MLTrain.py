

import torch
import numpy as np
import datetime
import path
from logger import get_logger
import yaml
from Utils.dataset import read_traintrajs,GraphSimpDataset
import shutil
from Model.GraphSimp import train_graphsimp
from Model.DiffuSimp import train_diffusimp

if __name__ == '__main__':

    dataset = 'Geolife'

    with open('Setting.yaml','r') as file:
        params = yaml.safe_load(file)[dataset]
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])


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

    trajs = read_traintrajs(dataset,params['n_train_start'],params['n_train'])

    simp_trajs_idx=None
    diff_trajs_idx=None
    load_diffu=True
    graph_train_dataset = GraphSimpDataset(trajs, params)
    for i in  range(params['mutual_epochs']):

        simp_trajs_idx = train_graphsimp(graph_train_dataset, params, diff_trajs_idx)
        diff_trajs_idx = train_diffusimp(trajs, simp_trajs_idx, params,load_diffu)
        load_diffu=True















