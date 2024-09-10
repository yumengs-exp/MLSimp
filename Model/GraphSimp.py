import torch
from torch import nn
from Utils.dataset import GraphSimpDataset,GraphSimpcollate
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
# from Simplification import test
class GAT(nn.Module):
    def __init__(self, params):
        super(GAT, self).__init__()
        in_features = params['in_features']
        out_features = params['out_features']
        hidden_features = out_features
        num_heads = params['num_heads']
        self.conv1 = GATConv(in_features, hidden_features, heads=num_heads)
        self.conv2 = GATConv(hidden_features * num_heads, out_features, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        norm = torch.norm(x,p=2,dim=1).unsqueeze(-1)
        return x/norm

    def simp_trajs(self,z,neighbor,rate=0.4):
        align_loss = 0
        for i in range(neighbor.size(0)):
            pos = z[neighbor[i]]

            align_loss += (z - pos).norm(dim=1).pow(2)
        align_loss = align_loss / neighbor.size(0)

        sim = torch.cdist(z, z).pow(2)
        uniform_loss = sim.mul(-2).exp().mean(dim=1).log()

        important = (torch.softmax(align_loss* uniform_loss,dim=0))[1:-1]
        _, top_k_indices = torch.sort(important, descending=True)
        K = int(rate * z.size(0))
        top_k_indices = top_k_indices[:K]
        trajs,_ = torch.sort(top_k_indices)
        return trajs

    def important(self,z,neighbor):
        align_loss = 0
        for i in range(neighbor.size(0)):
            pos = z[neighbor[i]]

            align_loss += (z-pos).norm(dim=1).pow(2)
        align_loss = align_loss/neighbor.size(0)

        sim = torch.cdist(z,z).pow(2)
        uniform_loss = sim.mul(-2).exp().mean(dim=1).log()


        important = (torch.softmax(align_loss* uniform_loss,dim=0))

        return important

    def important_sigmoid(self,z,neighbor):
        align_loss = 0
        for i in range(neighbor.size(0)):
            pos = z[neighbor[i]]
            # a = (z-pos).norm(dim=1)
            align_loss += (z-pos).norm(dim=1).pow(2)
        align_loss = align_loss/neighbor.size(0)

        sim = torch.cdist(z,z).pow(2)
        uniform_loss = sim.mul(-2).exp().mean(dim=1).log()

        important = torch.sigmoid(align_loss* uniform_loss)

        return important

    def loss(self,z,neighbor,amply_labels=None):

        align_loss = 0
        for i in range(neighbor.size(0)):
            pos = z[neighbor[i]]

            align_loss += (z-pos).norm(dim=1).pow(2)
        align_loss = align_loss/neighbor.size(0)

        sim = torch.cdist(z,z).pow(2)
        uniform_loss = sim.mul(-2).exp().mean(dim=1).log()

        important = (torch.softmax(align_loss* uniform_loss,dim=0))
        if amply_labels!=None:
            bce_loss = nn.BCELoss(reduction='mean')
            mutual_loss = bce_loss(important,amply_labels)
        else:
            mutual_loss=0
        _, top_k_indices = torch.sort(important[1:-1], descending=True)
        K=3
        top_k_indices,_ = torch.sort(top_k_indices[:K])
        important_simp = ((z[top_k_indices].sum(dim=0)+z[0]+z[-1]))/(K+2)



        return align_loss,uniform_loss,important_simp,mutual_loss



def train_graphsimp(train_dataset, params,simp_trajs_idx=None,load_model=False):

    device = params['device']


    gat_params = params['GAT']
    model = GAT(gat_params).to(device)
    lr = float(gat_params['lr'])
    wd = gat_params['wd']
    batch_size = gat_params['batch_size']
    num_epoch = gat_params['num_epoch']
    lambda1 = gat_params['lambda1']
    lambda2 = gat_params['lambda2']
    lambda3 = gat_params['lambda3']
    save_path = params['GAT']['save_path']

    if simp_trajs_idx!=None:
        train_dataset.update_simp(simp_trajs_idx)
    if load_model:
        model.load_state_dict(torch.load(save_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=wd)
    dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,collate_fn=GraphSimpcollate)
    cnt = 0
    for epoch in range(num_epoch):
        for batch in dataloader:
            model.train()
            optimizer.zero_grad()
            trajs_feature, trajs_edge_index, trajs_point_node_index, trajs_seg_node_index, trajs_emb, trajs_neighbor, amply_labels = batch

            align_losses = 0
            uniform_losses=0
            mutual_losses=0
            simp_trajs=[]
            for i in range(len(trajs_feature)):
                traj_feature = trajs_feature[i]
                traj_edge_index = trajs_edge_index[i]
                traj_point_node_index = trajs_point_node_index[i]
                traj_neighbor = trajs_neighbor[i]
                if simp_trajs_idx!=None:
                    amply_label = amply_labels[i]
                else:
                    amply_label=None
                traj_point_emb = model(traj_feature,traj_edge_index)
                align_loss, uniform_loss, important_simp,mutual_loss = model.loss(traj_point_emb[traj_point_node_index],traj_neighbor,amply_label )
                align_losses+=align_loss
                uniform_losses+=uniform_loss

                mutual_losses += mutual_loss

                simp_trajs.append(important_simp)
            simp_trajs = torch.stack(simp_trajs)
            simp_trajs_norm = torch.norm(simp_trajs, p=2, dim=1).unsqueeze(-1)
            simp_trajs = simp_trajs/simp_trajs_norm
            trajs_emb = torch.stack(trajs_emb)
            trajs_emb_norm = torch.norm(trajs_emb,p=2,dim=1).unsqueeze(-1)
            trajs_emb = trajs_emb/trajs_emb_norm
            batch_losses = F.mse_loss(simp_trajs@simp_trajs.T,trajs_emb@trajs_emb.T)
            align_losses = (align_losses/len(trajs_feature)).mean()
            uniform_losses = (uniform_losses/len(trajs_feature)).mean()
            if mutual_losses!=0:
                mutual_losses = mutual_losses/len(trajs_feature)
            losses = align_losses + lambda1 * uniform_losses + lambda2 * batch_losses + lambda3 * mutual_losses
            losses.backward()
            optimizer.step()
            if mutual_losses != 0:
                print(f'epoch:{epoch} | loss:{losses.item():.4f} ')
            else:
                print(
                    f'epoch:{epoch} | loss:{losses.item():.4f} ')
            if cnt %300 ==0:

                torch.save(model.state_dict(), save_path)


            cnt += 1



    save_path = params['GAT']['save_path']
    torch.save(model.state_dict(), save_path)
    model.eval()
    simp_trajs = []
    for i in range(len(train_dataset)):
        traj_feature, traj_edge_index, traj_point_node_index, traj_seg_node_index, traj_emb, traj_neighbor,_ = train_dataset[i]
        traj_point_emb = model(traj_feature, traj_edge_index)
        important_simp = model.simp_trajs(traj_point_emb[traj_point_node_index],traj_neighbor)
        simp_trajs.append(important_simp)
    simp_trajs = torch.stack(simp_trajs).detach()
    return simp_trajs


def simp(test_dataset, params):

    device = params['device']
    gat_params = params['GAT']
    model = GAT(gat_params).to(device)
    batch_size = gat_params['batch_size']
    save_path = params['GAT']['save_path']
    params_dict = torch.load(save_path)
    model.load_state_dict(params_dict)

    dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,collate_fn=GraphSimpcollate)
    trajs_score = []
    for batch in dataloader:
        with torch.no_grad():
            trajs_feature, trajs_edge_index, trajs_point_node_index, trajs_seg_node_index, trajs_emb, trajs_neighbor, amply_labels = batch
            for i in range(len(trajs_feature)):
                traj_feature = trajs_feature[i]
                traj_edge_index = trajs_edge_index[i]
                traj_point_node_index = trajs_point_node_index[i]
                traj_neighbor = trajs_neighbor[i]
                traj_point_emb = model(traj_feature,traj_edge_index)
                important = model.important_sigmoid(traj_point_emb[traj_point_node_index],traj_neighbor)

                trajs_score.append(important)

    trajs_score = torch.stack(trajs_score)
    trajs_score_norm = torch.norm(trajs_score, p=2, dim=1).unsqueeze(-1)
    trajs_score = trajs_score/trajs_score_norm
    return trajs_score



