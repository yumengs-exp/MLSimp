
import torch
from tqdm import tqdm
from torch_geometric.nn import Node2Vec
from Utils.gae_utils import load_data
from torch_geometric.utils import train_test_split_edges
from sklearn.metrics import average_precision_score, roc_auc_score
def node2vec_pretrain(G,embedding_dim,walk_length,context_size,walks_per_node,gridemb_lr,num_workers,gridemb_epochs):
    data = load_data(G)
    num_nodes = G.number_of_nodes()
    edge_index = data.edge_index
    data =   train_test_split_edges(data)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(
        edge_index,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_nodes = num_nodes
    ).to(device)
    max_test_ap=0
    num_workers = 0
    patient = 200
    cnt = 0
    last_loss = 0
    loader = model.loader(batch_size=128, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=gridemb_lr)

    for epoch in range(gridemb_epochs):
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in tqdm(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss = total_loss / len(loader)

        model.eval()
        with torch.no_grad():
            z = model()
            pos_y = z.new_ones(data.test_pos_edge_index.size(1))
            neg_y = z.new_zeros(data.test_neg_edge_index.size(1))

            y = torch.cat([pos_y, neg_y],dim=0)

            pos_pred = torch.sigmoid((z[data.test_pos_edge_index[0]] * z[data.test_pos_edge_index[1]]).sum(dim=1))
            neg_pred = torch.sigmoid((z[data.test_neg_edge_index[0]] * z[data.test_neg_edge_index[1]]).sum(dim=1))
            pred = torch.cat([pos_pred, neg_pred], dim=0)

            y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

            roc = roc_auc_score(y, pred)
            ap = average_precision_score(y, pred)

            if ap > max_test_ap:
                max_test_ap = ap

            if abs(loss  - last_loss) <1e-3 :
                # abs(loss - last_loss) > 1e-3:
                # emb = z.detach().cpu()

                cnt = cnt+1
                last_loss = loss
                if cnt == patient:
                    break
            else:
                cnt = 0
                last_loss = loss

            print (f'patient:{cnt}')


        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Roc: {roc:.4f}, Ap: {ap:.4f}, Max_ap:{max_test_ap:.4f}')
    emb = z.detach().cpu()
    return emb




