from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from epagec import EPAGEC, construct_S_and_M, evaluate
import torch
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os



def main():


    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    # dataset = AttributedGraphDataset(root='/tmp/Wiki', name='Wiki')

    data = dataset[0]
    print(data)
    adjacency = to_dense_adj(data.edge_index)[0]    

    features = data.x                              
    labels = data.y.cpu().numpy()


    # Hyperparameters (tuned for Cora)
    num_clusters = dataset.num_classes
    embedding_dim = 6  #8
    lambda_reg = 0.003
    p = 2
    t = 1
    k_neighbors=10
    epochs = 500
    lr = 0.01
    runs = 50  # Run multiple times for stable metrics

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    adjacency = adjacency.to(device)
    features = features.to(device)

    S, M = construct_S_and_M(adjacency, features, p=p, t=t, k_neighbors=k_neighbors)


    acc_list, nmi_list, f1_list = [], [], []
    for run in range(runs):
        print(f"\n--- Run {run+1}/{runs} ---")
        
        model = EPAGEC(
            num_nodes=features.shape[0],
            num_features=features.shape[1],
            embedding_dim=embedding_dim,
            num_clusters=num_clusters,
            lambda_reg=lambda_reg,
            p=p,
            t=t,
            k_neighbors=k_neighbors
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        best_acc, best_nmi, best_f1 = 0.0, 0.0, 0.0

        cluster_hist = {'acc':{}, 'nmi':{}}

      

        for epoch in range(epochs):
            if epoch < 50:
              warmup_lr = lr * (epoch + 1) / 10
              for param_group in optimizer.param_groups:
                  param_group['lr'] = warmup_lr
            model.train()
            optimizer.zero_grad()
            
            B, G = model(S, M)
            loss = model.compute_loss(M, S, G)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    B, G_val = model(S, M)
                    y_pred = model.predict(G_val).cpu().numpy()
                    acc, nmi, f1 = evaluate(labels, y_pred)
                    if acc > best_acc:
                        best_acc, best_nmi, best_f1 = acc, nmi, f1
                    print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | ACC: {acc:.4f}")
                    cluster_hist['acc'][epoch]=acc; cluster_hist['nmi'][epoch]=nmi

            





        acc_list.append(best_acc)
        nmi_list.append(best_nmi)
        f1_list.append(best_f1)

    # Final results
    print(f"\n=== Final Metrics (Avg of {runs} runs) ===")
    print(f"ACC: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"NMI: {np.mean(nmi_list):.4f} ± {np.std(nmi_list):.4f}")
    print(f"F1:  {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")




if __name__ == "__main__":
    main()