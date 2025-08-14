import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score

class EPAGEC(nn.Module):
    def __init__(self, num_nodes, num_features, embedding_dim, num_clusters, 
                 lambda_reg, p, t, k_neighbors, pr_damping=0.85, pr_iter=50):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_clusters = num_clusters
        self.embedding_dim = embedding_dim
        self.lambda_reg = lambda_reg
        self.p = p
        self.t = t
        self.k_neighbors = k_neighbors
        self.pr_damping = pr_damping
        self.pr_iter = pr_iter

        self.B = nn.Parameter(torch.empty(num_nodes, embedding_dim))
        self.Q = nn.Parameter(torch.empty(num_features, embedding_dim))
        self.Z = nn.Parameter(torch.empty(num_clusters, embedding_dim))

        self._init_B = False
        self._init_Z = False

        nn.init.orthogonal_(self.Q)
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.Z)

    def _pagerank(self, adj, personalization=None):
        device = adj.device
        N = adj.shape[0]
        out_degree = adj.sum(dim=1).clamp(min=1e-12)
        M = adj / out_degree.unsqueeze(1)

        if personalization is None:
            p_vec = torch.ones(N, device=device) / N
        else:
            p_vec = personalization / personalization.sum()

        pr = torch.ones(N, device=device) / N
        for _ in range(self.pr_iter):
            pr = (1 - self.pr_damping) * p_vec + self.pr_damping * (M @ pr)
        return pr

    def _initialize_B(self, S):
        device = S.device
        dtype = S.dtype
        try:
            U, _, _ = torch.linalg.svd(S, full_matrices=False)
            self.B.data = U[:, :self.embedding_dim]
        except Exception:
            S_cpu = S.detach().cpu().double().numpy()
            U_np, _, _ = np.linalg.svd(S_cpu, full_matrices=False)
            U_tensor = torch.from_numpy(U_np[:, :self.embedding_dim]).to(dtype).to(device)
            self.B.data = U_tensor
        self._init_B = True

    def _initialize_Z(self, S, features):
        device = S.device
        pr_scores = self._pagerank(S, personalization=features.sum(dim=1))
        _, proto_indices = torch.topk(pr_scores, self.num_clusters)
        proto_features = self.B[proto_indices]

        distances = torch.cdist(self.B, proto_features)
        labels = torch.argmin(distances, dim=1)
        G0 = F.one_hot(labels, self.num_clusters).float()

        GtG = G0.T @ G0 + 1e-8 * torch.eye(self.num_clusters, device=device)
        self.Z.data = torch.linalg.solve(GtG, G0.T @ S @ self.B)
        self._init_Z = True

    def forward(self, adjacency, features, num_iterations=10,temp=0.1):
        S, M = construct_S_and_M(adjacency, features, self.t, self.p, self.k_neighbors)
        S = S.to(adjacency.device)

        if not self._init_B:
            self._initialize_B(S)
        if not self._init_Z:
            self._initialize_Z(S, features)

        for _ in range(num_iterations):
            scores = S @ self.B @ self.Z.T

            G_soft = F.softmax(scores, dim=1)          
            hard_labels = scores.argmax(dim=1, keepdim=True)
            G_hard   = torch.zeros_like(scores).scatter_(1, hard_labels, 1.0)
            G = G_hard + (G_soft - G_soft.detach())

            
            U1, _, Vt1 = torch.linalg.svd( G.T @ S @ self.B, full_matrices=False)
            Z= U1 @ Vt1

            combined = M @ self.Q + self.lambda_reg * (S @ G) @ Z
            device = combined.device
            dtype = combined.dtype
            try:
                U, _, Vt = torch.linalg.svd(combined, full_matrices=False)
                B_new = U @ Vt

                self.B.data=B_new
            except Exception:
                comb_cpu = combined.detach().cpu().double().numpy()
                U_np, _, Vt_np = np.linalg.svd(comb_cpu, full_matrices=False)
                U_tensor = torch.from_numpy(U_np).to(dtype).to(device)
                Vt_tensor = torch.from_numpy(Vt_np).to(dtype).to(device)
                self.B.data = U_tensor @ Vt_tensor

        return self.B, G

    def compute_loss(self, M, S, G):
        term1 = torch.norm(M - self.B @ self.Q.T, p='fro') ** 2
        term2 = torch.norm(S - G @ self.Z @ self.B.T, p='fro') ** 2



        return term1 + self.lambda_reg * term2 

    def predict(self, G, adjacency=None, features=None):
        y = torch.argmax(G, dim=1)
        if adjacency is not None and features is not None:
            S, _ = construct_S_and_M(adjacency, features, self.t, self.p, self.k_neighbors)
            S = S.to(G.device)
            oh = F.one_hot(y, num_classes=self.num_clusters).float()
            smoothed = S @ oh
            y = torch.argmax(smoothed, dim=1)
        return y


def construct_S_and_M(adjacency, features, t, p, k_neighbors):
    device = adjacency.device
    n = adjacency.shape[0]
    feats = bm25_transform(features)
    feats = F.normalize(feats, p=2, dim=1)
    A = adjacency.to_dense() + torch.eye(n, device=device)

    row_sum = A.sum(dim=1).clamp(min=1e-12)
    D_inv = torch.diag(1.0 / row_sum)
    Wnet = 0.5 * (D_inv @ A + torch.eye(n, device=device))

    cos_sim = feats @ feats.T
    topk_vals, topk_idx = torch.topk(cos_sim, k=k_neighbors, dim=1)

    mask = torch.zeros_like(cos_sim)
    mask.scatter_(1, topk_idx, 1.0)
    mutual = (mask * mask.T) > 0
    Wx = cos_sim * mutual.float()

    deg = Wx.sum(dim=1).clamp(min=1e-12)
    deg_inv_sqrt = deg.pow(-0.5)
    Wx = deg_inv_sqrt.unsqueeze(1) * Wx * deg_inv_sqrt.unsqueeze(0)

    S = Wnet + Wx

    
    W_p = torch.linalg.matrix_power(Wnet, p)    # W_p = Wnet^p

    M = W_p @ feats
    return S, M


def bm25_transform(features, k1=1.2, b=0.75):
    N = features.shape[0]
    df = (features > 0).sum(dim=0).float()
    idf = torch.log((N - df + 0.5) / (df + 0.5) + 1.0)

    doc_lengths = features.sum(dim=1).unsqueeze(1)
    avgdl = doc_lengths.mean()

    tf = features.float()
    denominator = tf + k1 * (1 - b + b * (doc_lengths / avgdl))
    tf_component = (tf * (k1 + 1)) / (denominator + 1e-8)

    bm25 = tf_component * idf
    return F.normalize(bm25, p=2, dim=1)

def tfidf_transform(features):
    N = features.shape[0]
    df = (features > 0).sum(dim=0).float()
    idf = torch.log(N / (df + 1e-8))  
    tf = features.float()
    tfidf = tf * idf
    return F.normalize(tfidf, p=2, dim=1)


def evaluate(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    acc = w[row_ind, col_ind].sum() / y_pred.size
    nmi = normalized_mutual_info_score(y_true, y_pred)
    return acc, nmi




