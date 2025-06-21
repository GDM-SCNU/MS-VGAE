# coding=utf-8
# Author: Jung
# Time: 2025/4/14 21:32


import warnings
warnings.filterwarnings("ignore")

import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from Jung.utils import normalize_adj
from DiffVGAE.evaluation import *
import sys
import networkx as nx
import scipy.sparse as sp
from torch.optim import Adam
import matplotlib.pyplot as plt
import dgl.function as fn
from LLM_VGAE.wavelet_utils import *
from VariationalPG.VGAE.link_predict_utils import mask_test_edges, get_scores
DID = 0
# np.random.seed(826)
# torch.manual_seed(826)
# torch.cuda.manual_seed(826)
# torch.cuda.manual_seed(826)
# torch.cuda.manual_seed_all(826)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch_geometric.nn import APPNP

class Attention(nn.Module):
    def __init__(self, emb_dim, hidden_size= 64): # 64
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(emb_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=False)
        )
    def forward(self, z):
        w = self.project(z)
        self.beta = torch.softmax(w, dim=1)
        return (self.beta * z).sum(1)

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, GraphWaveletLayer):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.propagate = APPNP(K=1, alpha=0)

        self.GraphWaveletLayer = GraphWaveletLayer
        self.attention =  Attention(out_channels)

    def forward(self, x, edge_index, phi, phi_inverse):
        mean = self.linear(x)
        mean = F.normalize(mean, p=2, dim=1)  *  1
        mean = self.propagate(mean, edge_index)

        logstd = self.GraphWaveletLayer(phi, phi_inverse, x, activation=lambda x: x)
        logstd = self.attention(logstd)

        # logstd_low = self.GraphWaveletLayer(phi[0], phi_inverse[0], x, activation=lambda x: x)
        # logstd_mid = self.GraphWaveletLayer(phi[1], phi_inverse[1], x, activation=lambda x: x)
        # logstd_high = self.GraphWaveletLayer(phi[2], phi_inverse[2], x, activation=lambda x: x)
        # logstd = self.attention(torch.stack([logstd_low, logstd_mid, logstd_high], dim=1))

        return mean, logstd

class GraphWaveletLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, ncount):
        super(GraphWaveletLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ncount = ncount
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining diagonal filter matrix (Theta in the paper) and weight matrix.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.diagonal_weight = nn.Parameter(torch.empty(self.ncount))

    def init_parameters(self):
        """
        Initializing the diagonal filter and the weight matrix.
        """
        torch.nn.init.uniform_(self.diagonal_weight, 0.9, 1.1)
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, phi, phi_inverse, feature, activation = F.relu):
        x_ = feature @ self.weight_matrix
        log_std = []
        for i in range(len(phi)):
            x = torch.sparse.mm(phi_inverse[i], x_)
            x = self.diagonal_weight.view(-1, 1) * x
            x = torch.sparse.mm(phi[i], x)
            log_std.append(activation(x))
        return torch.stack(log_std, dim=1)
        # x = torch.sparse.mm(phi_inverse, (feature @ self.weight_matrix))
        # x =self.diagonal_weight.view(-1, 1) * x
        # x = torch.sparse.mm(phi, x)

        # x = phi @ torch.diag(self.diagonal_weight)
        # x = x @ phi_inverse
        # x = x @ (feature @ self.weight_matrix)
        # return activation(x)



class VGAE(nn.Module):
    def __init__(self, graph, adj, hid1_dim):
        super(VGAE, self).__init__()
        self.graph = graph
        self.label = graph.ndata['label']
        self.clusters = len(torch.unique(self.label))
        self.feat = graph.ndata['feat'].to(torch.float32)
        self.feat_dim = self.feat.shape[1]
        self.num_nodes = self.feat.shape[0]
        self.hid1_dim = hid1_dim

        self.adj = torch.from_numpy(adj.A).to(torch.float32) #graph.adjacency_matrix().to_dense()# 无self-loop
        G = nx.from_numpy_array(self.adj.numpy())
        self.norm_L = torch.from_numpy(nx.normalized_laplacian_matrix(G).A).to(torch.float32)
        del G

        self.encoder =  Encoder(self.feat_dim, hid1_dim, GraphWaveletLayer(in_channels=self.feat_dim, out_channels=hid1_dim, ncount=self.num_nodes)) #GraphConvSparse(self.feat_dim, hid1_dim)

        self.adj = self.adj + torch.eye(self.graph.num_nodes()) # with self-loop
        self.norm = self.adj.shape[0] * self.adj.shape[0] / float((self.adj.shape[0] * self.adj.shape[0] - self.adj.sum()) * 2)
        self.pos_weight = float(self.adj.shape[0] * self.adj.shape[0] - self.adj.sum()) / self.adj.sum()

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred


    def forward(self, phi, phi_inverse):

        self.mean, self.logstd =  self.encoder(self.feat, edge_index, phi, phi_inverse)

        gaussian_noise = torch.randn(self.feat.size(0), self.hid1_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        A_pred = self.dot_product_decode(sampled_z)
        return sampled_z, A_pred

def load_data(name):
    r = "D:\\PyCharm_WORK\\MyCode\\Jung\\datasets\\attributed_nets\\"
    with open(r + name + ".pkl", 'rb') as f:
        data = pkl.load(f)
    graph = dgl.from_scipy(data['adj'])
    graph.ndata['feat'] = torch.from_numpy(data['feat'].todense())
    graph.ndata['label'] = torch.from_numpy(data['label'])
    return graph, data['adj']


# def load_data(name):
#     r = "D:\\PyCharm_WORK\\MyCode\\Jung\\datasets\\attributed_nets\\"
#     with open(r + name + ".pkl", 'rb') as f:
#         data = pkl.load(f)
#
#     # link prediction
#     adj = data['adj'] # 无环图
#     adj_orig = adj
#     adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
#     #
#     graph = dgl.from_scipy(adj_train)
#     graph.ndata['feat'] = torch.from_numpy(data['feat'].todense())
#     graph.ndata['label'] = torch.from_numpy(data['label'])
#     return graph, adj_orig, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

# blog 0.4 0.2 VGAE > EPOCH(32) ACC(0.5620) NMI(0.3444) ARI(0.2249) F1(0.5629)

def freq_coeff_loss(alpha, H, tau_low=0.2, tau_high=0.5): # 0.8 0.2
    """
    alpha : [B, 3]  频段注意力系数 (α_low, α_mid, α_high)
    H     : [B]     同质性系数
    """
    # α_low 足够大m
    low_term  = (1-H) * F.relu(tau_low-alpha[:, 0])**2
    #mid_term = F.relu(tau_mid-alpha[:, 1])**2
    # α_high 足够大
    high_term = (H) * F.relu(tau_high-alpha[:, -1])**2
    # 熵正则，防塌缩
    ent_term  = - (alpha * torch.log(alpha + 1e-8)).sum(dim=1)

    loss = low_term  + high_term  + ent_term
    return loss.mean()


def to_torch_coo(scipy_mat, device=None, dtype=torch.float32):
    """
    把 SciPy 稀疏矩阵 (CSR / CSC / COO) 转成 torch.sparse_coo_tensor
    保证 indices 维度满足 [2, nnz]。
    """
    coo = scipy_mat.tocoo()  # → COO 格式
    indices = torch.tensor(
        [coo.row, coo.col],  # shape = (2, nnz)
        dtype=torch.long,  # 必须 long
        device=device
    )
    values = torch.tensor(
        coo.data,
        dtype=dtype,
        device=device
    )
    shape = coo.shape
    return torch.sparse_coo_tensor(indices, values, shape)
if __name__ == "__main__":
    name = "acm"
    graph, adj = load_data(name)
    org_adj = adj
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train


    model = VGAE(graph, adj, 16)

    opt1 = Adam(model.parameters(), lr=0.01) # 0.02 VGAE best >  ACC(0.6879) NMI(0.2868) ARI(0.3008) F1(0.6801)


    weight_mask = model.adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = model.pos_weight

    max_nmi = 0
    max_acc = 0
    max_ari = 0
    max_f1 = 0

    support_t = gabor_chebyshev_filter(org_adj, True, 2)#
    # support_t = wavelet_basis_garbo(dataset = name, adj = adj, s = 5, laplacian_normalize = True, sparse_ness = True, threshold = 0.0001, weight_normalize = True)
    phis = []
    phi_inverses = []
    for wavelet in support_t:
        phi = to_torch_coo(wavelet[0])
        phi_inverse = to_torch_coo(wavelet[1])

        # phi, phi_inverse = torch.FloatTensor(wavelet[0].A), torch.FloatTensor(wavelet[1].A)
        phis.append(phi)
        phi_inverses.append(phi_inverse)

    sp_adj = sp.coo_matrix(adj)
    edge_index = torch.tensor([sp_adj.row, sp_adj.col], dtype=torch.long)
    del sp_adj

    if name == "cora":
        homophily = 0.81
    elif name == "citeseer":
        homophily = 0.74
    elif name == "pubmed":
        homophily = 0.80
    elif name == "acm":
        homophily = 0.82
    elif name == "blogcatalog":
        homophily = 0.40
    elif name == "flickr":
        homophily = 0.24
    elif name == "wisconsin":
        homophily = 0.18
    elif name == "washington":
        homophily = 0.15

    best_pred = 0
    best_auc = 0
    for epoch in range(400):
        opt1.zero_grad()
        model.train()
        z, pred = model(phis, phi_inverses)

        loss = log_lik = model.norm * F.binary_cross_entropy(pred.view(-1), (model.adj).view(-1),
                                                             weight=weight_tensor)

        kl_divergence = 0.5 / pred.size(0) * (
                    1 + 2 * model.logstd - model.mean ** 2 - torch.exp(model.logstd) ** 2).sum(1).mean()

        loss -= kl_divergence

        loss += freq_coeff_loss(model.encoder.attention.beta, homophily)


        loss.backward()
        opt1.step()

        with torch.no_grad():
            val_auc, val_ap = get_scores(org_adj, val_edges, val_edges_false, pred)

            sys.stdout.write(
                'VGAE > EPOCH(%d) val_auc: {%.4f}, val_ap ={%.4f} \n' % (epoch, val_auc, val_ap))
            sys.stdout.flush()

            if best_auc < val_auc:
                best_auc = val_auc
                best_pred = pred


    test_auc, test_ap = get_scores(org_adj, test_edges, test_edges_false, best_pred)
    print(
        'test_auc: {%.4f}, test_ap ={%.4f}'% (test_auc, test_ap))


