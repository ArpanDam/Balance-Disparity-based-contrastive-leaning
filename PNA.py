import pickle
import torch
import random
from torch_geometric.data import Data
import torch.optim as optim
#from torch_geometric.nn import GCNConv
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import PNAConv
import torch.nn.functional as F
import numpy as np
from pytorch_metric_learning.losses import NTXentLoss
import torch_geometric
from torch_geometric.loader import DataLoader
import gc
import psutil
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph
print(psutil.virtual_memory())
print(torch.__version__)
print(torch_geometric.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PNAConv
from torch_geometric.utils import degree as pyg_degree
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
import torch


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, perceptron_hidden_dim, degree):
        super().__init__()

        # Shared PNA encoder
        self.conv1 = PNAConv(
            in_channels, hidden_channels,
            ["sum","mean","min","max","var","std"],
            ["identity","amplification","attenuation","linear","inverse_linear"],
            degree
        )

        self.conv2 = PNAConv(
            hidden_channels, out_channels,
            ["sum","mean","min","max","var","std"],
            ["identity","amplification","attenuation","linear","inverse_linear"],
            degree
        )

        # Projection head
        self.fc1 = torch.nn.Linear(out_channels, perceptron_hidden_dim)
        self.fc2 = torch.nn.Linear(perceptron_hidden_dim, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def project(self, x):
        return self.fc2(F.relu(self.fc1(x)))

    def forward(self, data1, data2):

        # G1
        h1 = self.encode(data1.x, data1.edge_index)

        # G2
        h2 = self.encode(data2.x, data2.edge_index)

        # Projection
        z1 = self.project(h1)
        z2 = self.project(h2)

        return h1, h2, z1, z2
        # return h1, h2
    

class AlphaMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 8),   # input: 4 features → hidden 8
            nn.ReLU(),
            nn.Linear(8, 1),   # output: alpha_u
            nn.Sigmoid()       # keep alpha in [0,1]
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Load data
# -----------------------------
with open("rinfluencer_followers_dict_2hop.pkl", "rb") as f:
    data = pickle.load(f)

with open("author_to_id.pkl", "rb") as f:
    author_to_id = pickle.load(f)

# -----------------------------
# Build edge_index
# -----------------------------
edge_index = []

for influencer, followers in data.items():

    if influencer not in author_to_id:
        continue

    u = author_to_id[influencer]  # influencer

    for follower in followers.keys():

        if follower not in author_to_id:
            continue

        v = author_to_id[follower]  

        # your direction: influencer → follower
        edge_index.append([u, v])

# -----------------------------
# Convert to tensor
# -----------------------------
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

print("edge_index shape:", edge_index.shape)





# -----------------------------
# Load helper
# -----------------------------
def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)

# -----------------------------
# Load required files
# -----------------------------
author_to_id = load_pkl("author_to_id.pkl")
balance_dict = load_pkl("author_balance_from_followers.pkl")
disparity_dict = load_pkl("author_disparity.pkl")
feature_dict = load_pkl("author_feature_vectors_normalized.pkl")  # 🔥 for x

# -----------------------------
# Create node feature matrix (x)
# -----------------------------
num_nodes = len(author_to_id)
num_features = len(next(iter(feature_dict.values())))

x = torch.zeros((num_nodes, num_features), dtype=torch.float)

for node_id, features in feature_dict.items():
    x[node_id] = torch.tensor(features, dtype=torch.float)

print(" Node feature matrix ready:", x.shape)

# -----------------------------
# Convert balance & disparity → tensors
# -----------------------------
balance = torch.zeros(num_nodes, dtype=torch.float)
disparity = torch.zeros(num_nodes, dtype=torch.float)

for author, idx in author_to_id.items():
    balance[idx] = float(balance_dict.get(author, 0.0))
    disparity[idx] = float(disparity_dict.get(author, 0.0))

print(" Balance & disparity ready")

# -----------------------------
# Graph 1: Balance-based
# -----------------------------
def create_G1(edge_index, balance):

    u = edge_index[0]
    v = edge_index[1]

    avg_balance = (balance[u] + balance[v]) / 2

    prob=avg_balance

    mask = torch.rand(prob.size(0), device=edge_index.device) < prob

    return edge_index[:, mask]

# -----------------------------
# Graph 2: Disparity-based
# -----------------------------
def create_G2(edge_index, disparity):

    u = edge_index[0]
    v = edge_index[1]

    avg_disp = (disparity[u] + disparity[v]) / 2

    # prob = torch.clamp(0.3 + 0.7 * (1 - avg_disp), 0.0, 1.0)
    prob=1-avg_disp

    mask = torch.rand(prob.size(0), device=edge_index.device) < prob

    return edge_index[:, mask]


edge_index = edge_index.long()
x = x.float()

# -----------------------------
# Create SINGLE graph
# -----------------------------
data = Data(x=x, edge_index=edge_index)



def contrastive_loss(h1, h2, tau=0.1):

    # h ← normalized embeddings (cosine similarity setup)
    # h = h / ||h||
    h1 = F.normalize(h1, dim=1)
    h2 = F.normalize(h2, dim=1)

    # sim(a, â) = (h_a^G1 · h_â^G2) / τ
    sim = torch.mm(h1, h2.t()) / tau   # [B x B]

    # Positive pairs:
    # sim(h_a^G1, h_a^G2)
    pos = torch.diag(sim)

    # exp(sim(h_a^G1, h_â^G2)/τ)
    exp_sim = torch.exp(sim)

    # Denominator:
    # Σ_{â ∈ V} exp(sim(h_a^G1, h_â^G2)/τ)
    denom = exp_sim.sum(dim=1)

    # L_con(a) = - log ( exp(sim(h_a^G1, h_a^G2)/τ) /
    #                   Σ_{â ∈ V} exp(sim(h_a^G1, h_â^G2)/τ) )
    loss = -torch.log(torch.exp(pos) / denom + 1e-9)

    # Final loss:
    # L_con = (1 / |V|) Σ_a L_con(a)
    return loss.mean()


def adaptive_loss(h1, h2, edge_index, S, balance, disparity,
                  alpha_mlp,
                  tau=0.1):
    device = h1.device
    N = h1.size(0)

    # h ← normalized embeddings
    # h_u = h_u / ||h_u||
    h1 = F.normalize(h1, dim=1)
    h2 = F.normalize(h2, dim=1)

    # -----------------------------
    # Build adjacency list
    # -----------------------------
    adj = [[] for _ in range(N)]
    for i in range(edge_index.size(1)):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        adj[u].append(v)

    total_loss = 0.0
    count = 0

    # -----------------------------
    # Sample nodes
    # -----------------------------
    sampled_nodes = torch.randperm(N, device=device)[:128]

    for u in sampled_nodes:

        neighbors = adj[u]   

        # C(u) = log(1 + |neighbors(u)|)
        citation_count = torch.log1p(
            torch.tensor(float(len(neighbors)), device=device)
        )

        if len(neighbors) == 0:
            continue

        # -----------------------------
        # LIMIT positives
        # -----------------------------
        max_pos = 50
        if len(neighbors) > max_pos:
            perm = torch.randperm(len(neighbors))
            neighbors = [neighbors[i] for i in perm[:max_pos]]

        # -----------------------------
        # similarity
        # -----------------------------
        # sim(u,v) = (h1_u · h2_v) / τ
        sim_u = torch.matmul(h1[u], h2.t()) / tau

        # stability clamp
        sim_u = torch.clamp(sim_u, 0, 1)

        # σ(sim(u,v))
        sim_u_sig = torch.sigmoid(sim_u)

        # -----------------------------
        # POSITIVE LOSS
        # -----------------------------
        pos_vals = sim_u_sig[neighbors]

        # -----------------------------
        # POSITIVE LOSS (FIXED)
        # -----------------------------
        # w_uv ~ random edge weights (approximation of AIS weights)
        w_uv = torch.rand(len(neighbors), device=device)

        # S(u) = B(u) * (1 - D(u))
        bal = balance[u]
        disp = disparity[u]
        fair_weight = bal * (1 - disp)

        # final edge weight:
        # w_uv ← w_uv * S(u)
        w_uv = w_uv * fair_weight

        pos_vals = sim_u_sig[neighbors]

        # L⁺_u = - (1 / |N(u)|) Σ_v [ S(u) * w_uv * log σ(sim(u,v)) ]
        pos_loss = - (w_uv * S[u] * torch.log(pos_vals + 1e-9)).mean()

        # -----------------------------
        # NEGATIVE SAMPLING
        # -----------------------------
        neg_mask = torch.ones(N, dtype=torch.bool, device=device)
        neg_mask[neighbors] = False
        neg_mask[u] = False

        candidate_nodes = torch.where(neg_mask)[0]

        if candidate_nodes.numel() == 0:
            continue

        k = min(100, candidate_nodes.size(0))
        perm = torch.randperm(candidate_nodes.size(0), device=device)
        neg_nodes = candidate_nodes[perm[:k]]

        neg_vals = sim_u_sig[neg_nodes]

        # L⁻_u = - (1 / Z⁻_u) Σ_n log(1 - σ(sim(u,n)))
        neg_loss = - torch.log(1 - neg_vals + 1e-9).mean()

        # -----------------------------
        # Dominance
        # -----------------------------
        # Dom(u) = mean( ReLU(sim(u, ·)) )
        D_u = torch.relu(sim_u).mean().detach()

        # -----------------------------
        # Adaptive weight via MLP
        # -----------------------------
        # x_u = [ B(u), D(u), Dom(u), C(u) ]
        bal = balance[u]
        disp = disparity[u]
        dom = D_u
        cit = citation_count

        alpha_input = torch.stack([bal, disp, dom, cit])  # shape [4]

        # q(u) = MLP(x_u)
        alpha_u = alpha_mlp(alpha_input).squeeze()

        # α(u) = α_max * σ(q(u))   (bounded adaptive penalty)
        # (sigmoid likely applied inside MLP or implicitly learned)

        # -----------------------------
        # Final loss
        # -----------------------------
        # L_adapt(u) = L⁺_u + (1 + α(u)) * L⁻_u
        L_u = pos_loss + (1 + alpha_u) * neg_loss

        total_loss += L_u
        count += 1

    # L_total = (1 / |U|) Σ_u L_adapt(u)
    return total_loss / (count + 1e-9)


# -----------------------------
# 1. Compute Degree Histogram
# -----------------------------
row = edge_index[0]  # influencer → follower

deg = pyg_degree(row, num_nodes=num_nodes, dtype=torch.long)

max_degree = int(deg.max().item())

degree_hist = torch.zeros(max_degree + 1, dtype=torch.long)

for d in deg:
    degree_hist[d] += 1

print("Degree histogram ready:", degree_hist.shape)


# -----------------------------
# 2. Define Model
# -----------------------------
model = Net(
    in_channels=x.shape[1],
    hidden_channels=128,
    out_channels=64,
    perceptron_hidden_dim=64,
    degree=degree_hist
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
alpha_mlp = AlphaMLP().to(device)


# Move to device if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
edge_index = edge_index.to(device)

# -----------------------------
# 3. Optimizer
# -----------------------------

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(alpha_mlp.parameters()),
    lr=0.001
)


# loader = NeighborLoader(
#     data,
#     num_neighbors=[100, 50],   # 2-hop neighbors
#     batch_size=512,
#     shuffle=True
# )

def get_neighbor_subgraph(data, num_nodes=30, num_hops=1):

    # -----------------------------
    # Step 1: Sample seed nodes
    # -----------------------------
    seed_nodes = torch.randperm(data.num_nodes)[:num_nodes]

    # -----------------------------
    # Step 2: k-hop subgraph
    # -----------------------------
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=seed_nodes,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True
    )

    # -----------------------------
    # Step 3: Get features
    # -----------------------------
    x = data.x[subset]

    # -----------------------------
    # Step 4: Return
    # -----------------------------
    sub_data = Data(
        x=x,
        edge_index=edge_index
    )

    return sub_data, subset


# -----------------------------
# 4. Training Loop (500 epochs)
# -----------------------------
num_epochs = 500
loss_history = []



with open("author_special_score.pkl", "rb") as f:
    S_dict = pickle.load(f)

S = torch.zeros(num_nodes)

for author, idx in author_to_id.items():
    S[idx] = float(S_dict.get(author, 0.0))

S = S.to(device)
balance = balance.to(device)
disparity = disparity.to(device)

with open("author_pagerank.pkl", "rb") as f:
    pagerank_dict = pickle.load(f)

# Convert to tensor
pagerank_tensor = torch.zeros(len(author_to_id))
for author, idx in author_to_id.items():
    pagerank_tensor[idx] = pagerank_dict.get(author, 0.0)

for epoch in range(num_epochs):

    model.train()
    total_loss = 0
    total_con_loss = 0
    total_adapt_loss = 0
    for _ in range(10):   # number of mini-batches

        batch, idx = get_neighbor_subgraph(data, num_nodes=30, num_hops=1)
        batch = batch.to(device)

        optimizer.zero_grad()

        # IMPORTANT: use LOCAL balance/disparity
        batch_balance = balance[idx].to(device)
        batch_disparity = disparity[idx].to(device)

        # create G1, G2
        edge_index_G1 = create_G1(batch.edge_index, batch_balance)
        # edge_index_G2 = create_G2(batch.edge_index, batch_disparity)
        edge_index_G2=create_G2(batch.edge_index, batch_disparity)

        data_G1 = Data(x=batch.x, edge_index=edge_index_G1)
        data_G2 = Data(x=batch.x, edge_index=edge_index_G2)

        # Forward
        h1, h2, z1, z2 = model(data_G1, data_G2)

        batch_S = S[idx]

        L_con = contrastive_loss(h1, h2)
        L_adapt = adaptive_loss(h1, h2, batch.edge_index,batch_S, batch_balance, batch_disparity,alpha_mlp)
        # Total Loss = contrastive + lamda * adaptive
        loss = 1*L_con + 0.5 * L_adapt

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_con_loss += L_con.item()
        total_adapt_loss += L_adapt.item()
        # free memory
        del h1, h2, z1, z2
        torch.cuda.empty_cache()

    print(f"Epoch {epoch}: Loss = {total_loss:.8f}")

print("Training finished!")


# -----------------------------
# Eval mode
# -----------------------------
# -----------------------------
# FULL GRAPH (NO SAMPLING)
# -----------------------------
model.eval()

with torch.no_grad():

    # create full G1 and G2
    edge_index_G1_full = create_G1(edge_index, balance)
    edge_index_G2_full = create_G2(edge_index, disparity)

    data_G1_full = Data(x=x, edge_index=edge_index_G1_full).to(device)
    data_G2_full = Data(x=x, edge_index=edge_index_G2_full).to(device)

    # forward on FULL graph
    h1, h2, z1, z2 = model(data_G1_full, data_G2_full)

# Move to CPU
h1 = h1.cpu()
h2 = h2.cpu()
z1 = z1.cpu()
z2 = z2.cpu()

# -----------------------------
# Save separately
# -----------------------------
with open("h1_embeddingsbdz.pkl", "wb") as f:
    pickle.dump(h1, f)

with open("h2_embeddingsbdz.pkl", "wb") as f:
    pickle.dump(h2, f)
'''
with open("z1_embeddingsbdz.pkl", "wb") as f:
    pickle.dump(z1, f)

with open("z2_embeddingsbdz.pkl", "wb") as f:
    pickle.dump(z2, f)'''

print(" All embeddings saved separately")