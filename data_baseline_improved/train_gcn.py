import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_add_pool
from loss import contrastive_loss
from data_utils import (
    load_id2emb,
    PreprocessedGraphDataset, collate_fn
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool


# =========================================================
# CONFIG
# =========================================================
# Data paths
TRAIN_GRAPHS =  "/content/drive/MyDrive/ALTEGRAD/data/train_graphs.pkl"
VAL_GRAPHS   = "/content/drive/MyDrive/ALTEGRAD/data/validation_graphs.pkl"
TEST_GRAPHS  = "/content/drive/MyDrive/ALTEGRAD/data/test_graphs.pkl"

TRAIN_EMB_CSV = "/content/drive/MyDrive/ALTEGRAD/data/train_embeddings.csv"
VAL_EMB_CSV   = "/content/drive/MyDrive/ALTEGRAD/data/validation_embeddings.csv"

# Training parameters
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# MODEL: GNN to encode graphs (simple GCN, no edge features)
# =========================================================
class MolGNN(nn.Module):
    def __init__(self, hidden=128, out_dim=256, layers=3):
        super().__init__()

        # =========================
        # Node embeddings (9 features)
        # =========================
        node_emb_dim = hidden // 4

        self.emb_atomic_num = nn.Embedding(119, node_emb_dim)
        self.emb_chirality = nn.Embedding(9, node_emb_dim)
        self.emb_degree = nn.Embedding(11, node_emb_dim)
        self.emb_formal_charge = nn.Embedding(12, node_emb_dim)
        self.emb_num_hs = nn.Embedding(9, node_emb_dim)
        self.emb_num_radical = nn.Embedding(5, node_emb_dim)
        self.emb_hybridization = nn.Embedding(8, node_emb_dim)
        self.emb_aromatic = nn.Embedding(2, node_emb_dim)
        self.emb_in_ring = nn.Embedding(2, node_emb_dim)

        self.node_proj = nn.Linear(9 * node_emb_dim, hidden)

        # =========================
        # Edge embeddings (3 features)
        # =========================
        edge_emb_dim = hidden // 4

        self.emb_bond_type = nn.Embedding(22, edge_emb_dim)
        self.emb_stereo = nn.Embedding(6, edge_emb_dim)
        self.emb_conjugated = nn.Embedding(2, edge_emb_dim)

        self.edge_proj = nn.Linear(3 * edge_emb_dim, hidden)

        # =========================
        # GNN layers (GINE)
        # =========================
        self.convs = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden))

        # =========================
        # Graph projection
        # =========================
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, batch):
        x = batch.x          # (N, 9)
        edge_attr = batch.edge_attr  # (E, 3)

        # ---- Node encoding ----
        h = torch.cat([
            self.emb_atomic_num(x[:, 0]),
            self.emb_chirality(x[:, 1]),
            self.emb_degree(x[:, 2]),
            self.emb_formal_charge(x[:, 3]),
            self.emb_num_hs(x[:, 4]),
            self.emb_num_radical(x[:, 5]),
            self.emb_hybridization(x[:, 6]),
            self.emb_aromatic(x[:, 7]),
            self.emb_in_ring(x[:, 8]),
        ], dim=-1)

        h = self.node_proj(h)

        # ---- Edge encoding ----
        e = torch.cat([
            self.emb_bond_type(edge_attr[:, 0]),
            self.emb_stereo(edge_attr[:, 1]),
            self.emb_conjugated(edge_attr[:, 2]),
        ], dim=-1)

        e = self.edge_proj(e)

        # ---- Message passing ----
        for conv in self.convs:
            h = conv(h, batch.edge_index, e)
            h = F.relu(h)

        # ---- Graph pooling ----
        g = global_add_pool(h, batch.batch)
        g = self.proj(g)
        g = F.normalize(g, dim=-1)

        return g


# =========================================================
# Training and Evaluation
# =========================================================
def train_epoch(mol_enc, loader, optimizer, device):
    mol_enc.train()

    total_loss, total = 0.0, 0
    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)

        mol_vec = mol_enc(graphs)
        # mol_vec = F.normalize(mol_vec, dim=-1)
        # txt_vec = F.normalize(text_emb, dim=-1)

        loss = contrastive_loss(mol_vec, text_emb, temperature=0.07)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = graphs.num_graphs
        total_loss += loss.item() * bs
        total += bs

    return total_loss / total


@torch.no_grad()
def eval_retrieval(data_path, emb_dict, mol_enc, device):
    ds = PreprocessedGraphDataset(data_path, emb_dict)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_mol, all_txt = [], []
    for graphs, text_emb in dl:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        all_mol.append(mol_enc(graphs))
        all_txt.append(F.normalize(text_emb, dim=-1))
    all_mol = torch.cat(all_mol, dim=0)
    all_txt = torch.cat(all_txt, dim=0)

    sims = all_txt @ all_mol.t()
    ranks = sims.argsort(dim=-1, descending=True)

    N = all_txt.size(0)
    device = sims.device
    correct = torch.arange(N, device=device)

    pos = (ranks == correct.unsqueeze(1)).nonzero()[:, 1] + 1

    mrr = (1.0 / pos.float()).mean().item()

    results = {"MRR": mrr}

    for k in (1, 5, 10):
        hitk = (pos <= k).float().mean().item()
        results[f"R@{k}"] = hitk
        results[f"Hit@{k}"] = hitk

    return results



# =========================================================
# Main Training Loop
# =========================================================
def main():
    print(f"Device: {DEVICE}")

    train_emb = load_id2emb(TRAIN_EMB_CSV)
    val_emb = load_id2emb(VAL_EMB_CSV) if os.path.exists(VAL_EMB_CSV) else None

    emb_dim = len(next(iter(train_emb.values())))

    if not os.path.exists(TRAIN_GRAPHS):
        print(f"Error: Preprocessed graphs not found at {TRAIN_GRAPHS}")
        print("Please run: python prepare_graph_data.py")
        return
    
    train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    mol_enc = MolGNN(out_dim=emb_dim).to(DEVICE)

    optimizer = torch.optim.Adam(mol_enc.parameters(), lr=LR)

    for ep in range(EPOCHS):
        train_loss = train_epoch(mol_enc, train_dl, optimizer, DEVICE)
        if val_emb is not None and os.path.exists(VAL_GRAPHS):
            val_scores = eval_retrieval(VAL_GRAPHS, val_emb, mol_enc, DEVICE)
        else:
            val_scores = {}
        print(f"Epoch {ep+1}/{EPOCHS} - loss={train_loss:.4f} - val={val_scores}")
    
    model_path = "/content/drive/MyDrive/ALTEGRAD/model_checkpoint.pt"
    torch.save(mol_enc.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
