import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_add_pool, GINEConv

from data_utils import (
    load_id2emb,
    PreprocessedGraphDataset, collate_fn
)


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
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def contrastive_loss(graph_emb, text_emb, temperature=0.07):
    # graph_emb, text_emb: [B, D] normalisés
    logits = graph_emb @ text_emb.T / temperature
    labels = torch.arange(len(graph_emb), device=graph_emb.device)
    return F.cross_entropy(logits, labels)

def weak_multi_positive_loss(graph_emb, text_emb, temperature=0.07, sim_threshold=0.85, alpha=0.2):
    logits = graph_emb @ text_emb.T / temperature
    log_probs = F.log_softmax(logits, dim=1)

    with torch.no_grad():
        text_sim = text_emb @ text_emb.T
        pos = (text_sim >= sim_threshold).float()
        pos.fill_diagonal_(1.0)

        # poids: diagonale = 1, autres positifs = alpha
        weights = pos * alpha
        weights.fill_diagonal_(1.0)

        # normalisation en distribution cible
        weights = weights / weights.sum(dim=1, keepdim=True)

    loss = -(weights * log_probs).sum(dim=1).mean()
    return loss

def capped_multi_positive_loss(graph_emb, text_emb, temperature=0.07, m=2):
    logits = graph_emb @ text_emb.T / temperature
    log_probs = F.log_softmax(logits, dim=1)

    with torch.no_grad():
        text_sim = text_emb @ text_emb.T
        text_sim.fill_diagonal_(-1e9)
        topm = text_sim.topk(m, dim=1).indices

        pos_mask = torch.zeros_like(text_sim)
        pos_mask.scatter_(1, topm, 1.0)
        pos_mask.fill_diagonal_(1.0)

        weights = pos_mask / pos_mask.sum(dim=1, keepdim=True)

    loss = -(weights * log_probs).sum(dim=1).mean()
    return loss


import torch
import torch.nn.functional as F

def symmetric_contrastive_loss(graph_emb, text_emb, temperature=0.07):
    logits = graph_emb @ text_emb.T / temperature
    labels = torch.arange(len(graph_emb), device=graph_emb.device)

    loss_g2t = F.cross_entropy(logits, labels)
    loss_t2g = F.cross_entropy(logits.T, labels)

    return (loss_g2t + loss_t2g) / 2



# =========================================================
# MODEL: GNN to encode graphs (simple GCN, no edge features)
# =========================================================
class MolGNN(nn.Module):
    def __init__(self, hidden=64, out_dim=768, layers=4):
        super().__init__()

        # ========= Node embeddings (9 atom features) =========
        self.node_embeddings = nn.ModuleList([
            nn.Embedding(119, hidden),  # atomic_num
            nn.Embedding(4, hidden),    # chirality
            nn.Embedding(11, hidden),   # degree
            nn.Embedding(12, hidden),   # formal_charge
            nn.Embedding(9, hidden),    # num_hs
            nn.Embedding(5, hidden),    # num_radical_electrons
            nn.Embedding(7, hidden),    # hybridization
            nn.Embedding(2, hidden),    # is_aromatic
            nn.Embedding(2, hidden),    # is_in_ring
        ])

        self.node_dim = hidden * 9

        # ========= Edge embeddings (3 bond features) =========
        self.edge_embeddings = nn.ModuleList([
            nn.Embedding(13, hidden),  # bond_type (0–12)
            nn.Embedding(6, hidden),   # stereo
            nn.Embedding(2, hidden),   # is_conjugated
        ])

        self.edge_dim = hidden * 3

        # ========= GINE layers =========
        self.convs = nn.ModuleList()
        for i in range(layers):
            mlp = nn.Sequential(
                nn.Linear(self.node_dim if i == 0 else hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(mlp, edge_dim=self.edge_dim))

        # ========= Projection =========
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, batch):
        # ----- Node feature embedding -----
        x = torch.cat(
            [emb(batch.x[:, i]) for i, emb in enumerate(self.node_embeddings)],
            dim=-1
        )  # [N, hidden*9]

        # ----- Edge feature embedding -----
        edge_attr = torch.cat(
            [emb(batch.edge_attr[:, i]) for i, emb in enumerate(self.edge_embeddings)],
            dim=-1
        )  # [E, hidden*3]

        # ----- Message passing -----
        h = x
        for conv in self.convs:
            h = conv(h, batch.edge_index, edge_attr)
            h = F.relu(h)

        # ----- Graph pooling -----
        g = global_add_pool(h, batch.batch)

        # ----- Projection + normalization -----
        g = self.proj(g)
        return F.normalize(g, dim=-1)



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
        txt_vec = F.normalize(text_emb, dim=-1)

        loss = symmetric_contrastive_loss(
                                mol_vec, txt_vec,
                                
                            )

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
