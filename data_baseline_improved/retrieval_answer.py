import os
import math
import pickle

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils import (
    load_id2emb, load_descriptions_from_graphs, PreprocessedGraphDataset, collate_fn
)

from train_gcn import (
    MolGNN, DEVICE, TRAIN_GRAPHS, TEST_GRAPHS, TRAIN_EMB_CSV
)

# ================================
# Reranking helpers (structure-aware)
# ================================
def graph_signature(g):
    """
    Cheap structural signature for a molecule graph (no RDKit).
    Returns: (num_atoms, num_bonds_approx, bond_density)
    """
    n = int(g.num_nodes)
    e_dir = int(g.edge_index.size(1))          # directed edges in PyG
    m = max(1, e_dir // 2)                     # approximate undirected bonds
    density = m / max(1, n)                    # bonds per atom
    return n, m, density

def structure_similarity(sig1, sig2):
    """
    Similarity in [0,1]. Higher = more structurally similar.
    Uses log-distance for size terms + abs distance for density.
    """
    n1, m1, d1 = sig1
    n2, m2, d2 = sig2

    dn = abs(math.log((n1 + 1) / (n2 + 1)))
    dm = abs(math.log((m1 + 1) / (m2 + 1)))
    dd = abs(d1 - d2)

    dist = dn + 0.7 * dm + 0.4 * dd
    return 1.0 / (1.0 + dist)


@torch.no_grad()
def retrieve_descriptions(model, train_data, test_data, train_emb_dict, device, output_csv,
                         top_k=5, alpha=0.1):
    """
    Retrieval + reranking:
      1) cosine similarity between test molecule embeddings and train text embeddings
      2) take top_k candidates
      3) rerank with: final_score = cosine + alpha * structure_similarity
    """
    # --- Load id->description for train ---
    train_id2desc = load_descriptions_from_graphs(train_data)

    # --- Train text embeddings matrix (ordered by train_ids) ---
    train_ids = list(train_emb_dict.keys())
    train_embs = torch.stack([train_emb_dict[id_] for id_ in train_ids]).to(device)
    train_embs = F.normalize(train_embs, dim=-1)
    print(f"Train set size: {len(train_ids)}")

    # --- Load test graphs ---
    test_ds = PreprocessedGraphDataset(test_data)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    print(f"Test set size: {len(test_ds)}")

    # --- Encode test molecules with the trained GNN ---
    test_mol_embs = []
    test_ids_ordered = []
    for graphs in test_dl:
        graphs = graphs.to(device)
        mol_emb = model(graphs)  # already normalized in your MolGNN forward
        test_mol_embs.append(mol_emb)

        bs = graphs.num_graphs
        start = len(test_ids_ordered)
        test_ids_ordered.extend(test_ds.ids[start:start + bs])

    test_mol_embs = torch.cat(test_mol_embs, dim=0)
    print(f"Encoded {test_mol_embs.size(0)} test molecules")

    # --- Cosine similarities: (N_test, N_train) ---
    similarities = test_mol_embs @ train_embs.t()

    # ================================
    # NEW: build structural signatures for reranking
    # ================================
    # signatures for test graphs in the same order as test_ds.ids
    test_sigs = [graph_signature(g) for g in test_ds.graphs]

    # signatures for train graphs aligned with train_ids order
    with open(train_data, "rb") as f:
        train_graphs = pickle.load(f)
    train_id2sig = {g.id: graph_signature(g) for g in train_graphs}
    train_sigs = [train_id2sig[tid] for tid in train_ids]

    # --- Rerank within top_k ---
    results = []
    for i, test_id in enumerate(test_ids_ordered):
        sims = similarities[i]  # (N_train,)
        topk = sims.topk(top_k)
        topk_idx = topk.indices.tolist()

        best_score = -1e18
        best_train_id = None

        for idx in topk_idx:
            base = sims[idx].item()  # cosine
            ssim = structure_similarity(test_sigs[i], train_sigs[idx])
            score = base + alpha * ssim

            if score > best_score:
                best_score = score
                best_train_id = train_ids[idx]

        retrieved_desc = train_id2desc[best_train_id]
        results.append({"ID": test_id, "description": retrieved_desc})

        if i < 5:
            print(f"\nTest ID {test_id}: selected train ID {best_train_id}")
            print(f"final_score={best_score:.4f} | top_k={top_k} | alpha={alpha}")
            print(f"Description: {retrieved_desc[:150]}...")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\n{'='*80}")
    print(f"Saved {len(results)} retrieved descriptions to: {output_csv}")

    return results_df


def main():
    print(f"Device: {DEVICE}")

    output_csv = "/content/drive/MyDrive/ALTEGRAD/test_retrieved_descriptions.csv"
    model_path = "/content/drive/MyDrive/ALTEGRAD/model_checkpoint.pt"

    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint '{model_path}' not found.")
        print("Please train a model first using train_gcn.py")
        return

    if not os.path.exists(TEST_GRAPHS):
        print(f"Error: Preprocessed graphs not found at {TEST_GRAPHS}")
        return

    train_emb = load_id2emb(TRAIN_EMB_CSV)
    emb_dim = len(next(iter(train_emb.values())))

    model = MolGNN(out_dim=emb_dim).to(DEVICE)
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # You can quickly tune these two:
    # top_k: 5 or 10
    # alpha: 0.05 / 0.15 / 0.30
    retrieve_descriptions(
        model=model,
        train_data=TRAIN_GRAPHS,
        test_data=TEST_GRAPHS,
        train_emb_dict=train_emb,
        device=DEVICE,
        output_csv=output_csv,
        top_k=10,
        alpha=0.15
    )


if __name__ == "__main__":
    main()
