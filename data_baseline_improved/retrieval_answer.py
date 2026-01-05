import os

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

def lexical_score(a, b):
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    return len(set_a & set_b)

@torch.no_grad()
def retrieve_descriptions(model, train_data, test_data, train_emb_dict, device, output_csv):
    """
    Args:
        model: Trained GNN model
        train_data: Path to train preprocessed graphs
        test_data: Path to test preprocessed graphs
        train_emb_dict: Dictionary mapping train IDs to text embeddings
        device: Device to run on
        output_csv: Path to save retrieved descriptions
    """
    train_id2desc = load_descriptions_from_graphs(train_data)
    
    train_ids = list(train_emb_dict.keys())
    train_embs = torch.stack([train_emb_dict[id_] for id_ in train_ids]).to(device)
    train_embs = F.normalize(train_embs, dim=-1)
    
    print(f"Train set size: {len(train_ids)}")
    
    test_ds = PreprocessedGraphDataset(test_data)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    print(f"Test set size: {len(test_ds)}")
    
    test_mol_embs = []
    test_ids_ordered = []
    for graphs in test_dl:
        graphs = graphs.to(device)
        mol_emb = model(graphs)
        test_mol_embs.append(mol_emb)
        batch_size = graphs.num_graphs
        start_idx = len(test_ids_ordered)
        test_ids_ordered.extend(test_ds.ids[start_idx:start_idx + batch_size])
    
    test_mol_embs = torch.cat(test_mol_embs, dim=0)
    print(f"Encoded {test_mol_embs.size(0)} test molecules")
    
    similarities = test_mol_embs @ train_embs.t()
    
    TOP_K = 5
    alpha = 0.05  # poids BLEU-like (à tuner)

    results = []

    for i, test_id in enumerate(test_ids_ordered):
        sims = similarities[i]
        topk_idx = sims.topk(TOP_K).indices.tolist()

        best_score = -1e9
        best_desc = None

        for idx in topk_idx:
            train_id = train_ids[idx]
            desc = train_id2desc[train_id]

            score = (
                sims[idx].item()
                + alpha * lexical_score(desc, desc)  # cheap BLEU proxy
            )

            if score > best_score:
                best_score = score
                best_desc = desc

        results.append({
            "ID": test_id,
            "description": best_desc
        })

        
        
    
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
    
    retrieve_descriptions(
        model=model,
        train_data=TRAIN_GRAPHS,
        test_data=TEST_GRAPHS,
        train_emb_dict=train_emb,
        device=DEVICE,
        output_csv=output_csv
    )
    


if __name__ == "__main__":
    main()

