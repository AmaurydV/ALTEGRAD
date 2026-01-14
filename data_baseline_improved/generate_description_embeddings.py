"""
Generate dual-encoder Sentence-BERT embeddings for molecular descriptions.
MiniLM + MPNet (concatenated).
"""

# =========================================================
# Text preprocessing
# =========================================================
def truncate(text, max_words):
    words = text.split()
    return " ".join(words[:max_words])

def enrich_and_truncate(graph, max_desc_words=90):
    desc = truncate(graph.description, max_desc_words)

    extras = []

    if graph.num_nodes > 50:
        extras.append("large molecule")
    else:
        extras.append("small molecule")

    if graph.edge_index.size(1) / graph.num_nodes > 1.5:
        extras.append("dense bonds")

    extra = " [Molecule info] " + " ".join(extras)

    return desc + extra

def enrich_description(graph):
    """
    Enrichit la description textuelle avec des informations moléculaires globales.
    Compatible avec BERT / SBERT / GPT-like models.
    """
    desc = graph.description

    extras = []

    # ======================================================
    # Taille du graphe
    # ======================================================
    num_atoms = graph.num_nodes
    num_bonds = graph.edge_index.size(1) // 2  # edges are bidirectional

    extras.append(f"{num_atoms} atoms")
    extras.append(f"{num_bonds} bonds")

    if num_atoms <= 20:
        extras.append("very small molecule")
    elif num_atoms <= 50:
        extras.append("small molecule")
    elif num_atoms <= 100:
        extras.append("medium molecule")
    else:
        extras.append("large molecule")

    # ======================================================
    # Densité du graphe
    # ======================================================
    bond_density = num_bonds / max(num_atoms, 1)

    if bond_density < 1.2:
        extras.append("sparse bonding")
    elif bond_density < 2.0:
        extras.append("moderate bonding")
    else:
        extras.append("dense bonding")

    # ======================================================
    # Atomes (si graph.x existe)
    # ======================================================
    if hasattr(graph, "x") and graph.x is not None:
        # atomic number is usually x[:, 0]
        atomic_nums = graph.x[:, 0]

        mean_atomic_num = atomic_nums.float().mean().item()
        max_atomic_num = atomic_nums.max().item()

        extras.append(f"average atomic number {mean_atomic_num:.1f}")
        extras.append(f"max atomic number {int(max_atomic_num)}")

    # ======================================================
    # Cycles / aromaticité (heuristique)
    # ======================================================
    if num_bonds >= num_atoms:
        extras.append("likely cyclic structure")

    # ======================================================
    # Assemblage final
    # ======================================================
    extra_str = " [Molecule properties] " + ", ".join(extras)

    return desc + extra_str

# =========================================================
# Imports
# =========================================================
import pickle
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# =========================================================
# Configuration
# =========================================================
MODEL_A = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_B = "sentence-transformers/all-mpnet-base-v2"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =========================================================
# Load models
# =========================================================
print("Loading text encoders...")
model_a = SentenceTransformer(MODEL_A, device=device)
model_b = SentenceTransformer(MODEL_B, device=device)
print("Models loaded.")

# =========================================================
# Process splits
# =========================================================
for split in ["train", "validation"]:
    print(f"\nProcessing {split}...")

    pkl_path = f"/content/drive/MyDrive/ALTEGRAD/data/{split}_graphs.pkl"
    with open(pkl_path, "rb") as f:
        graphs = pickle.load(f)

    print(f"Loaded {len(graphs)} graphs")

    descriptions = [enrich_description(g) for g in graphs]
    ids = [g.id for g in graphs]

    print("Encoding with MiniLM...")
    emb_a = model_a.encode(
        descriptions,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print("Encoding with MPNet...")
    emb_b = model_b.encode(
        descriptions,
        batch_size=32,  # MPNet is heavier
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print("Concatenating embeddings...")
    emb_a = torch.from_numpy(emb_a)
    emb_b = torch.from_numpy(emb_b)

    embeddings = torch.cat([emb_a, emb_b], dim=1)
    embeddings = F.normalize(embeddings, dim=-1).numpy()

    df = pd.DataFrame({
        "ID": ids,
        "embedding": [",".join(map(str, e)) for e in embeddings]
    })

    output_path = f"/content/drive/MyDrive/ALTEGRAD/data/{split}_embeddings.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved embeddings to {output_path}")

print("\nDone!")
