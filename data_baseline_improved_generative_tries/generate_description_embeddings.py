# #!/usr/bin/env python3
# """Generate BERT embeddings for molecular descriptions."""

# import pickle
# import pandas as pd
# import torch
# from transformers import AutoTokenizer, AutoModel
# from tqdm import tqdm

# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output.last_hidden_state
#     mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return (token_embeddings * mask).sum(1) / mask.sum(1)


# # Configuration
# MAX_TOKEN_LENGTH = 128

# # Load BERT model
# print("Loading BERT model...")
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# model = AutoModel.from_pretrained('bert-base-uncased')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)
# model.eval()
# print(f"Model loaded on: {device}")

# # Process each split
# for split in ['train', 'validation']:
#     print(f"\nProcessing {split}...")
    
#     # Load graphs from pkl file
#     pkl_path = f'/content/drive/MyDrive/ALTEGRAD/data/{split}_graphs.pkl'
#     print(f"Loading from {pkl_path}...")
#     with open(pkl_path, 'rb') as f:
#         graphs = pickle.load(f)
#     print(f"Loaded {len(graphs)} graphs")
    
#     # Generate embeddings
#     ids = []
#     embeddings = []
    
#     for graph in tqdm(graphs, total=len(graphs)):
#         # Get description from graph
#         description = graph.description
        
#         # Tokenize
#         inputs = tokenizer(description, return_tensors='pt', 
#                           truncation=True, max_length=MAX_TOKEN_LENGTH, padding=True)
#         inputs = {k: v.to(device) for k, v in inputs.items()}
        
#         # Get embedding
#         with torch.no_grad():
#             outputs = model(**inputs)
#             embedding = mean_pooling(outputs, inputs["attention_mask"])
#             embedding = torch.nn.functional.normalize(embedding, dim=-1)

#         embedding = embedding.cpu().numpy().flatten()
        
#         ids.append(graph.id)
#         embeddings.append(embedding)
    
#     # Save to CSV
#     result = pd.DataFrame({
#         'ID': ids,
#         'embedding': [','.join(map(str, emb)) for emb in embeddings]
#     })
#     output_path = f'/content/drive/MyDrive/ALTEGRAD/data/{split}_embeddings.csv'
#     result.to_csv(output_path, index=False)
#     print(f"Saved to {output_path}")

# print("\nDone!")

#!/usr/bin/env python3






























# """
# Generate Sentence-BERT embeddings for molecular descriptions.
# Replaces BERT CLS embeddings with retrieval-optimized embeddings.
# """

# def truncate(text, max_words):
#     words = text.split()
#     return " ".join(words[:max_words])

# def enrich_and_truncate(graph, max_desc_words=90):
#     desc = truncate(graph.description, max_desc_words)

#     extras = []

#     if graph.num_nodes > 50:
#         extras.append("large molecule")
#     else:
#         extras.append("small molecule")

#     if graph.edge_index.size(1) / graph.num_nodes > 1.5:
#         extras.append("dense bonds")

#     extra = " [Molecule info] " + " ".join(extras)

#     return desc + extra



# import pickle
# import pandas as pd
# import torch
# from sentence_transformers import SentenceTransformer
# from tqdm import tqdm

# # =========================================================
# # Configuration
# # =========================================================
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# # Load Sentence-BERT model
# print("Loading Sentence-BERT model...")
# model = SentenceTransformer(MODEL_NAME, device=device)
# print("Model loaded.")

# # =========================================================
# # Process each split
# # =========================================================
# for split in ["train", "validation"]:
#     print(f"\nProcessing {split}...")

#     pkl_path = f"/content/drive/MyDrive/ALTEGRAD/data/{split}_graphs.pkl"
#     print(f"Loading graphs from {pkl_path}...")

#     with open(pkl_path, "rb") as f:
#         graphs = pickle.load(f)

#     print(f"Loaded {len(graphs)} graphs")

#     ids = []
#     embeddings = []

#     # Encode descriptions
#     descriptions = [enrich_and_truncate(g) for g in graphs]

#     ids = [g.id for g in graphs]

#     print("Encoding descriptions...")
#     embeddings = model.encode(
#         descriptions,
#         batch_size=64,
#         show_progress_bar=True,
#         convert_to_numpy=True,
#         normalize_embeddings=True,  # IMPORTANT for cosine similarity
#     )

#     # Save to CSV (same format as before)
#     df = pd.DataFrame({
#         "ID": ids,
#         "embedding": [",".join(map(str, emb)) for emb in embeddings]
#     })

#     output_path = f"/content/drive/MyDrive/ALTEGRAD/data/{split}_embeddings.csv"
#     df.to_csv(output_path, index=False)
#     print(f"Saved embeddings to {output_path}")

# print("\nDone!")






























#!/usr/bin/env python3
# """
# Generate ChemBERTa embeddings for molecular descriptions
# (mean pooling + L2 normalization).
# """

# import pickle
# import pandas as pd
# import torch
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModel
# from tqdm import tqdm

# # =========================================================
# # Configuration
# # =========================================================
# MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
# MAX_TOKEN_LENGTH = 256
# BATCH_SIZE = 32

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # =========================================================
# # Load model
# # =========================================================
# print("Loading ChemBERTa...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModel.from_pretrained(MODEL_NAME).to(device)
# model.eval()
# print("Model loaded.")

# # =========================================================
# # Mean pooling
# # =========================================================
# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output.last_hidden_state
#     mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return (token_embeddings * mask).sum(1) / mask.sum(1)

# # =========================================================
# # Process splits
# # =========================================================
# for split in ["train", "validation"]:
#     print(f"\nProcessing {split}...")

#     pkl_path = f"/content/drive/MyDrive/ALTEGRAD/data/{split}_graphs.pkl"
#     with open(pkl_path, "rb") as f:
#         graphs = pickle.load(f)

#     descriptions = [g.description for g in graphs]
#     ids = [g.id for g in graphs]

#     all_embeddings = []

#     for i in tqdm(range(0, len(descriptions), BATCH_SIZE)):
#         batch_text = descriptions[i:i+BATCH_SIZE]

#         inputs = tokenizer(
#             batch_text,
#             padding=True,
#             truncation=True,
#             max_length=MAX_TOKEN_LENGTH,
#             return_tensors="pt"
#         ).to(device)

#         with torch.no_grad():
#             outputs = model(**inputs)

#         emb = mean_pooling(outputs, inputs["attention_mask"])
#         emb = F.normalize(emb, dim=-1)  # CRUCIAL
#         all_embeddings.append(emb.cpu())

#     embeddings = torch.cat(all_embeddings, dim=0).numpy()

#     df = pd.DataFrame({
#         "ID": ids,
#         "embedding": [",".join(map(str, e)) for e in embeddings]
#     })

#     output_path = f"/content/drive/MyDrive/ALTEGRAD/data/{split}_embeddings.csv"
#     df.to_csv(output_path, index=False)
#     print(f"Saved embeddings to {output_path}")

# print("\nDone!")


















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
