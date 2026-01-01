#!/usr/bin/env python3
"""Generate BERT embeddings for molecular descriptions."""

import pickle
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
BORING_PATTERNS = [
    # provenance / dérivation
    "it derives from",
    "it is derived from",
    "derived from",
    "it originates from",

    # rôles biologiques / fonctionnels vagues
    "it has a role as",
    "it plays a role in",
    "it is involved in",
    "it is used as",
    "it functions as",
    "it acts as",

    # relations chimiques génériques
    "it is a conjugate",
    "it is the conjugate",
    "conjugate base of",
    "conjugate acid of",
    "it is a salt of",
    "it is a salt",
    "it is a hydrate",
    "it is an ester of",
    "it is an amide of",
    "it is an ether of",

    # classification / taxonomie
    "it is a member of",
    "it belongs to",
    "it is classified as",
    "it is a type of",
    "it is a kind of",

    # métadonnées biologiques
    "found in",
    "isolated from",
    "obtained from",
    "present in",
    "occurs in nature",
    "naturally occurring",

    # pH / conditions expérimentales (souvent verbeux)
    "at physiological ph",
    "at ph",
    "major species at ph",

    # redondances fréquentes
    "it is a chemical entity",
    "it is a molecular entity",
    "this compound is",
    "this molecule is"
]
import re

def filter_boring(desc, min_len=20):
    # découpe grossière en phrases
    sentences = re.split(r"\.\s+", desc)
    
    kept = []
    for s in sentences:
        s_low = s.lower()
        if any(p in s_low for p in BORING_PATTERNS):
            continue
        if len(s.strip()) < min_len:   # évite les phrases trop courtes
            continue
        kept.append(s.strip())

    return ". ".join(kept)
# Configuration
MAX_TOKEN_LENGTH = 128

# Load BERT model
print("Loading BERT model...")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()
print(f"Model loaded on: {device}")

# Process each split
for split in ['train', 'validation']:
    print(f"\nProcessing {split}...")
    
    # Load graphs from pkl file
    pkl_path = f'/content/drive/MyDrive/ALTEGRAD/data/{split}_graphs.pkl'
    print(f"Loading from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        graphs = pickle.load(f)
    print(f"Loaded {len(graphs)} graphs")
    
    # Generate embeddings
    ids = []
    embeddings = []
    
    for graph in tqdm(graphs, total=len(graphs)):
        # Get description from graph
        description = graph.description
        # description = filter_boring(description)
        
        # Tokenize
        inputs = tokenizer(description, return_tensors='pt', 
                          truncation=True, max_length=MAX_TOKEN_LENGTH, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embedding
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        
        ids.append(graph.id)
        embeddings.append(embedding)
    
    # Save to CSV
    result = pd.DataFrame({
        'ID': ids,
        'embedding': [','.join(map(str, emb)) for emb in embeddings]
    })
    output_path = f'/content/drive/MyDrive/ALTEGRAD/data/{split}_embeddings.csv'
    result.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

print("\nDone!")

