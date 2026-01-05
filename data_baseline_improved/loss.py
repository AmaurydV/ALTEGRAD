import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def contrastive_loss(mol_vec, txt_vec, temperature=0.07):
    """
    mol_vec: (B, D)
    txt_vec: (B, D)
    """
    mol_vec = F.normalize(mol_vec, dim=-1)
    txt_vec = F.normalize(txt_vec, dim=-1)

    logits = mol_vec @ txt_vec.T  # (B, B)
    logits = logits / temperature

    labels = torch.arange(logits.size(0), device=logits.device)

    loss_m2t = F.cross_entropy(logits, labels)
    loss_t2m = F.cross_entropy(logits.T, labels)

    return (loss_m2t + loss_t2m) / 2
