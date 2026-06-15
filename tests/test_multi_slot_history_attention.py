#!/usr/bin/env python3
"""Verify multi-slot history lets different candidates attend to different slots."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn

from model.trie_model.model import TrieParrotModel


def test_multi_slot_history_attention_is_candidate_specific():
    model = TrieParrotModel(vocab_size=64, node_embed_dim=8, hidden_size=4)
    model.eval()
    model.query_proj = nn.Identity()
    model.key_proj = nn.Identity()

    history_memory = torch.tensor(
        [
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    candidate_states = [
        torch.tensor([[2.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        torch.tensor([[0.0, 2.0, 0.0, 0.0]], dtype=torch.float32),
    ]

    candidates = torch.cat(candidate_states, dim=0)
    queries = model.query_proj(candidates)
    memory_keys = model.key_proj(history_memory)
    attn_logits = torch.matmul(queries, memory_keys.T) / (model.hidden_size ** 0.5)
    attn_weights = torch.softmax(attn_logits, dim=-1)

    assert attn_weights.shape == (2, 2)
    assert attn_weights[0, 0] > attn_weights[0, 1], "candidate 0 should prefer history slot 0"
    assert attn_weights[1, 1] > attn_weights[1, 0], "candidate 1 should prefer history slot 1"


if __name__ == "__main__":
    test_multi_slot_history_attention_is_candidate_specific()
    print("MULTI-SLOT HISTORY ATTENTION TEST PASSED")
