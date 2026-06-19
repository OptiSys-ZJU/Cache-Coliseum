#!/usr/bin/env python3
"""Minimal verification for forward plumbing and candidate-conditioned retrieval."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from cache.trie.trie_algorithms import TrieModelPredictAlgorithm
from model.trie_model.model import TrieParrotModel


def test_forward_candidate_conditioned_retrieval_shapes():
    model = TrieParrotModel(vocab_size=100, node_embed_dim=8, hidden_size=4)
    model.eval()

    microstep_history_memory = torch.zeros(1, 4)
    request_history_memory = torch.zeros(1, 4)
    candidate_states = [
        torch.tensor([[2.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 2.0, 0.0, 0.0]]),
    ]
    lru_features = [
        (1.0, 1.0, 1.0, 1.0, 1.0),
        (2.0, 2.0, 2.0, 2.0, 1.0),
    ]

    with torch.no_grad():
        logits_a, reuse_a = model.forward(
            microstep_history_memory,
            request_history_memory,
            lru_features,
            candidate_states=candidate_states,
            inference=True,
        )
        logits_b, reuse_b = model.forward(
            microstep_history_memory,
            request_history_memory,
            lru_features,
            candidate_states=candidate_states,
            inference=True,
        )

    assert logits_a.shape == (1, 2)
    assert reuse_a.shape == (1, 2)
    assert logits_b.shape == (1, 2)
    assert reuse_b.shape == (1, 2)
    assert torch.allclose(logits_a, logits_b)
    assert torch.allclose(reuse_a, reuse_b)


def test_predict_algorithm_passes_microstep_history_memory_to_model():
    class RecordingModel(TrieParrotModel):
        def __init__(self):
            super().__init__(vocab_size=100, node_embed_dim=16, hidden_size=32)
            self.forward_calls = []

        def forward(
            self,
            microstep_history_memory,
            request_history_memory,
            lru_features,
            candidate_states=None,
            candidate_paths=None,
            inference=True,
        ):
            if isinstance(microstep_history_memory, list):
                microstep_shape = (
                    len(microstep_history_memory),
                ) + tuple(microstep_history_memory[0].shape[1:])
            else:
                microstep_shape = tuple(microstep_history_memory.shape)
            request_len = 0 if request_history_memory is None else len(request_history_memory)
            self.forward_calls.append(
                {
                    "microstep_shape": microstep_shape,
                    "request_len": request_len,
                    "lru_count": len(lru_features),
                    "num_candidates": len(candidate_states) if candidate_states is not None else None,
                    "inference": inference,
                }
            )
            num_candidates = len(candidate_states) if candidate_states is not None else len(candidate_paths)
            logits = torch.arange(num_candidates, dtype=torch.float32).unsqueeze(0)
            reuse = torch.zeros(1, num_candidates, dtype=torch.float32)
            return logits, reuse

    model = RecordingModel()
    model.eval()
    alg = TrieModelPredictAlgorithm(max_node_num=4, model=model)

    alg.access([1, 2, 3])
    alg.access([4, 5, 6])
    alg.access([1, 2, 7])

    assert model.forward_calls, "eviction path should call model.forward"
    last_call = model.forward_calls[-1]
    assert last_call["inference"] is True
    assert last_call["microstep_shape"][0] >= 1
    assert last_call["request_len"] >= 1
    assert last_call["lru_count"] == last_call["num_candidates"]
    assert last_call["num_candidates"] is not None and last_call["num_candidates"] >= 1


if __name__ == "__main__":
    test_forward_candidate_conditioned_retrieval_shapes()
    test_predict_algorithm_passes_microstep_history_memory_to_model()
    print("FORWARD PLUMBING TEST PASSED")
