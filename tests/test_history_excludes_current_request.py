#!/usr/bin/env python3
"""Verify eviction-time history excludes the current microstep being processed."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from cache.trie.trie_algorithms import TrieModelPredictAlgorithm
from model.trie_model.model import TrieParrotModel


class RecordingModel(TrieParrotModel):
    def __init__(self):
        super().__init__(vocab_size=100, node_embed_dim=16, hidden_size=32, max_attention_history=8)
        self.history_lengths = []

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
            self.history_lengths.append(len(microstep_history_memory))
        elif microstep_history_memory is None:
            self.history_lengths.append(0)
        else:
            self.history_lengths.append(int(microstep_history_memory.shape[0]))
        assert request_history_memory is None or len(request_history_memory) >= 1
        num_candidates = len(candidate_states) if candidate_states is not None else len(candidate_paths)
        assert len(lru_features) == num_candidates
        logits = torch.zeros(1, num_candidates, dtype=torch.float32)
        reuse = torch.zeros(1, num_candidates, dtype=torch.float32)
        return logits, reuse


def test_eviction_history_excludes_current_microstep():
    model = RecordingModel()
    model.eval()
    alg = TrieModelPredictAlgorithm(max_node_num=4, model=model)

    alg.access([1, 2])
    alg.access([3, 4])
    assert list(alg.microstep_history_path_window) == [(1,), (1, 2), (3,), (3, 4)]
    assert len(alg.microstep_history_hidden_states) == 4, "first two requests should populate prefix history"

    alg.access([5, 6])
    assert model.history_lengths, "eviction-time forward should be called"
    assert model.history_lengths == [4, 5], (
        "both evictions for request [5,6] should see only the history that "
        "exists before each microstep"
    )
    assert list(alg.microstep_history_path_window) == [
        (1,),
        (1, 2),
        (3,),
        (3, 4),
        (5,),
        (5, 6),
    ]


if __name__ == "__main__":
    test_eviction_history_excludes_current_microstep()
    print("HISTORY EXCLUDES CURRENT MICROSTEP TEST PASSED")
