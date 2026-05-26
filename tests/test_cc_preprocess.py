#!/usr/bin/env python3
"""Tests for CC trace preprocessing helpers."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.data_process.preprocess_cc_traces import encode_traces


records = [
    {
        "trace_id": "trace_a",
        "requests": [
            {"hash_ids": ["h1", "h2"]},
            {"hash_ids": ["h1", "h3"]},
        ],
    },
    {
        "trace_id": "trace_b",
        "requests": [
            {"hash_ids": ["h1", "h2"]},
        ],
    },
]

trace_scoped, trace_vocab = encode_traces(records, identity_scope="trace")
assert trace_scoped[0][1][0][0] != trace_scoped[1][1][0][0]
assert len(trace_vocab) == 5

global_scoped, global_vocab = encode_traces(records, identity_scope="global")
assert global_scoped[0][1][0][0] == global_scoped[1][1][0][0]
assert len(global_vocab) == 3

print("CC preprocess tests passed")
