"""
Preprocess CC Traces / Weka prefix-cache traces into SequenceTrieDataTrace data.

The output format matches data_trace.trie_data_trace.SequenceTrieDataTrace:
    data/<dataset>/train.pkl
    data/<dataset>/valid.pkl
    data/<dataset>/test.pkl
    data/<dataset>/vocab.json
    data/<dataset>/metadata.json
"""

import argparse
import json
import os
import pickle
import random
from typing import Dict, Iterable, List, Tuple


DEFAULT_HF_DATASET = "semianalysisai/cc-traces-weka-042026"


def _json_load_maybe(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def iter_local_records(input_path: str):
    if input_path.endswith(".jsonl"):
        with open(input_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    with open(input_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "traces" in data:
            data = data["traces"]
        else:
            data = [data]

    for record in data:
        yield record


def iter_hf_records(dataset: str, split: str, config: str = None):
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Hugging Face input requires the 'datasets' package. "
            "Install it or pass --input_path for a local JSON/JSONL file."
        ) from exc

    if config:
        ds = load_dataset(dataset, config, split=split)
    else:
        ds = load_dataset(dataset, split=split)

    for record in ds:
        yield record


def extract_requests(record: dict, fallback_trace_id: int) -> Tuple[str, List[List]]:
    trace_id = (
        record.get("trace_id")
        or record.get("traceId")
        or record.get("id")
        or record.get("trace")
        or f"trace_{fallback_trace_id}"
    )

    if "requests" in record:
        requests = _json_load_maybe(record["requests"])
    elif "hash_ids" in record:
        requests = [record]
    else:
        raise ValueError(f"Record {trace_id} has neither 'requests' nor 'hash_ids'")

    sequences = []
    for req in requests:
        req = _json_load_maybe(req)
        hash_ids = (
            req.get("hash_ids")
            or req.get("hashIds")
            or req.get("block_hashes")
            or req.get("blockHashes")
        )
        if hash_ids is None:
            continue
        sequences.append(list(hash_ids))

    return str(trace_id), sequences


def split_trace_indices(
    trace_count: int,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
    shuffle: bool,
):
    indices = list(range(trace_count))
    if shuffle:
        random.Random(seed).shuffle(indices)

    test_count = int(round(trace_count * test_fraction))
    valid_count = int(round(trace_count * valid_fraction))
    test_count = min(test_count, trace_count)
    valid_count = min(valid_count, trace_count - test_count)

    test_indices = set(indices[:test_count])
    valid_indices = set(indices[test_count:test_count + valid_count])
    train_indices = set(indices[test_count + valid_count:])
    return train_indices, valid_indices, test_indices


def encode_traces(
    records: Iterable[dict],
    identity_scope: str,
    max_traces: int = None,
    max_requests_per_trace: int = None,
):
    token_to_id: Dict[str, int] = {}
    traces = []

    def encode_token(trace_id: str, raw_hash) -> int:
        if identity_scope == "trace":
            key = f"{trace_id}:{raw_hash}"
        else:
            key = str(raw_hash)

        dense_id = token_to_id.get(key)
        if dense_id is None:
            dense_id = len(token_to_id) + 1  # reserve 0 for unknown/padding
            token_to_id[key] = dense_id
        return dense_id

    for record_idx, record in enumerate(records):
        if max_traces is not None and len(traces) >= max_traces:
            break

        trace_id, raw_sequences = extract_requests(record, record_idx)
        if max_requests_per_trace is not None:
            raw_sequences = raw_sequences[:max_requests_per_trace]

        encoded_sequences = []
        for sequence in raw_sequences:
            if not sequence:
                continue
            encoded_sequences.append([
                encode_token(trace_id, hash_id)
                for hash_id in sequence
            ])

        if encoded_sequences:
            traces.append((trace_id, encoded_sequences))

    return traces, token_to_id


def write_split(output_dir: str, split_name: str, split_traces):
    sequences = []
    trace_ids = []
    trace_request_counts = []

    for trace_id, trace_sequences in split_traces:
        trace_ids.append(trace_id)
        trace_request_counts.append(len(trace_sequences))
        sequences.extend(trace_sequences)

    with open(os.path.join(output_dir, f"{split_name}.pkl"), "wb") as f:
        pickle.dump(sequences, f)

    return {
        "trace_ids": trace_ids,
        "trace_request_counts": trace_request_counts,
        "request_count": len(sequences),
        "block_count": sum(len(seq) for seq in sequences),
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess CC Traces for trie KV-cache simulation")
    parser.add_argument("--input_path", type=str, default=None,
                        help="Local JSON/JSONL trace file. If omitted, load from Hugging Face.")
    parser.add_argument("--hf_dataset", type=str, default=DEFAULT_HF_DATASET)
    parser.add_argument("--hf_config", type=str, default=None)
    parser.add_argument("--hf_split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="data/cc_weka")
    parser.add_argument("--identity_scope", choices=["trace", "global"], default="trace",
                        help="Use trace scope for CC traces because hash_ids are trace-local.")
    parser.add_argument("--valid_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--max_traces", type=int, default=None)
    parser.add_argument("--max_requests_per_trace", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--no_save_vocab_mapping", action="store_true",
                        help="Save only vocab_size, not the full token_to_id mapping.")
    args = parser.parse_args()

    if args.input_path:
        records = iter_local_records(args.input_path)
        source = {"type": "local", "path": args.input_path}
    else:
        records = iter_hf_records(args.hf_dataset, args.hf_split, args.hf_config)
        source = {
            "type": "huggingface",
            "dataset": args.hf_dataset,
            "config": args.hf_config,
            "split": args.hf_split,
        }

    traces, token_to_id = encode_traces(
        records,
        identity_scope=args.identity_scope,
        max_traces=args.max_traces,
        max_requests_per_trace=args.max_requests_per_trace,
    )
    if not traces:
        raise ValueError("No requests with hash_ids were found")

    train_indices, valid_indices, test_indices = split_trace_indices(
        len(traces),
        args.valid_fraction,
        args.test_fraction,
        args.seed,
        shuffle=not args.no_shuffle,
    )

    split_traces = {
        "train": [trace for idx, trace in enumerate(traces) if idx in train_indices],
        "valid": [trace for idx, trace in enumerate(traces) if idx in valid_indices],
        "test": [trace for idx, trace in enumerate(traces) if idx in test_indices],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    split_meta = {
        name: write_split(args.output_dir, name, items)
        for name, items in split_traces.items()
    }

    vocab = {
        "vocab_size": len(token_to_id) + 1,
        "unk_id": 0,
        "identity_scope": args.identity_scope,
    }

    with open(os.path.join(args.output_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)

    vocab_mapping_path = None
    if not args.no_save_vocab_mapping:
        vocab_mapping_path = os.path.join(args.output_dir, "vocab_mapping.json")
        with open(vocab_mapping_path, "w") as f:
            json.dump({"token_to_id": token_to_id}, f)

    metadata = {
        "source": source,
        "identity_scope": args.identity_scope,
        "block_size": args.block_size,
        "trace_count": len(traces),
        "vocab_size": len(token_to_id) + 1,
        "vocab_mapping_path": vocab_mapping_path,
        "splits": split_meta,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    total_requests = sum(item["request_count"] for item in split_meta.values())
    total_blocks = sum(item["block_count"] for item in split_meta.values())
    print(f"CC preprocess: wrote {args.output_dir}")
    print(f"  traces={len(traces)} requests={total_requests} blocks={total_blocks}")
    print(f"  vocab_size={len(token_to_id) + 1} identity_scope={args.identity_scope}")


if __name__ == "__main__":
    main()
