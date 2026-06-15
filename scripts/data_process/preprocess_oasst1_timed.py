"""
Preprocess local OASST1 timestamped traces into SequenceTrieDataTrace data.

This script aligns:
    oasst1_sequence.json
    oasst1_reqs_<model>_unique.json
    oass1_train.csv / oass1_val.csv

The output request order follows the timestamp order already present in
oasst1_sequence.json. Token rows are converted to fixed-size KV block ids.
By default block identity is session-local, so different dialogue_id values do
not share blocks even if their token chunks are identical.
"""

import argparse
import copy
import json
import os
import pickle
from collections import Counter
from typing import Dict, Iterable, List, Tuple


ROLE_MAP = {
    "prompter": "user",
    "assistant": "assistant",
}

SOURCE_TO_OUTPUT_SPLIT = {
    "train": "train",
    "validation": "valid",
}


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_token_rows(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield [int(item) for item in line.split(",") if item]


def generate_request_events(access_sequence: Iterable[dict]) -> List[dict]:
    """
    Reproduce scripts/oasst1.py generate_openai_requests(..., duplicate=False),
    but keep timestamp/dialogue metadata for each emitted request.
    """
    prev_message = None
    events = []
    dialogues: Dict[str, List[dict]] = {}

    for access in access_sequence:
        dialogue_id = access["dialogue_id"]
        if dialogue_id not in dialogues:
            dialogues[dialogue_id] = []

        dialogues[dialogue_id].append({
            "role": ROLE_MAP[access["role"]],
            "content": access["text"],
        })

        this_message = dialogues[dialogue_id]
        if prev_message is None or prev_message != this_message:
            events.append({
                "timestamp": access["timestamp"],
                "dialogue_id": dialogue_id,
                "turn_index": access["turn_index"],
                "role": access["role"],
                "lang": access.get("lang"),
                "messages": copy.deepcopy(this_message),
            })

        prev_message = this_message

    return events


def verify_request_alignment(events: List[dict], request_rows: List[dict], split: str):
    if len(events) != len(request_rows):
        raise ValueError(
            f"{split}: generated {len(events)} request events but request JSON has "
            f"{len(request_rows)} rows"
        )

    for idx, (event, request) in enumerate(zip(events, request_rows)):
        if event["messages"] != request["messages"]:
            raise ValueError(
                f"{split}: request JSON mismatch at row {idx}; "
                "oasst1_sequence.json and request JSON are not aligned"
            )


def chunk_tokens(tokens: List[int], block_token_size: int):
    for block_index in range(0, len(tokens), block_token_size):
        yield block_index // block_token_size, tuple(tokens[block_index:block_index + block_token_size])


def encode_blocks(
    event: dict,
    tokens: List[int],
    token_to_id: Dict[Tuple, int],
    block_token_size: int,
    identity_scope: str,
):
    sequence = []
    for block_index, block_tokens in chunk_tokens(tokens, block_token_size):
        if identity_scope == "session":
            key = (event["dialogue_id"], block_index, block_tokens)
        elif identity_scope == "global":
            key = (block_tokens,)
        else:
            raise ValueError(f"Unknown identity_scope: {identity_scope}")

        dense_id = token_to_id.get(key)
        if dense_id is None:
            dense_id = len(token_to_id) + 1  # reserve 0 for unknown/padding
            token_to_id[key] = dense_id
        sequence.append(dense_id)
    return sequence


def common_prefix_len(left: List[Tuple[int, ...]], right: List[Tuple[int, ...]]) -> int:
    total = 0
    for left_block, right_block in zip(left, right):
        if left_block != right_block:
            break
        total += 1
    return total


def score_token_row_alignment(events: List[dict], rows: List[List[int]], offset: int, event_count: int, block_token_size: int):
    block_rows = [
        [
            tuple(tokens[idx:idx + block_token_size])
            for idx in range(0, len(tokens), block_token_size)
        ]
        for tokens in rows[offset:offset + event_count]
    ]

    by_dialogue: Dict[str, List[int]] = {}
    for idx, event in enumerate(events):
        by_dialogue.setdefault(event["dialogue_id"], []).append(idx)

    pairs_with_common_prefix = 0
    total_common_prefix = 0
    for indices in by_dialogue.values():
        for left_idx, right_idx in zip(indices, indices[1:]):
            common = common_prefix_len(block_rows[left_idx], block_rows[right_idx])
            if common:
                pairs_with_common_prefix += 1
                total_common_prefix += common

    return pairs_with_common_prefix, total_common_prefix


def load_token_rows_for_events(
    csv_path: str,
    events: List[dict],
    split: str,
    block_token_size: int,
):
    rows = list(iter_token_rows(csv_path))
    event_count = len(events)
    dropped_extra_rows = 0
    token_row_offset = 0

    if len(rows) < event_count:
        raise ValueError(
            f"{split}: token CSV has {len(rows)} rows but {event_count} events are required"
        )
    if len(rows) > event_count:
        dropped_extra_rows = len(rows) - event_count
        best_score = None
        best_offset = 0
        for offset in range(dropped_extra_rows + 1):
            score = score_token_row_alignment(
                events,
                rows,
                offset,
                event_count,
                block_token_size,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_offset = offset
        token_row_offset = best_offset
        rows = rows[token_row_offset:token_row_offset + event_count]

    return rows, dropped_extra_rows, token_row_offset


def write_events_jsonl(path: str, events: List[dict], token_counts: List[int], block_counts: List[int]):
    with open(path, "w", encoding="utf-8") as f:
        for event, token_count, block_count in zip(events, token_counts, block_counts):
            row = {
                "timestamp": event["timestamp"],
                "dialogue_id": event["dialogue_id"],
                "turn_index": event["turn_index"],
                "role": event["role"],
                "lang": event.get("lang"),
                "token_count": token_count,
                "block_count": block_count,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_split(
    source_split: str,
    output_split: str,
    access_sequences: dict,
    request_json: dict,
    csv_path: str,
    output_dir: str,
    token_to_id: Dict[Tuple, int],
    block_token_size: int,
    identity_scope: str,
    event_role: str,
    max_events: int = None,
):
    events = generate_request_events(access_sequences[source_split])
    verify_request_alignment(events, request_json[source_split], source_split)
    token_rows, dropped_extra_rows, token_row_offset = load_token_rows_for_events(
        csv_path,
        events,
        source_split,
        block_token_size,
    )

    if event_role != "all":
        keep_indices = [
            idx for idx, event in enumerate(events)
            if event["role"] == event_role
        ]
        events = [events[idx] for idx in keep_indices]
        token_rows = [token_rows[idx] for idx in keep_indices]

    if max_events is not None:
        events = events[:max_events]
        token_rows = token_rows[:max_events]

    sequences = []
    token_counts = []
    block_counts = []
    for event, tokens in zip(events, token_rows):
        sequence = encode_blocks(
            event,
            tokens,
            token_to_id,
            block_token_size,
            identity_scope,
        )
        if not sequence:
            continue
        sequences.append(sequence)
        token_counts.append(len(tokens))
        block_counts.append(len(sequence))

    with open(os.path.join(output_dir, f"{output_split}.pkl"), "wb") as f:
        pickle.dump(sequences, f)

    events_path = os.path.join(output_dir, f"{output_split}_events.jsonl")
    write_events_jsonl(events_path, events, token_counts, block_counts)

    role_counts = Counter(event["role"] for event in events)
    dialogue_ids = {event["dialogue_id"] for event in events}
    timestamps = [event["timestamp"] for event in events]
    return {
        "source_split": source_split,
        "request_count": len(sequences),
        "token_count": sum(token_counts),
        "block_count": sum(block_counts),
        "dialogue_count": len(dialogue_ids),
        "role_counts": dict(role_counts),
        "timestamp_min": min(timestamps) if timestamps else None,
        "timestamp_max": max(timestamps) if timestamps else None,
        "events_path": events_path,
        "dropped_extra_tokenized_rows": dropped_extra_rows,
        "token_row_offset": token_row_offset,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess OASST1 timestamped KV-cache traces")
    parser.add_argument("--input_dir", type=str, default="data/traces")
    parser.add_argument("--output_dir", type=str, default="data/oasst1_timed")
    parser.add_argument("--model_name", type=str, default="DeepSeek-R1-Distill-Qwen-14B")
    parser.add_argument("--sequence_json", type=str, default="oasst1_sequence.json")
    parser.add_argument("--request_json", type=str, default=None)
    parser.add_argument("--train_csv", type=str, default="oass1_train.csv")
    parser.add_argument("--valid_csv", type=str, default="oass1_val.csv")
    parser.add_argument("--source_splits", nargs="+", choices=["train", "validation"],
                        default=["train", "validation"])
    parser.add_argument("--block_token_size", type=int, default=64)
    parser.add_argument("--identity_scope", choices=["session", "global"], default="session")
    parser.add_argument("--event_role", choices=["all", "prompter", "assistant"], default="all")
    parser.add_argument("--max_events_per_split", type=int, default=None)
    args = parser.parse_args()

    request_json = args.request_json
    if request_json is None:
        request_json = f"oasst1_reqs_{args.model_name}_unique.json"

    os.makedirs(args.output_dir, exist_ok=True)

    access_sequences = load_json(os.path.join(args.input_dir, args.sequence_json))
    request_rows = load_json(os.path.join(args.input_dir, request_json))

    csv_paths = {
        "train": os.path.join(args.input_dir, args.train_csv),
        "validation": os.path.join(args.input_dir, args.valid_csv),
    }

    token_to_id: Dict[Tuple, int] = {}
    split_meta = {}
    for source_split in args.source_splits:
        output_split = SOURCE_TO_OUTPUT_SPLIT[source_split]
        split_meta[output_split] = process_split(
            source_split=source_split,
            output_split=output_split,
            access_sequences=access_sequences,
            request_json=request_rows,
            csv_path=csv_paths[source_split],
            output_dir=args.output_dir,
            token_to_id=token_to_id,
            block_token_size=args.block_token_size,
            identity_scope=args.identity_scope,
            event_role=args.event_role,
            max_events=args.max_events_per_split,
        )

    vocab = {
        "vocab_size": len(token_to_id) + 1,
        "unk_id": 0,
        "identity_scope": args.identity_scope,
        "unit": "kv_block",
    }
    with open(os.path.join(args.output_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)

    metadata = {
        "source": {
            "type": "local_oasst1",
            "input_dir": args.input_dir,
            "sequence_json": args.sequence_json,
            "request_json": request_json,
            "train_csv": args.train_csv,
            "valid_csv": args.valid_csv,
            "model_name": args.model_name,
        },
        "event_order": "timestamp",
        "event_role": args.event_role,
        "identity_scope": args.identity_scope,
        "block_size": args.block_token_size,
        "block_token_size": args.block_token_size,
        "vocab_size": len(token_to_id) + 1,
        "splits": split_meta,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"OASST1 timed preprocess: wrote {args.output_dir}")
    print(f"  event_role={args.event_role} identity_scope={args.identity_scope}")
    print(f"  block_token_size={args.block_token_size} vocab_size={len(token_to_id) + 1}")
    for split_name, item in split_meta.items():
        print(
            f"  {split_name}: requests={item['request_count']} "
            f"tokens={item['token_count']} blocks={item['block_count']} "
            f"dialogues={item['dialogue_count']} roles={item['role_counts']}"
        )
        if item["dropped_extra_tokenized_rows"]:
            print(
                f"    dropped_extra_tokenized_rows={item['dropped_extra_tokenized_rows']} "
                f"token_row_offset={item['token_row_offset']}"
            )


if __name__ == "__main__":
    main()
