#!/usr/bin/env python3
"""
Preprocess YooChoose dataset for Trie-LSTM cache model.

YooChoose dataset format (yoochoose-clicks.dat):
    session_id, timestamp, item_id, category

Output format:
    - sessions.pkl: List[List[int]] - encoded session sequences
    - vocab.json: {item_id_str: int_id} mapping
"""

import argparse
import csv
import json
import os
import pickle
import random
import sys
from collections import defaultdict
from typing import List, Dict, Tuple


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess YooChoose dataset for Trie-LSTM model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input YooChoose clicks file (e.g., yoochoose-clicks.dat)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output directory for processed files"
    )
    
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=50,
        help="Maximum sequence length (longer sessions will be truncated)"
    )
    
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=2,
        help="Minimum sequence length (shorter sessions will be filtered)"
    )
    
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of data to use for training"
    )
    
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.1,
        help="Ratio of data to use for validation"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def aggregate_sessions(input_path: str, min_seq_len: int, max_seq_len: int) -> List[List[str]]:
    """
    Aggregate click events by session_id and sort by timestamp.
    
    Args:
        input_path: Path to YooChoose clicks file
        min_seq_len: Minimum sequence length to keep
        max_seq_len: Maximum sequence length (truncate if longer)
    
    Returns:
        List of sessions, each session is a list of item_ids (strings)
    """
    print(f"Reading and aggregating sessions from {input_path}...")
    
    # session_id -> list of (timestamp, item_id)
    session_events: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            session_id, timestamp, item_id = row[0], row[1], row[2]
            session_events[session_id].append((timestamp, item_id))
    
    print(f"  Found {len(session_events)} unique sessions")
    
    # Sort by timestamp and extract item sequences
    sessions: List[List[str]] = []
    filtered_short = 0
    truncated = 0
    
    for session_id, events in session_events.items():
        # Sort by timestamp
        events.sort(key=lambda x: x[0])
        item_sequence = [item_id for _, item_id in events]
        
        # Filter short sessions
        if len(item_sequence) < min_seq_len:
            filtered_short += 1
            continue
        
        # Truncate long sessions
        if len(item_sequence) > max_seq_len:
            item_sequence = item_sequence[:max_seq_len]
            truncated += 1
        
        sessions.append(item_sequence)
    
    print(f"  Kept {len(sessions)} sessions")
    print(f"  Filtered {filtered_short} sessions (too short)")
    print(f"  Truncated {truncated} sessions (too long)")
    
    return sessions


class ItemVocabBuilder:
    """Build vocabulary mapping from item_id strings to integer IDs."""
    
    def __init__(self):
        self.item_to_id: Dict[str, int] = {}
        self.id_to_item: Dict[int, str] = {}
        self._next_id = 0
    
    def add_item(self, item_id: str) -> int:
        """Add an item to vocabulary, return its integer ID."""
        if item_id not in self.item_to_id:
            self.item_to_id[item_id] = self._next_id
            self.id_to_item[self._next_id] = item_id
            self._next_id += 1
        return self.item_to_id[item_id]
    
    def encode_sequence(self, sequence: List[str]) -> List[int]:
        """Encode a sequence of item_ids to integer IDs."""
        return [self.add_item(item_id) for item_id in sequence]
    
    def encode_sequences(self, sessions: List[List[str]]) -> List[List[int]]:
        """Encode multiple sessions."""
        return [self.encode_sequence(seq) for seq in sessions]
    
    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self.item_to_id)
    
    def save(self, path: str) -> None:
        """Save vocabulary to JSON file."""
        vocab_data = {
            "item_to_id": self.item_to_id,
            "vocab_size": self.vocab_size
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2)
        print(f"  Saved vocabulary to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ItemVocabBuilder':
        """Load vocabulary from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        instance = cls()
        instance.item_to_id = vocab_data["item_to_id"]
        instance.id_to_item = {v: k for k, v in instance.item_to_id.items()}
        instance._next_id = len(instance.item_to_id)
        return instance


def save_sequences(sequences: List[List[int]], path: str) -> None:
    """Save encoded sequences to pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(sequences, f)
    print(f"  Saved {len(sequences)} sequences to {path}")


def split_data(
    sequences: List[List[int]], 
    train_ratio: float, 
    valid_ratio: float, 
    seed: int
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """Split sequences into train/valid/test sets."""
    random.seed(seed)
    
    # Shuffle sequences
    shuffled = sequences.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    
    train_data = shuffled[:train_end]
    valid_data = shuffled[train_end:valid_end]
    test_data = shuffled[valid_end:]
    
    print(f"  Split: train={len(train_data)}, valid={len(valid_data)}, test={len(test_data)}")
    
    return train_data, valid_data, test_data


def main():
    """Main entry point for preprocessing."""
    args = parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    if args.min_seq_len < 1:
        print("Error: min_seq_len must be at least 1", file=sys.stderr)
        sys.exit(1)
    
    if args.max_seq_len < args.min_seq_len:
        print("Error: max_seq_len must be >= min_seq_len", file=sys.stderr)
        sys.exit(1)
    
    if args.train_ratio + args.valid_ratio >= 1.0:
        print("Error: train_ratio + valid_ratio must be < 1.0", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Preprocessing YooChoose dataset...")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Sequence length: [{args.min_seq_len}, {args.max_seq_len}]")
    print(f"  Split ratios: train={args.train_ratio}, valid={args.valid_ratio}, test={1-args.train_ratio-args.valid_ratio:.2f}")
    
    # Step 1: Aggregate sessions
    print("\nStep 1: Aggregating sessions...")
    sessions = aggregate_sessions(args.input, args.min_seq_len, args.max_seq_len)
    
    # Step 2: Build vocabulary and encode
    print("\nStep 2: Building vocabulary and encoding...")
    vocab = ItemVocabBuilder()
    encoded_sessions = vocab.encode_sequences(sessions)
    print(f"  Vocabulary size: {vocab.vocab_size}")
    
    # Step 3: Split data
    print("\nStep 3: Splitting data...")
    train_data, valid_data, test_data = split_data(
        encoded_sessions, args.train_ratio, args.valid_ratio, args.seed
    )
    
    # Step 4: Save outputs
    print("\nStep 4: Saving outputs...")
    vocab.save(os.path.join(args.output, "vocab.json"))
    save_sequences(train_data, os.path.join(args.output, "train.pkl"))
    save_sequences(valid_data, os.path.join(args.output, "valid.pkl"))
    save_sequences(test_data, os.path.join(args.output, "test.pkl"))
    
    # Save metadata
    metadata = {
        "vocab_size": vocab.vocab_size,
        "num_train": len(train_data),
        "num_valid": len(valid_data),
        "num_test": len(test_data),
        "min_seq_len": args.min_seq_len,
        "max_seq_len": args.max_seq_len,
        "train_ratio": args.train_ratio,
        "valid_ratio": args.valid_ratio,
        "seed": args.seed
    }
    with open(os.path.join(args.output, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {os.path.join(args.output, 'metadata.json')}")
    
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
