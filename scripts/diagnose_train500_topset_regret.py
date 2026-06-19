"""Online top-set/regret diagnostics for the train500 Trie-PARROT rollout.

This script intentionally mirrors TrieModelPredictAlgorithm.access() instead of
using saved training snapshots, because the question is about online policy
behavior: which leaf the model actually evicts, and whether that action is
better or worse than LRU under the same live candidate set.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cache.trie.oracle import PrefixFutureOracle
from cache.trie.trie_algorithms import TrieModelPredictAlgorithm, TrieNode
from data_trace.trie_data_trace import SequenceTrieDataTrace
from model.trie_model.model import TrieParrotModel


def load_sequences(data_path: str, vocab_path: str) -> List[List[int]]:
    with SequenceTrieDataTrace(data_path, vocab_path) as trace:
        return list(trace.iter_sequences())


def load_model(config_path: str, checkpoint_path: str, device: torch.device) -> TrieParrotModel:
    with open(config_path, "r") as f:
        config = json.load(f)

    model = TrieParrotModel(
        vocab_size=config["vocab_size"],
        node_embed_dim=config.get("node_embed_dim", 64),
        hidden_size=config.get("hidden_size", 128),
        max_attention_history=config.get("max_attention_history", 30),
        max_request_history=config.get("max_request_history"),
        max_microstep_history=config.get("max_microstep_history"),
        lru_feature_dim=config.get("lru_feature_dim", 5),
        ranking_loss_weight=config.get("ranking_loss_weight", 1.0),
        reuse_loss_weight=config.get("reuse_loss_weight", 0.1),
        ce_loss_weight=config.get("ce_loss_weight", 0.0),
        ce_target_policy=config.get("ce_target_policy", "argmax"),
        reuse_distance_log_cap=config.get("reuse_distance_log_cap", 5.0),
        ndcg_alpha=config.get("ndcg_alpha", 10.0),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def transformed_relevance(model: TrieParrotModel, distance: float) -> float:
    cap = float(model.reuse_distance_log_cap)
    if not math.isfinite(distance):
        return cap
    if distance <= 1:
        return 0.0
    return min(math.log10(distance), cap)


def finite_percentile(values: List[float], q: float) -> Optional[float]:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    pos = (len(finite) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return finite[lo]
    return finite[lo] * (hi - pos) + finite[hi] * (pos - lo)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def inf_aware_delta(chosen: float, best: float) -> float:
    """Regret in eviction-distance space: best - chosen, with inf ties handled."""
    if math.isinf(best) and math.isinf(chosen):
        return 0.0
    if math.isinf(best):
        return float("inf")
    if math.isinf(chosen):
        return float("-inf")
    return best - chosen


def compare_distance(left: float, right: float) -> int:
    """Return 1 when left is farther/better, -1 when worse, 0 when tied."""
    if math.isinf(left) and math.isinf(right):
        return 0
    if math.isinf(left):
        return 1
    if math.isinf(right):
        return -1
    if left > right:
        return 1
    if left < right:
        return -1
    return 0


def bucket_candidate_count(n: int) -> str:
    if n <= 8:
        return "001-008"
    if n <= 16:
        return "009-016"
    if n <= 32:
        return "017-032"
    if n <= 64:
        return "033-064"
    if n <= 128:
        return "065-128"
    return "129+"


def bucket_history_len(n: int, maxlen: int) -> str:
    if n == 0:
        return "00"
    if n < maxlen:
        return f"01-{maxlen - 1:02d}"
    return f"{maxlen:02d}+"


def bucket_step_index(n: int) -> str:
    if n <= 3:
        return "00-03"
    if n <= 7:
        return "04-07"
    if n <= 15:
        return "08-15"
    return "16+"


def candidate_depth(path: Tuple[int, ...]) -> int:
    return len(path)


class OnlineTopsetRegretDiagnostic:
    def __init__(
        self,
        model: TrieParrotModel,
        sequences: List[List[int]],
        max_node_num: int,
        device: torch.device,
    ):
        self.model = model
        self.sequences = sequences
        self.max_node_num = max_node_num
        self.device = device
        self.alg = TrieModelPredictAlgorithm(max_node_num=max_node_num, model=model)
        self.oracle = PrefixFutureOracle(sequences, max_prefix_len=max_node_num)
        self.rows: List[Dict[str, object]] = []
        self.total_blocks = 0
        self.hit_blocks = 0
        self.miss_blocks = 0
        self.request_full_hits = 0
        self.previously_exposed_leaf_ids = set()

    def run(self, max_requests: int) -> List[Dict[str, object]]:
        request_limit = min(max_requests, len(self.sequences))
        for request_idx, sequence in enumerate(self.sequences[:request_limit]):
            self.access(request_idx, sequence)
        return self.rows

    def access(self, request_idx: int, sequence: List[int]) -> None:
        cache_sequence = sequence[: self.max_node_num]
        self.oracle.consume_current(cache_sequence, request_idx)

        self.alg.counter += 1
        this_node = self.alg.root_node
        hit_nodes = 0
        current_prefix: List[int] = []

        for step_index, node_id in enumerate(cache_sequence):
            current_prefix.append(node_id)
            if node_id in this_node.children:
                this_node = this_node.children[node_id]
                self.alg.__visit_node__(this_node)
                hit_nodes += 1
            else:
                evict_num = self.alg.cur_node_num + 1 - self.alg.max_node_num
                if evict_num > 0:
                    self.evict_and_record(
                        request_idx=request_idx,
                        step_index=step_index,
                        evict_num=evict_num,
                        this_node=this_node,
                        current_path=current_prefix,
                    )

                new_node = TrieNode()
                new_node.key = node_id
                new_node.node_id = node_id
                new_node.parent = this_node
                self.alg.__add_node__(new_node)
                self.previously_exposed_leaf_ids.discard(id(this_node))
                self.alg.__mark_as_non_leaf__(this_node)
                this_node.children[node_id] = new_node
                this_node = new_node
                self.alg.cur_node_num += 1
                self.alg.__mark_as_leaf__(this_node)

            self.alg._record_microstep_history_path(
                current_prefix,
                this_node.hidden_state,
            )

        self.alg._record_request_history_path(cache_sequence, this_node.hidden_state)
        self.alg.timestamp += 1
        self.total_blocks += len(sequence)
        self.hit_blocks += hit_nodes
        self.miss_blocks += len(sequence) - hit_nodes
        if hit_nodes == len(sequence):
            self.request_full_hits += 1

    def evict_and_record(
        self,
        request_idx: int,
        step_index: int,
        evict_num: int,
        this_node: TrieNode,
        current_path: List[int],
    ) -> None:
        protected_leaves = self.alg._get_protected_leaves(current_path)
        candidates = self.alg._get_eviction_candidates(
            current_path,
            protected_node=this_node,
        )
        newly_exposed_leaf_ids = set()

        for eviction_ordinal in range(evict_num):
            candidates = self.alg._get_eviction_candidates(
                current_path,
                protected_node=this_node,
                candidates=candidates,
            )
            if not candidates:
                raise ValueError("No eviction candidates available")

            candidate_ids = {id(candidate) for candidate in candidates}
            newly_exposed_leaf_ids.intersection_update(candidate_ids)
            self.previously_exposed_leaf_ids.intersection_update(candidate_ids)
            row, target_idx = self.score_step(
                candidates=candidates,
                request_idx=request_idx,
                step_index=step_index,
                eviction_ordinal=eviction_ordinal,
                current_path=current_path,
                newly_exposed_leaf_ids=newly_exposed_leaf_ids,
                previously_exposed_leaf_ids=self.previously_exposed_leaf_ids,
            )
            self.rows.append(row)

            target_leaf = candidates.pop(target_idx)
            newly_exposed_leaf_ids.discard(id(target_leaf))
            self.previously_exposed_leaf_ids.discard(id(target_leaf))
            parent = target_leaf.parent
            self.alg.__delete_leaf_node__(target_leaf)

            if parent is not None and parent != self.alg.root_node and parent.is_leaf():
                if parent not in protected_leaves:
                    candidates.append(parent)
                    newly_exposed_leaf_ids.add(id(parent))
                    self.previously_exposed_leaf_ids.add(id(parent))

    def score_step(
        self,
        candidates: List[TrieNode],
        request_idx: int,
        step_index: int,
        eviction_ordinal: int,
        current_path: List[int],
        newly_exposed_leaf_ids: set,
        previously_exposed_leaf_ids: set,
    ) -> Tuple[Dict[str, object], int]:
        paths = [TrieNode.get_path_tuple_from_node(leaf) for leaf in candidates]
        node_id_paths = [TrieNode.get_node_id_path(leaf) for leaf in candidates]
        oracle_distances = [
            self.oracle.reuse_distance(path, request_idx) for path in paths
        ]
        relevances = [
            transformed_relevance(self.model, distance)
            for distance in oracle_distances
        ]

        best_raw = max(oracle_distances)
        best_rel = max(relevances)
        raw_top_set = {
            idx for idx, distance in enumerate(oracle_distances)
            if compare_distance(distance, best_raw) == 0
        }
        rel_top_set = {
            idx for idx, rel in enumerate(relevances)
            if rel == best_rel
        }

        leaf_states = []
        for leaf in candidates:
            if leaf.hidden_state is not None:
                leaf_states.append(leaf.hidden_state[0])
            else:
                leaf_states.append(torch.zeros(1, self.model.hidden_size, device=self.device))

        with torch.no_grad():
            scores, _ = self.model.forward(
                self.alg._microstep_history_memory(),
                self.alg._request_history_memory(),
                self.alg._candidate_lru_features(candidates),
                candidate_states=leaf_states,
            )
            model_idx = int(scores.squeeze(0).argmax().item())
            score_values = scores.squeeze(0).detach().float().cpu().tolist()

        lru_idx = min(
            range(len(candidates)),
            key=lambda idx: (
                candidates[idx].metadata
                if candidates[idx].metadata is not None
                else float("-inf")
            ),
        )
        oracle_idx = max(range(len(oracle_distances)), key=lambda idx: oracle_distances[idx])

        model_distance = oracle_distances[model_idx]
        lru_distance = oracle_distances[lru_idx]
        oracle_distance = oracle_distances[oracle_idx]
        model_relevance = relevances[model_idx]
        lru_relevance = relevances[lru_idx]

        model_vs_lru_cmp = compare_distance(model_distance, lru_distance)
        model_regret_raw = inf_aware_delta(model_distance, best_raw)
        lru_regret_raw = inf_aware_delta(lru_distance, best_raw)
        model_regret_rel = best_rel - model_relevance
        lru_regret_rel = best_rel - lru_relevance

        model_depth = candidate_depth(paths[model_idx])
        lru_depth = candidate_depth(paths[lru_idx])
        oracle_depth = candidate_depth(paths[oracle_idx])
        max_depth = max(candidate_depth(path) for path in paths)
        min_depth = min(candidate_depth(path) for path in paths)
        avg_depth = mean(candidate_depth(path) for path in paths)
        microstep_history_len = len(self.alg.microstep_history_hidden_states)
        request_history_len = len(self.alg.request_history_hidden_states)
        newly_exposed_candidate_ids = {
            idx for idx, candidate in enumerate(candidates)
            if id(candidate) in newly_exposed_leaf_ids
        }
        previously_exposed_candidate_ids = {
            idx for idx, candidate in enumerate(candidates)
            if id(candidate) in previously_exposed_leaf_ids
        }

        row = {
            "request_idx": request_idx,
            "step_index": step_index,
            "eviction_ordinal": eviction_ordinal,
            "num_candidates": len(candidates),
            "microstep_history_len": microstep_history_len,
            "request_history_len": request_history_len,
            "current_path_len": len(current_path),
            "candidate_count_bucket": bucket_candidate_count(len(candidates)),
            "microstep_history_len_bucket": bucket_history_len(
                microstep_history_len,
                self.model.max_microstep_history,
            ),
            "step_index_bucket": bucket_step_index(step_index),
            "parent_exposed_input": int(bool(newly_exposed_candidate_ids)),
            "new_parent_leaf_candidate_count": len(newly_exposed_candidate_ids),
            "model_selected_new_parent_leaf": int(model_idx in newly_exposed_candidate_ids),
            "lru_selected_new_parent_leaf": int(lru_idx in newly_exposed_candidate_ids),
            "oracle_selected_new_parent_leaf": int(oracle_idx in newly_exposed_candidate_ids),
            "previously_exposed_parent_input": int(bool(previously_exposed_candidate_ids)),
            "previously_exposed_parent_candidate_count": len(previously_exposed_candidate_ids),
            "model_selected_previously_exposed_parent_leaf": int(model_idx in previously_exposed_candidate_ids),
            "lru_selected_previously_exposed_parent_leaf": int(lru_idx in previously_exposed_candidate_ids),
            "oracle_selected_previously_exposed_parent_leaf": int(oracle_idx in previously_exposed_candidate_ids),
            "raw_top_set_size": len(raw_top_set),
            "relevance_top_set_size": len(rel_top_set),
            "raw_top_set_frac": len(raw_top_set) / len(candidates),
            "relevance_top_set_frac": len(rel_top_set) / len(candidates),
            "oracle_idx": oracle_idx,
            "model_idx": model_idx,
            "lru_idx": lru_idx,
            "model_in_raw_top_set": int(model_idx in raw_top_set),
            "model_in_relevance_top_set": int(model_idx in rel_top_set),
            "lru_in_raw_top_set": int(lru_idx in raw_top_set),
            "lru_in_relevance_top_set": int(lru_idx in rel_top_set),
            "model_equals_lru": int(model_idx == lru_idx),
            "model_vs_lru_oracle_cmp": model_vs_lru_cmp,
            "model_oracle_distance": model_distance,
            "lru_oracle_distance": lru_distance,
            "best_oracle_distance": best_raw,
            "model_relevance": model_relevance,
            "lru_relevance": lru_relevance,
            "best_relevance": best_rel,
            "model_regret_raw": model_regret_raw,
            "lru_regret_raw": lru_regret_raw,
            "model_regret_relevance": model_regret_rel,
            "lru_regret_relevance": lru_regret_rel,
            "model_score": score_values[model_idx],
            "lru_score": score_values[lru_idx],
            "score_std": float(torch.tensor(score_values).std(unbiased=False).item()),
            "model_path_depth": model_depth,
            "lru_path_depth": lru_depth,
            "oracle_path_depth": oracle_depth,
            "candidate_min_depth": min_depth,
            "candidate_avg_depth": avg_depth,
            "candidate_max_depth": max_depth,
            "model_path": " ".join(map(str, node_id_paths[model_idx])),
            "lru_path": " ".join(map(str, node_id_paths[lru_idx])),
            "oracle_path": " ".join(map(str, node_id_paths[oracle_idx])),
        }
        return row, model_idx


def aggregate_rows(rows: List[Dict[str, object]], key: Optional[str] = None) -> Dict[str, object]:
    if key is None:
        groups = {"overall": rows}
    else:
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[key])].append(row)

    result = {}
    for group_key, group_rows in groups.items():
        n = len(group_rows)
        raw_regrets = [float(row["model_regret_raw"]) for row in group_rows]
        rel_regrets = [float(row["model_regret_relevance"]) for row in group_rows]
        lru_rel_regrets = [float(row["lru_regret_relevance"]) for row in group_rows]
        top_sizes = [float(row["raw_top_set_size"]) for row in group_rows]
        rel_top_sizes = [float(row["relevance_top_set_size"]) for row in group_rows]
        num_candidates = [float(row["num_candidates"]) for row in group_rows]
        score_stds = [float(row["score_std"]) for row in group_rows]
        finite_raw_regrets = [v for v in raw_regrets if math.isfinite(v)]
        inf_raw_regrets = sum(1 for v in raw_regrets if math.isinf(v) and v > 0)

        model_wins = sum(1 for row in group_rows if int(row["model_vs_lru_oracle_cmp"]) > 0)
        model_losses = sum(1 for row in group_rows if int(row["model_vs_lru_oracle_cmp"]) < 0)
        model_ties = sum(1 for row in group_rows if int(row["model_vs_lru_oracle_cmp"]) == 0)

        result[group_key] = {
            "steps": n,
            "avg_candidates": mean(num_candidates),
            "avg_raw_top_set_size": mean(top_sizes),
            "p50_raw_top_set_size": finite_percentile(top_sizes, 0.5),
            "p90_raw_top_set_size": finite_percentile(top_sizes, 0.9),
            "avg_relevance_top_set_size": mean(rel_top_sizes),
            "model_raw_top_set_acc": mean(int(row["model_in_raw_top_set"]) for row in group_rows),
            "model_relevance_top_set_acc": mean(int(row["model_in_relevance_top_set"]) for row in group_rows),
            "lru_raw_top_set_acc": mean(int(row["lru_in_raw_top_set"]) for row in group_rows),
            "lru_relevance_top_set_acc": mean(int(row["lru_in_relevance_top_set"]) for row in group_rows),
            "model_equals_lru_rate": mean(int(row["model_equals_lru"]) for row in group_rows),
            "model_selected_previously_exposed_parent_leaf_rate": mean(
                int(row.get("model_selected_previously_exposed_parent_leaf", 0))
                for row in group_rows
            ),
            "lru_selected_previously_exposed_parent_leaf_rate": mean(
                int(row.get("lru_selected_previously_exposed_parent_leaf", 0))
                for row in group_rows
            ),
            "oracle_selected_previously_exposed_parent_leaf_rate": mean(
                int(row.get("oracle_selected_previously_exposed_parent_leaf", 0))
                for row in group_rows
            ),
            "previously_exposed_parent_input_rate": mean(
                int(row.get("previously_exposed_parent_input", 0))
                for row in group_rows
            ),
            "model_vs_lru_win_rate": model_wins / n if n else 0.0,
            "model_vs_lru_tie_rate": model_ties / n if n else 0.0,
            "model_vs_lru_loss_rate": model_losses / n if n else 0.0,
            "model_regret_raw_mean_finite": mean(finite_raw_regrets),
            "model_regret_raw_inf_frac": inf_raw_regrets / n if n else 0.0,
            "model_regret_raw_p50_finite": finite_percentile(raw_regrets, 0.5),
            "model_regret_raw_p90_finite": finite_percentile(raw_regrets, 0.9),
            "model_regret_relevance_mean": mean(rel_regrets),
            "model_regret_relevance_p50": finite_percentile(rel_regrets, 0.5),
            "model_regret_relevance_p90": finite_percentile(rel_regrets, 0.9),
            "lru_regret_relevance_mean": mean(lru_rel_regrets),
            "score_std_mean": mean(score_stds),
        }
    return result


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_summary(
    rows: List[Dict[str, object]],
    diagnostic: OnlineTopsetRegretDiagnostic,
    args: argparse.Namespace,
) -> Dict[str, object]:
    total = diagnostic.total_blocks
    hit_rate = diagnostic.hit_blocks / total if total else 0.0
    return {
        "inputs": {
            "data_path": args.data_path,
            "vocab_path": args.vocab_path,
            "config_path": args.config_path,
            "checkpoint_path": args.checkpoint_path,
            "max_requests": args.max_requests,
            "max_node_num": args.max_node_num,
            "device": args.device,
        },
        "hit_rate": hit_rate,
        "total_blocks": diagnostic.total_blocks,
        "hit_blocks": diagnostic.hit_blocks,
        "miss_blocks": diagnostic.miss_blocks,
        "request_full_hits": diagnostic.request_full_hits,
        "eviction_steps": len(rows),
        "overall": aggregate_rows(rows)["overall"] if rows else {},
        "by_candidate_count_bucket": aggregate_rows(rows, "candidate_count_bucket"),
        "by_microstep_history_len_bucket": aggregate_rows(
            rows,
            "microstep_history_len_bucket",
        ),
        "by_step_index_bucket": aggregate_rows(rows, "step_index_bucket"),
        "by_parent_exposed_input": aggregate_rows(rows, "parent_exposed_input"),
        "by_previously_exposed_parent_input": aggregate_rows(
            rows,
            "previously_exposed_parent_input",
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_path",
        default="data/oasst1_timed_global_b16/train.pkl",
    )
    parser.add_argument(
        "--vocab_path",
        default="data/oasst1_timed_global_b16/vocab.json",
    )
    parser.add_argument(
        "--config_path",
        default=(
            "checkpoints/diagnostics/overfit_ce_train500_s300/"
            "trie_model/oasst1_timed_global_b16/config.json"
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        default=(
            "checkpoints/diagnostics/overfit_ce_train500_s300/"
            "trie_model/oasst1_timed_global_b16/best.ckpt"
        ),
    )
    parser.add_argument("--output_csv", default="data/diagnostics/train500_ce_topset_regret.csv")
    parser.add_argument("--output_json", default="data/diagnostics/train500_ce_topset_regret_summary.json")
    parser.add_argument("--max_requests", type=int, default=500)
    parser.add_argument("--max_node_num", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    sequences = load_sequences(args.data_path, args.vocab_path)
    model = load_model(args.config_path, args.checkpoint_path, device)
    diagnostic = OnlineTopsetRegretDiagnostic(
        model=model,
        sequences=sequences,
        max_node_num=args.max_node_num,
        device=device,
    )
    rows = diagnostic.run(args.max_requests)
    summary = make_summary(rows, diagnostic, args)

    write_csv(args.output_csv, rows)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2, allow_nan=True)

    overall = summary["overall"]
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_json}")
    print(
        "hit_rate={:.4f} eviction_steps={} model_topset={:.4f} "
        "model_vs_lru(win/tie/loss)={:.4f}/{:.4f}/{:.4f}".format(
            summary["hit_rate"],
            summary["eviction_steps"],
            overall.get("model_raw_top_set_acc", 0.0),
            overall.get("model_vs_lru_win_rate", 0.0),
            overall.get("model_vs_lru_tie_rate", 0.0),
            overall.get("model_vs_lru_loss_rate", 0.0),
        )
    )


if __name__ == "__main__":
    main()
