# TODO: Post-Baseline Trie-PARROT Ideas

This file collects optional improvements that should not be mixed into the
first PARROT-faithful implementation. Treat these as ablation candidates after
the baseline is working and benchmarked.

## Candidate Metadata Features

- Add explicit candidate age, such as `now - candidate.last_access_time`.
- Add candidate LRU rank among resident leaves.
- Add candidate path depth.
- Add candidate hit/touch count or reuse frequency.
- Compare against the PARROT-faithful baseline to check whether these features
  merely recover LRU behavior or provide extra predictive signal.

## History Metadata Features

- Add age or relative position embeddings to history slots.
- Add depth embeddings for history paths.
- Test fixed positional embeddings versus learned age buckets.
- Test whether newest-to-oldest or oldest-to-newest history ordering works
  better for attention.

## History Sequence Encoders

- After the path-slot baseline, try a small GRU/LSTM over recent history path
  embeddings.
- Compare direct attention over path slots with attention over recurrent hidden
  states.
- Consider a tiny Transformer encoder over the recent history slots only if the
  recurrent variant is promising.

## Loss Variants

- Add KL loss using a soft oracle policy derived from oracle reuse distances.
- Add approximate NDCG/listwise ranking loss, following the original PARROT
  options.
- Add NDCG@K or top-set aware listwise loss so ties among equally good eviction
  candidates are not punished as arbitrary top-1 mistakes.
- Compare cross-entropy only, reuse-distance regression only, and combined loss.
- Tune how infinite reuse distances are capped for log-distance training.

## Oracle And Label Diagnostics

- Log the full distribution of oracle distances per eviction step.
- Track how often multiple candidates tie at infinite distance.
- Track whether sequential pruning repeatedly walks up the same branch.
- Add examples where deleting a leaf exposes a parent whose oracle distance is
  much shorter than the deleted child.

## Trie-Specific Model Ideas

- Add explicit parent/child relation features for candidates that become leaves
  after pruning.
- Use subtree statistics, such as number of descendants before pruning.
- Add a branch-local feature indicating whether sibling branches were recently
  touched.
- Explore separate encoders for shallow shared prefixes and deep private suffixes.

## Benchmark Ablations

- Run the PARROT-faithful model and metadata-enhanced variants on the same
  capacity/block-size settings.
- Report recovered oracle gap:
  `(model_hit_rate - lru_hit_rate) / (oracle_hit_rate - lru_hit_rate)`.
- Keep request-clock oracle fixed while changing only the student architecture.
- Compare model policy against model-guard policy after the baseline is stable.

## DAgger Scale-Up

- Run full-data DAgger on the remote A100 host, not only train100/train500
  overfit slices.
- Increase model-induced snapshot collection enough that DAgger actually sees
  states created by the student's own eviction mistakes.
- Decouple DAgger refresh cadence from mini-batch size so online distribution
  correction is not hidden behind `ceil(snapshots / batch_size)`.
- Track online rollout hit rate, fixed-snapshot top-set/regret, and examples
  where oracle-rollin loss looks good but model-rollout evicts a near-future
  leaf.
