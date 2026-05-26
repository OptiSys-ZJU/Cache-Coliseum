from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, List, Optional, Tuple


class PrefixFutureOracle:
    """
    Tracks future request indices for every request prefix.

    A cached trie leaf is reused when its root-to-leaf path appears as a prefix
    of a later request. This index turns a future trace scan into a deque lookup.
    """

    def __init__(
        self,
        sequences: Optional[Iterable[List[int]]] = None,
        max_prefix_len: Optional[int] = None,
    ):
        self.max_prefix_len = max_prefix_len
        self._future: Dict[Tuple[int, ...], Deque[int]] = defaultdict(deque)
        self.current_step = 0
        if sequences is not None:
            self.load(sequences)

    def load(self, sequences: Iterable[List[int]]):
        self._future.clear()
        self.current_step = 0
        for request_idx, sequence in enumerate(sequences):
            for prefix in self.iter_prefixes(sequence, self.max_prefix_len):
                self._future[prefix].append(request_idx)

    @staticmethod
    def iter_prefixes(
        sequence: List[int],
        max_prefix_len: Optional[int] = None,
    ):
        limit = len(sequence)
        if max_prefix_len is not None:
            limit = min(limit, max_prefix_len)

        prefix = []
        for key in sequence[:limit]:
            prefix.append(key)
            yield tuple(prefix)

    def consume_current(self, sequence: List[int], request_idx: Optional[int] = None):
        """
        Remove the current request from all of its prefix queues.

        After this is called for request i, next_request_index(prefix) returns
        the first future request strictly after i.
        """
        if request_idx is None:
            request_idx = self.current_step

        for prefix in self.iter_prefixes(sequence, self.max_prefix_len):
            accesses = self._future.get(prefix)
            if not accesses:
                continue

            while accesses and accesses[0] <= request_idx:
                accesses.popleft()
            if not accesses:
                del self._future[prefix]

        self.current_step = request_idx + 1

    def next_request_index(self, prefix: Tuple[int, ...]) -> float:
        accesses = self._future.get(tuple(prefix))
        if not accesses:
            return float("inf")
        return accesses[0]

    def reuse_distance(self, prefix: Tuple[int, ...], current_request_idx: int) -> float:
        next_idx = self.next_request_index(prefix)
        if next_idx == float("inf"):
            return float("inf")
        return next_idx - current_request_idx
