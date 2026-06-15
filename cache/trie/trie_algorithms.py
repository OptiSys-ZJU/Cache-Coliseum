from collections import defaultdict, deque
import copy
from functools import partial
import random
import types

try:
    import torch
except ModuleNotFoundError:
    torch = None

from cache.evict.algorithms import BaseEvictAlgorithm
from typing import List, Tuple, Type, Union, Optional, Any

from cache.evict.evictor import Evictor, LRUEvictor, RandEvictor
from cache.evict.predictor import OraclePredictor, Predictor
from cache.trie.oracle import PrefixFutureOracle

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.parent: Optional['TrieNode'] = None
        self.key = None
        self.metadata = None

        self.old_visited = -1
        self.guard = -1
        
        # New fields for Tree-LSTM integration (PRD 2.1)
        self.node_id: Optional[int] = None  # Token ID from vocabulary
        self.hidden_state: Optional[Tuple[Any, Any]] = None  # Cached LSTM (h, c) state
        self.embedding: Optional[Any] = None  # Node embedding tensor
        self.path_tuple: Optional[Tuple] = None
        self.node_id_path: Optional[Tuple[int, ...]] = None
    
    def __str__(self):
        return f"TrieNode(key={self.key}, node_id={self.node_id}, metadata={self.metadata}, children_count={len(self.children)})"

    def __repr__(self):
        return self.__str__()
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def clear_hidden_state(self):
        """Clear cached hidden state (useful when model is updated)."""
        self.hidden_state = None
        self.embedding = None
    
    @staticmethod
    def is_prefix(sub, full):
        return len(sub) <= len(full) and sub == full[:len(sub)]

    @staticmethod
    def get_path_tuple_from_node(node: 'TrieNode'):
        if node.path_tuple is not None:
            return node.path_tuple

        path = []
        current = node
        while current is not None:
            if current.parent is not None and current.key is not None:
                path.append(current.key)
            current = current.parent

        node.path_tuple = tuple(reversed(path))
        return node.path_tuple

    @staticmethod
    def get_node_id_path(node: 'TrieNode') -> tuple:
        """
        Return root-to-leaf path of node_ids (int tokens) for model encoding.
        
        Walks from node up to root, collecting node_id values (skipping root
        which has node_id=None). Returns in root→leaf order.
        """
        if node.node_id_path is not None:
            return node.node_id_path

        path = []
        current = node
        while current is not None:
            if current.node_id is not None:
                path.append(current.node_id)
            current = current.parent
        node.node_id_path = tuple(reversed(path))
        return node.node_id_path

class SimpleTrie:
    def __init__(self):
        self.root = {}

    def add(self, keys):
        d = self.root
        for k in keys:
            d = d.setdefault(k, {})

    def full_match(self, keys):
        """Return True if keys fully matched the trie"""
        d = self.root
        for k in keys:
            if k not in d:
                return False
            d = d[k]
        return True

class Trie:
    def __init__(self):
        self.root_node = TrieNode()
    
    def match(self, keys):
        this_node = self.root_node
        for idx, key in enumerate(keys):
            if key in this_node.children.keys():
                this_node = this_node.children[key]
            else:
                return this_node, keys[idx:]
        return this_node, []

    def add(self, keys):
        this_node = self.root_node
        for idx, key in enumerate(keys):
            if key in this_node.children.keys():
                this_node = this_node.children[key]
            else:
                new_node = TrieNode()
                new_node.key = key
                this_node.children[key] = new_node
                new_node.parent = this_node
                this_node = new_node


class TrieEvictAlgorithm(BaseEvictAlgorithm):
    def __init__(self, max_node_num):
        super().__init__()
        self.root_node = TrieNode()
        self.root_node.key = ()
        self.cur_node_num = 0
        self.max_node_num = max_node_num
        self.leaf_nodes = {self.root_node: None}
        self.eviction_count = 0
    
    def __leaves__(self):
        return list(self.leaf_nodes)

    def __is_live_leaf__(self, node):
        if node == self.root_node:
            return node in self.leaf_nodes
        if node not in self.leaf_nodes or not node.is_leaf():
            return False

        current = node
        seen = set()
        while current is not None and current != self.root_node:
            if current in seen:
                return False
            seen.add(current)

            parent = current.parent
            if parent is None:
                return False
            if parent.children.get(current.key) is not current:
                return False
            current = parent
        return current == self.root_node

    def __eviction_candidates__(self, protected_node=None):
        return [
            leaf for leaf in self.__leaves__()
            if leaf != self.root_node
            and leaf != protected_node
            and self.__is_live_leaf__(leaf)
        ]

    def __refresh_candidates__(self, candidates, protected_node=None):
        return [
            leaf for leaf in candidates
            if leaf != self.root_node
            and leaf != protected_node
            and self.__is_live_leaf__(leaf)
        ]
    
    def __mark_as_non_leaf__(self, node):
        if node in self.leaf_nodes:
            del self.leaf_nodes[node]

    def __mark_as_leaf__(self, node):
        if node not in self.leaf_nodes:
            self.leaf_nodes[node] = None
    
    def __delete_leaf_node__(self, node):
        target_parent = node.parent
        del target_parent.children[node.key]
        self.cur_node_num -= 1
        self.eviction_count += 1
        del self.leaf_nodes[node]
        if len(target_parent.children) == 0:
            self.__mark_as_leaf__(target_parent)
    
    def prune(self, leaf_nodes: List[TrieNode]) -> int:
        """
        Batch delete leaf nodes and recursively clean up empty parent nodes.
        
        Args:
            leaf_nodes: List of leaf nodes to delete
            
        Returns:
            Number of nodes actually deleted
        """
        deleted_count = 0
        
        for node in leaf_nodes:
            # Skip if not a valid leaf or already deleted
            if node not in self.leaf_nodes:
                continue
            if node == self.root_node:
                continue  # Never delete root
            
            # Delete the leaf
            self.__delete_leaf_node__(node)
            deleted_count += 1
            
            # Recursively clean up empty parents (up to branch point)
            parent = node.parent
            while parent is not None and parent != self.root_node:
                if len(parent.children) == 0:
                    # Parent became empty, delete it too
                    grandparent = parent.parent
                    if grandparent is not None:
                        del grandparent.children[parent.key]
                        self.cur_node_num -= 1
                        self.eviction_count += 1
                        if parent in self.leaf_nodes:
                            del self.leaf_nodes[parent]
                        deleted_count += 1
                        # Check if grandparent should become a leaf
                        if len(grandparent.children) == 0:
                            self.__mark_as_leaf__(grandparent)
                    parent = grandparent
                else:
                    break  # Parent still has other children, stop
        
        return deleted_count
    
    def __match__(self, aligned_address: List) -> Tuple[TrieNode, List]:
        this_node = self.root_node
        for idx, key in enumerate(aligned_address):
            if key in this_node.children.keys():
                this_node = this_node.children[key]
                self.__visit_node__(this_node)
            else:
                return this_node, aligned_address[idx:]
        return this_node, []
    
    def __insert__(self, this_node, insert_list: List[Tuple]):
        # evict test
        insert_len = len(insert_list)
        if insert_len == 0:
            return
        
        evict_num = self.cur_node_num + insert_len - self.max_node_num
        if evict_num > 0:
            self.__evict__(evict_num, this_node)
        for key in insert_list:
            new_node = TrieNode()
            new_node.key = key
            new_node.parent = this_node  # Set parent before __add_node__ for subclass access
            self.__add_node__(new_node)
            self.__mark_as_non_leaf__(this_node)
            this_node.children[key] = new_node
            this_node = new_node
            self.cur_node_num += 1
        self.__mark_as_leaf__(this_node)
    
    def __evict__(self, evict_num, this_node):
        pass

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#cur_node_num: [{self.cur_node_num}/{self.max_node_num}]")
        
    def _print_helper(self, node: TrieNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key[:10],
                current_node.metadata if current_node.metadata is not None else '',
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

    def __visit_node__(self, node: TrieNode):
        pass

    def __add_node__(self, node: TrieNode):
        pass

    def access(self, pc, aligned_address: List) -> Tuple:
        pass


class TriePredictAlgorithm(TrieEvictAlgorithm):
    def __init__(self, max_node_num, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial]):
        super().__init__(max_node_num)

        cls_type = predictor_type.func if hasattr(predictor_type, 'func') else predictor_type
        if issubclass(cls_type, OraclePredictor):
            def oracle_access(self, pc, address, next_access_time):
                self.predictor.oracle_access(pc, address, next_access_time)
            self.oracle_access = types.MethodType(oracle_access, self)

        self.timestamp = 0
        self.evictor = evictor_type()
        self.predictor = predictor_type()

        ### store preds per access
        self.to_fill_nodes = deque()
    
    def __visit_node__(self, node: TrieNode):
        self.to_fill_nodes.append(node)
    
    def __add_node__(self, node: TrieNode):
        self.to_fill_nodes.append(node)
    
    def __evict__(self, evict_num, this_node):
        for i in range(evict_num):
            leaves = self.__leaves__()
            if leaves[0] == self.root_node:
                raise ValueError("Can't evict root node")
            if this_node in leaves:
                leaves.remove(this_node)

            a = list(enumerate([leaf.metadata for leaf in leaves]))
            target_idx = self.evictor.evict(a)
            target_leaf = leaves[target_idx]

            self.__delete_leaf_node__(target_leaf)
    
    def after_pred(self, pc ,address):
        cur_preds = deque(self.predictor.predict_score(self.timestamp, pc, address, None))
        assert len(self.to_fill_nodes) == len(cur_preds)
        while cur_preds:
            pred = cur_preds.popleft()
            node = self.to_fill_nodes.popleft()
            node.metadata = pred

        self.timestamp += 1

    def access(self, pc, aligned_address: List) -> Tuple:
        this_node, insert_list = self.__match__(aligned_address)
        self.__insert__(this_node, insert_list)

        self.after_pred(pc, aligned_address)

        assert len(self.to_fill_nodes) == 0

        return (len(aligned_address), len(aligned_address) - len(insert_list), len(insert_list))

class TrieGuard(TriePredictAlgorithm):
    def __init__(self, max_node_num, evictor_type, predictor_type, **kwargs):
        super().__init__(max_node_num, evictor_type, predictor_type)

        self.phase_timestamp = 0
        self.phase_evicted_set = SimpleTrie()
        self.error_times = 0

        self.follow_if_guarded = kwargs.get('follow_if_guarded', False)
        
        if 'relax_times' in kwargs:
            self.relax_times = kwargs['relax_times']
        else:
            self.relax_times = 0
        if 'relax_prob' in kwargs:
            self.relax_prob = kwargs['relax_prob']
        else:
            self.relax_prob = 0

    def __insert__(self, this_node, insert_list: List[Tuple], aligned_address):
        # evict test
        to_guard = False
        insert_len = len(insert_list)
        if insert_len == 0:
            return to_guard

        evict_num = self.cur_node_num + insert_len - self.max_node_num
        if evict_num > 0:
            to_guard = self.__evict__(evict_num, this_node, aligned_address)

        for key in insert_list:
            new_node = TrieNode()
            new_node.key = key
            new_node.parent = this_node  # Set parent before __add_node__ for subclass access
            self.__add_node__(new_node)
            self.__mark_as_non_leaf__(this_node)
            this_node.children[key] = new_node
            this_node = new_node
            self.cur_node_num += 1

        self.__mark_as_leaf__(this_node)

        return to_guard

    def __mark_old_visited__(self, keys):
        this_node = self.root_node
        for idx, key in enumerate(keys):
            assert key in this_node.children.keys()
            this_node.old_visited = self.timestamp
            this_node = this_node.children[key]
    
    def __mark_guarded__(self, keys):
        this_node = self.root_node
        for idx, key in enumerate(keys):
            assert key in this_node.children.keys()
            this_node.guard = self.timestamp
            this_node = this_node.children[key]

    def __evict__(self, evict_num, this_node, aligned_address):
        to_guard = False
        for _ in range(evict_num):
            leaves = self.__leaves__()
            if leaves[0] == self.root_node:
                raise ValueError("Can't evict root node")
            if this_node in leaves:
                leaves.remove(this_node)

            unvisited_idx = []
            for idx, leaf in enumerate(leaves):
                if leaf.old_visited < self.phase_timestamp:
                    unvisited_idx.append(idx)

            if not unvisited_idx:
                # new phase
                self.phase_timestamp = self.timestamp
                self.phase_evicted_set = SimpleTrie()
                self.error_times = 0
            
            if self.phase_evicted_set.full_match(aligned_address):
                if self.relax_times != 0:
                    self.error_times += 1
                    if self.error_times >= self.relax_times:
                        to_guard = True
                else:
                    if random.random() > self.relax_prob:
                        to_guard = True

            if to_guard and not self.follow_if_guarded:
                target_idx = random.choice(unvisited_idx)
            else:
                unguarded_idx = []
                for idx, leaf in enumerate(leaves):
                    if leaf.guard < self.phase_timestamp:
                        unguarded_idx.append(idx)
                target_idx = self.evictor.evict([(i, leaves[i].metadata) for i in unguarded_idx])

            target_leaf = leaves[target_idx]

            ## handle evict
            self.phase_evicted_set.add(TrieNode.get_path_tuple_from_node(target_leaf))

            self.__delete_leaf_node__(target_leaf)
        
        return to_guard

    def access(self, pc, aligned_address: List) -> Tuple:
        this_node, insert_list = self.__match__(aligned_address)
        to_guard = self.__insert__(this_node, insert_list, aligned_address)

        self.__mark_old_visited__(tuple(aligned_address))
        if to_guard:
            self.__mark_guarded__(tuple(aligned_address))

        self.after_pred(pc, aligned_address)

        assert len(self.to_fill_nodes) == 0

        return (len(aligned_address), len(aligned_address) - len(insert_list), len(insert_list))

#############################################
class TrieRandAlgorithm(TrieEvictAlgorithm):
    def __init__(self, max_node_num):
        super().__init__(max_node_num)
        self.evictor = RandEvictor()

    def __evict__(self, evict_num, this_node):
        candidates = self.__eviction_candidates__(protected_node=this_node)
        for _ in range(evict_num):
            candidates = self.__refresh_candidates__(
                candidates,
                protected_node=this_node,
            )
            if not candidates:
                raise ValueError("No eviction candidates available")

            target_idx = self.evictor.evict(list(enumerate(candidates)))
            target_leaf = candidates.pop(target_idx)
            parent = target_leaf.parent
            self.__delete_leaf_node__(target_leaf)

            if parent is not None and parent != self.root_node and parent.is_leaf():
                if parent != this_node:
                    candidates.append(parent)
    
    def access(self, pc, aligned_address: List) -> Tuple:
        this_node, insert_list = self.__match__(aligned_address)
        self.__insert__(this_node, insert_list)
        return (len(aligned_address), len(aligned_address) - len(insert_list), len(insert_list))

class TrieLRUAlgorithm(TrieEvictAlgorithm):
    def __init__(self, max_node_num):
        super().__init__(max_node_num)
        self.counter = 0
        self.evictor = LRUEvictor()
    
    def __visit_node__(self, node: TrieNode):
        node.metadata = self.counter
    
    def __add_node__(self, node: TrieNode):
        node.metadata = self.counter
    
    def __evict__(self, evict_num, this_node):
        candidates = self.__eviction_candidates__(protected_node=this_node)
        for _ in range(evict_num):
            candidates = self.__refresh_candidates__(
                candidates,
                protected_node=this_node,
            )
            if not candidates:
                raise ValueError("No eviction candidates available")

            target_idx = self.evictor.evict(
                list(enumerate([leaf.metadata for leaf in candidates]))
            )
            target_leaf = candidates.pop(target_idx)
            parent = target_leaf.parent
            self.__delete_leaf_node__(target_leaf)

            if parent is not None and parent != self.root_node and parent.is_leaf():
                if parent != this_node:
                    candidates.append(parent)
    
    def access(self, pc, aligned_address: List) -> Tuple:
        this_node, insert_list = self.__match__(aligned_address)
        self.__insert__(this_node, insert_list)
        self.counter += 1
        return (len(aligned_address), len(aligned_address) - len(insert_list), len(insert_list))


class TrieOracleAlgorithm(TrieEvictAlgorithm):
    """
    Belady-style oracle for prefix-trie eviction.

    It evicts the leaf whose root-to-leaf path is reused farthest in the future.
    A path is reused when it is a prefix of a later request.
    """

    def __init__(self, max_node_num, future_oracle: Optional[PrefixFutureOracle] = None):
        super().__init__(max_node_num)
        self.future_oracle = future_oracle
        self.timestamp = 0

    def set_future_oracle(self, future_oracle: PrefixFutureOracle):
        self.future_oracle = future_oracle
        self.timestamp = 0

    def __evict__(self, evict_num, this_node):
        candidates = self.__eviction_candidates__(protected_node=this_node)
        for _ in range(evict_num):
            candidates = self.__refresh_candidates__(
                candidates,
                protected_node=this_node,
            )
            if not candidates:
                raise ValueError("No eviction candidates available")

            best_idx = 0
            best_next_use = -1
            for idx, leaf in enumerate(candidates):
                path = TrieNode.get_path_tuple_from_node(leaf)
                next_use = (
                    float("inf")
                    if self.future_oracle is None
                    else self.future_oracle.next_request_index(path)
                )
                if next_use > best_next_use:
                    best_idx = idx
                    best_next_use = next_use

            target_leaf = candidates.pop(best_idx)
            parent = target_leaf.parent
            self.__delete_leaf_node__(target_leaf)

            if parent is not None and parent != self.root_node and parent.is_leaf():
                if parent != this_node:
                    candidates.append(parent)

    def access(self, pc, aligned_address: List = None) -> Tuple:
        sequence = aligned_address if aligned_address is not None else pc
        if self.future_oracle is not None:
            self.future_oracle.consume_current(sequence, self.timestamp)

        this_node, insert_list = self.__match__(sequence)
        self.__insert__(this_node, insert_list)
        self.timestamp += 1
        return (len(sequence), len(sequence) - len(insert_list), len(insert_list))


class TrieModelPredictAlgorithm(TrieEvictAlgorithm):
    """
    Tree-LSTM model-based eviction algorithm.
    
    Uses a TrieParrotModel to score leaf nodes for eviction.
    Implements "Protected Leaf Eviction" (PRD 2.3):
    - Current access path is protected from eviction
    - Model scores leaf nodes via attention mechanism
    - Lowest scored leaves are evicted
    """
    
    def __init__(self, max_node_num, model=None):
        """
        Args:
            max_node_num: Maximum number of nodes in the trie
            model: TrieParrotModel instance (can be set later via set_model)
        """
        super().__init__(max_node_num)
        self.model = model
        self.history_state = None  # rolling (h, c) state for sequential history encoding
        self.history_hidden_states = deque([], maxlen=self._history_maxlen())
        self.history_token_window = deque([], maxlen=self._history_maxlen())
        self.timestamp = 0
        self.counter = 0
    
    def _history_maxlen(self):
        if self.model is None:
            return 1
        return max(1, int(self.model.max_attention_history))

    def _reset_history(self):
        self.history_state = None
        self.history_hidden_states = deque([], maxlen=self._history_maxlen())
        self.history_token_window = deque([], maxlen=self._history_maxlen())

    def set_model(self, model):
        """Set or replace the prediction model."""
        self.model = model
        self._reset_history()

    def _history_memory(self):
        if not self.history_hidden_states:
            return None
        return list(self.history_hidden_states)

    def _record_history_step(self, node_id: int):
        if self.model is None:
            return
        if torch is None:
            raise ImportError("TrieModelPredictAlgorithm requires torch when model is set")
        with torch.no_grad():
            self.history_state = self.model.encode_history_step(
                node_id, self.history_state
            )
        self.history_hidden_states.append(self.history_state[0])
        self.history_token_window.append(node_id)

    def _record_history_sequence(self, sequence: List[int]):
        for node_id in sequence:
            self._record_history_step(node_id)
    
    def _get_protected_leaves(self, current_path: List) -> set:
        """
        Get the set of leaf nodes on the current access path (protected from eviction).
        
        Since the eviction candidate set is already all leaf nodes, we only need
        to protect leaf nodes that lie on the current path.
        
        Args:
            current_path: The sequence currently being accessed
            
        Returns:
            Set of leaf TrieNode objects on the current path
        """
        protected = set()
        node = self.root_node
        for key in current_path:
            if key in node.children:
                node = node.children[key]
                if node.is_leaf():
                    protected.add(node)
            else:
                break
        return protected
    
    def _get_eviction_candidates(
        self,
        current_path: List,
        protected_node: TrieNode = None,
        candidates: List[TrieNode] = None,
    ) -> List[TrieNode]:
        """
        Get live leaf nodes eligible for eviction for the current access.
        
        Args:
            current_path: The sequence currently being accessed
            protected_node: Additional node to exclude (for example, the current match node)
            candidates: Optional existing candidate pool to refresh incrementally
            
        Returns:
            List of candidate leaf nodes
        """
        protected_leaves = self._get_protected_leaves(current_path)
        pool = self.__leaves__() if candidates is None else candidates
        return [
            leaf for leaf in pool
            if leaf not in protected_leaves
            and leaf != self.root_node
            and leaf != protected_node
            and self.__is_live_leaf__(leaf)
        ]
    
    def __evict_with_protection__(self, evict_num: int, this_node: TrieNode, current_path: List):
        """
        Evict leaf nodes using model scores with path protection.
        
        Protected leaves and candidates are computed once. After each eviction,
        candidates are updated incrementally: remove the evicted leaf, and if its
        parent becomes a new leaf (and is not root), add the parent to candidates.
        
        Args:
            evict_num: Number of nodes to evict
            this_node: Node at the end of the matched path
            current_path: The full current access sequence
        """
        protected_leaves = self._get_protected_leaves(current_path)
        candidates = self._get_eviction_candidates(
            current_path,
            protected_node=this_node,
        )
        
        for _ in range(evict_num):
            candidates = self._get_eviction_candidates(
                current_path,
                protected_node=this_node,
                candidates=candidates,
            )
            if not candidates:
                raise ValueError("No eviction candidates available (all nodes protected)")
            
            if self.model is not None and self.history_hidden_states:
                if torch is None:
                    raise ImportError("TrieModelPredictAlgorithm requires torch when model is set")
                # Use model to score candidates
                leaf_states = []
                for c in candidates:
                    if c.hidden_state is not None:
                        leaf_states.append(c.hidden_state[0])  # h component
                    else:
                        # Fallback: zero state for nodes without cached state
                        leaf_states.append(torch.zeros(1, self.model.hidden_size))
                
                with torch.no_grad():
                    scores, _ = self.model.forward(
                        self._history_memory(),
                        candidate_states=leaf_states,
                    )
                
                # Evict the node with the highest eviction logit
                target_idx = scores.squeeze(0).argmax().item()
            else:
                # Fallback: random eviction when model not available
                target_idx = random.randint(0, len(candidates) - 1)
            
            target_leaf = candidates.pop(target_idx)
            parent = target_leaf.parent
            self.__delete_leaf_node__(target_leaf)
            
            # Incremental update: if parent became a leaf and is not root, add to candidates
            if parent is not None and parent != self.root_node and parent.is_leaf():
                if parent not in protected_leaves:
                    candidates.append(parent)
    
    def __evict__(self, evict_num, this_node):
        """Fallback random eviction when path protection is unavailable."""
        for _ in range(evict_num):
            leaves = self.__leaves__()
            candidates = [l for l in leaves if l != self.root_node and l != this_node]
            if not candidates:
                raise ValueError("No eviction candidates available")
            target_leaf = candidates[random.randint(0, len(candidates) - 1)]
            self.__delete_leaf_node__(target_leaf)
    
    def __visit_node__(self, node: TrieNode):
        """Track visited nodes for potential state updates."""
        node.metadata = self.counter
    
    def __add_node__(self, node: TrieNode):
        """When a new node is added, compute its Tree-LSTM state incrementally."""
        node.metadata = self.counter
        if self.model is not None:
            if torch is None:
                raise ImportError("TrieModelPredictAlgorithm requires torch when model is set")
            parent = node.parent
            parent_state = parent.hidden_state if parent is not None else None
            if node.node_id is not None:
                with torch.no_grad():
                    h, c = self.model.compute_node_state(node.node_id, parent_state)
                node.hidden_state = (h, c)
    
    def __insert__(self, this_node, insert_list: List, current_path: List = None):
        """Insert with path-aware eviction."""
        insert_len = len(insert_list)
        if insert_len == 0:
            return
        
        evict_num = self.cur_node_num + insert_len - self.max_node_num
        if evict_num > 0:
            if current_path is not None:
                self.__evict_with_protection__(evict_num, this_node, current_path)
            else:
                # Fallback to basic eviction
                self.__evict__(evict_num, this_node)
        
        for key in insert_list:
            new_node = TrieNode()
            new_node.key = key
            new_node.node_id = key  # For Tree-LSTM, key is the token ID
            new_node.parent = this_node  # Must set parent BEFORE __add_node__ (needs parent.hidden_state)
            self.__add_node__(new_node)
            self.__mark_as_non_leaf__(this_node)
            this_node.children[key] = new_node
            this_node = new_node
            self.cur_node_num += 1
        self.__mark_as_leaf__(this_node)

    def access(self, sequence: List[int]) -> Tuple:
        """
        Process the cache-visible prefix of one sequence access.

        Args:
            sequence: List of token IDs representing the access path.
            
        Returns:
            Tuple of (total_nodes, hit_nodes, miss_nodes) for the prefix that
            can participate in the trie under the current capacity semantics.
        """
        sequence = sequence[:self.max_node_num]
        self.counter += 1
        this_node = self.root_node
        hit_nodes = 0

        current_prefix = []
        for node_id in sequence:
            current_prefix.append(node_id)
            if node_id in this_node.children:
                this_node = this_node.children[node_id]
                self.__visit_node__(this_node)
                hit_nodes += 1
            else:
                self.__insert__(this_node, [node_id], current_path=current_prefix)
                this_node = this_node.children[node_id]

            # Step-wise PARROT semantics: step i becomes visible only after
            # the eviction/insertion decision of step i has already finished.
            self._record_history_step(node_id)

        self.timestamp += 1
        return (len(sequence), hit_nodes, len(sequence) - hit_nodes)


#############################################
# Task 4.3: Model-based Guard with confidence fallback
#############################################

class TrieModelGuard(TrieModelPredictAlgorithm):
    """
    Extends TrieModelPredictAlgorithm with confidence-based fallback.
    
    When the model's score variance across candidates is below a threshold,
    the model is considered "unsure" and we fall back to LRU eviction.
    Tracks guard statistics for analysis.
    """
    
    def __init__(self, max_node_num, model=None, variance_threshold: float = 0.01):
        """
        Args:
            max_node_num: Maximum number of nodes in the trie
            model: TrieParrotModel instance
            variance_threshold: If score variance < this, fall back to LRU
        """
        super().__init__(max_node_num, model)
        self.variance_threshold = variance_threshold
        # Statistics
        self.total_evictions = 0
        self.guarded_evictions = 0  # Times we fell back to LRU
    
    def __evict_with_protection__(self, evict_num: int, this_node: TrieNode, current_path: List):
        """
        Evict with confidence check: if model score variance is too low,
        fall back to LRU eviction instead of trusting the model.
        """
        protected_leaves = self._get_protected_leaves(current_path)
        candidates = self._get_eviction_candidates(
            current_path,
            protected_node=this_node,
        )
        
        for _ in range(evict_num):
            candidates = self._get_eviction_candidates(
                current_path,
                protected_node=this_node,
                candidates=candidates,
            )
            if not candidates:
                raise ValueError("No eviction candidates available (all nodes protected)")
            
            self.total_evictions += 1
            use_model = False
            
            if self.model is not None and self.history_hidden_states and len(candidates) > 1:
                if torch is None:
                    raise ImportError("TrieModelGuard requires torch when model is set")
                leaf_states = []
                for c in candidates:
                    if c.hidden_state is not None:
                        leaf_states.append(c.hidden_state[0])
                    else:
                        leaf_states.append(torch.zeros(1, self.model.hidden_size))
                
                with torch.no_grad():
                    scores, _ = self.model.forward(
                        self._history_memory(),
                        candidate_states=leaf_states,
                    )
                
                score_variance = scores.var().item()
                
                if score_variance >= self.variance_threshold:
                    # Model is confident: use model scores
                    target_idx = scores.squeeze(0).argmax().item()
                    use_model = True
                # else: fall through to LRU
            
            if not use_model:
                # Fallback: LRU eviction (model unsure or unavailable)
                target_idx = min(
                    range(len(candidates)),
                    key=lambda idx: candidates[idx].metadata
                    if candidates[idx].metadata is not None
                    else float("-inf"),
                )
                self.guarded_evictions += 1
            
            target_leaf = candidates.pop(target_idx)
            parent = target_leaf.parent
            self.__delete_leaf_node__(target_leaf)
            
            if parent is not None and parent != self.root_node and parent.is_leaf():
                if parent not in protected_leaves:
                    candidates.append(parent)
    
    @property
    def guard_rate(self) -> float:
        """Fraction of evictions that fell back to LRU."""
        if self.total_evictions == 0:
            return 0.0
        return self.guarded_evictions / self.total_evictions
