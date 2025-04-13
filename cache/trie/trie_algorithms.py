from collections import defaultdict, deque
import copy
from functools import partial
import random
import types
from cache.evict.algorithms import BaseEvictAlgorithm
from typing import List, Tuple, Type, Union

from cache.evict.evictor import Evictor, LRUEvictor, RandEvictor
from cache.evict.predictor import OraclePredictor, Predictor

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.parent = None
        self.key = None
        self.metadata = None
    
    def __str__(self):
        return f"TrieNode(key={self.key}, metadata={self.metadata}, children_count={len(self.children)})"

    def __repr__(self):
        return self.__str__()
    
    def is_leaf(self):
        return len(self.children) == 0
    
    @staticmethod
    def is_prefix(sub, full):
        return len(sub) <= len(full) and sub == full[:len(sub)]

    @staticmethod
    def get_path_tuple_from_node(node: 'TrieNode'):
        path = []
        current = node
        while current is not None:
            if current.key is not None:
                path.append(current.key)
            current = current.parent
        reversed_path = list(reversed(path))
        return tuple(reversed_path[1:])

class TrieEvictAlgorithm(BaseEvictAlgorithm):
    def __init__(self, max_node_num):
        super().__init__()
        self.root_node = TrieNode()
        self.root_node.key = ()
        self.cur_node_num = 0
        self.max_node_num = max_node_num
    
    def __leaves__(self) -> List[TrieNode]:
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list
    
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
        evict_num = self.cur_node_num + insert_len - self.max_node_num
        if evict_num > 0:
            self.__evict__(evict_num, this_node)

        for key in insert_list:
            new_node = TrieNode()
            new_node.key = key
            self.__add_node__(new_node)
            this_node.children[key] = new_node
            new_node.parent = this_node
            this_node = new_node
            self.cur_node_num += 1
    
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
        leaves = self.__leaves__()
        if leaves[0] == self.root_node:
            raise ValueError("Can't evict root node")
        if this_node in leaves:
            leaves.remove(this_node)
        
        for i in range(evict_num):
            a = list(enumerate([leaf.metadata for leaf in leaves]))
            target_idx = self.evictor.evict(a)
            target_leaf = leaves[target_idx]

            target_parent = target_leaf.parent
            del target_parent.children[target_leaf.key]
            self.cur_node_num -= 1
            if len(target_parent.children) == 0:
                leaves.append(target_parent)
            leaves.remove(target_leaf)
    
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

        self.old_visited_set = []
        self.guarded_set = []
        self.phase_evicted_set = []
        self.error_times = 0

        if 'follow_if_guarded' in kwargs:
            self.follow_if_guarded = kwargs['follow_if_guarded']
        else:
            self.follow_if_guarded = False
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
            self.__add_node__(new_node)
            this_node.children[key] = new_node
            new_node.parent = this_node
            this_node = new_node
            self.cur_node_num += 1

        return to_guard

    def __evict__(self, evict_num, this_node, aligned_address):
        leaves = self.__leaves__()
        if leaves[0] == self.root_node:
            raise ValueError("Can't evict root node")
        if this_node in leaves:
            leaves.remove(this_node)

        to_guard = False
        for i in range(evict_num):
            unvisited_idx = []
            for i, leaf in enumerate(leaves):
                path = TrieNode.get_path_tuple_from_node(leaf)
                if not any(TrieNode.is_prefix(path, vp) for vp in self.old_visited_set):
                    unvisited_idx.append(i)

            if not unvisited_idx:
                # new phase
                self.old_visited_set = []
                self.guarded_set = []
                self.phase_evicted_set = []
                self.error_times = 0
            
            if any(TrieNode.is_prefix(p, tuple(aligned_address)) for p in self.phase_evicted_set):
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
                for i, leaf in enumerate(leaves):
                    path = TrieNode.get_path_tuple_from_node(leaf)
                    if not any(TrieNode.is_prefix(path, gp) for gp in self.guarded_set):
                        unguarded_idx.append(i)
                target_idx = self.evictor.evict([(i, leaves[i].metadata) for i in unguarded_idx])

            target_leaf = leaves[target_idx]

            ## handle evict
            self.phase_evicted_set.append(TrieNode.get_path_tuple_from_node(target_leaf))

            ## do delete
            target_parent = target_leaf.parent
            del target_parent.children[target_leaf.key]
            self.cur_node_num -= 1
            if len(target_parent.children) == 0:
                leaves.append(target_parent)
            
            leaves.remove(target_leaf)
        
        return to_guard

    def access(self, pc, aligned_address: List) -> Tuple:
        this_node, insert_list = self.__match__(aligned_address)
        to_guard = self.__insert__(this_node, insert_list, aligned_address)

        self.old_visited_set.append(tuple(aligned_address))
        if to_guard:
            self.guarded_set.append(tuple(aligned_address))

        self.after_pred(pc, aligned_address)

        assert len(self.to_fill_nodes) == 0

        return (len(aligned_address), len(aligned_address) - len(insert_list), len(insert_list))

#############################################
class TrieRandAlgorithm(TrieEvictAlgorithm):
    def __init__(self, max_node_num):
        super().__init__(max_node_num)
        self.evictor = RandEvictor()

    def __evict__(self, evict_num, this_node):
        leaves = self.__leaves__()
        if leaves[0] == self.root_node:
            raise ValueError("Can't evict root node")
        if this_node in leaves:
            leaves.remove(this_node)
        
        for i in range(evict_num):
            target_idx = self.evictor.evict(list(enumerate(leaves)))
            target_leaf = leaves[target_idx]
            target_parent = target_leaf.parent
            del target_parent.children[target_leaf.key]
            self.cur_node_num -= 1
            if len(target_parent.children) == 0:
                leaves.append(target_parent)
            leaves.remove(target_leaf)
    
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
        leaves = self.__leaves__()
        if leaves[0] == self.root_node:
            raise ValueError("Can't evict root node")
        if this_node in leaves:
            leaves.remove(this_node)
        
        for i in range(evict_num):
            target_idx = self.evictor.evict(list(enumerate([leaf.metadata for leaf in leaves])))
            target_leaf = leaves[target_idx]

            target_parent = target_leaf.parent
            del target_parent.children[target_leaf.key]
            self.cur_node_num -= 1
            if len(target_parent.children) == 0:
                leaves.append(target_parent)
            leaves.remove(target_leaf)
    
    def access(self, pc, aligned_address: List) -> Tuple:
        this_node, insert_list = self.__match__(aligned_address)
        self.__insert__(this_node, insert_list)
        self.counter += 1
        return (len(aligned_address), len(aligned_address) - len(insert_list), len(insert_list))