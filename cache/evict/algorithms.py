from abc import ABC, abstractmethod
from functools import partial
from typing import List, Union, Type
from cache.evict.evictor import *
from cache.evict.predictor import *
import numpy as np
import types
import copy
import random

class EvictAlgorithm(ABC):
    """Evict an entry from one cache line
    
    Max size is associativity
    """
    def __init__(self, associativity) -> None:
        self.cache = [None] * associativity
        self.pcs = [None] * associativity
        self.associativity = associativity
    
    def snapshot(self):
        return list(zip(self.cache, self.pcs))
    
    @abstractmethod
    def access(self, pc, address) -> bool:
        pass

class PredictAlgorithm(EvictAlgorithm):
    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial]) -> None:
        super().__init__(associativity)
        self.timestamp = 0

        cls_type = predictor_type.func if hasattr(predictor_type, 'func') else predictor_type
        if issubclass(cls_type, ReuseDistancePredictor):
            self.preds = [np.inf] * associativity
        elif issubclass(cls_type, BinaryPredictor):
            self.preds = [0] * associativity
        elif issubclass(cls_type, PhasePredictor):
            self.preds = [1] * associativity
        else:
            self.preds = None
        
        if issubclass(cls_type, OraclePredictor):
            def oracle_access(self, pc, address, next_access_time):
                self.predictor.oracle_access(pc, address, next_access_time)
            self.oracle_access = types.MethodType(oracle_access, self)
        
        self.evictor = evictor_type()
        self.predictor = predictor_type()
    
    def snapshot(self):
        return (list(zip(self.cache, self.pcs)), self.preds)
    
    def before_pred(self, pc, address):
        preds = self.predictor.refresh_scores(self.timestamp, pc, address, self.snapshot()[0])
        if preds is not None:
            self.preds = preds
    
    def after_pred(self, pc ,address, target_index):
        pred = self.predictor.predict_score(self.timestamp, pc, address, self.snapshot()[0])
        if pred is not None:
            self.preds[target_index] = pred
        self.timestamp += 1

    def access(self, pc, address):
        target_index = -1
        hit = False

        self.before_pred(pc, address)
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            target_index = self.evictor.evict(list(enumerate(self.preds)))
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.after_pred(pc, address, target_index)
        return hit

######################################################################

class PredictiveMarker(PredictAlgorithm):
    """
    PredictiveMarker algorithm

    Designed by Thodoris Lykouris and Sergei Vassilvitskii. 2018. Competitive Caching with Machine Learned Advice.
    https://dl.acm.org/doi/10.1145/3447579
    """
    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial]) -> None:
        def harmonic_number(k):
            return sum(1 / i for i in range(1, k + 1))
        super().__init__(associativity, evictor_type, predictor_type)
        self.marked = [0] * associativity
        self.tracking_set = []
        self.h_k = harmonic_number(associativity)
        self.chains_len = []
        self.chains_rep = []
    
    def access(self, pc, address):
        hit = False
        self.before_pred(pc, address)
        target_index = -1

        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if all(mark == 1 for mark in self.marked):
                # new phase
                self.tracking_set = copy.deepcopy(self.cache)
                self.marked = [0] * self.associativity
            if address not in self.tracking_set:
                target_index = self.evictor.evict([(i, self.preds[i]) for i, mark in enumerate(self.marked) if mark == 0])
                self.chains_len.append(1)
                self.chains_rep.append(self.cache[target_index])
            if address in self.tracking_set:
                index = self.chains_rep.index(address)
                if self.chains_len[index] <= self.h_k:
                    target_index = self.evictor.evict([(i, self.preds[i]) for i, mark in enumerate(self.marked) if mark == 0])
                else:
                    target_index = random.choice([i for i, mark in enumerate(self.marked) if mark == 0])
                self.chains_rep[index] = self.cache[target_index]

        self.cache[target_index], self.pcs[target_index] = address, pc
        self.marked[target_index] = 1
        self.after_pred(pc, address, target_index)
        return hit

class LMarker(PredictAlgorithm):
    """
    LMARKER Algorithm

    Designed by Dhruv Rohatgi. 2020. Near-Optimal Bounds for Online Caching with Machine Learned Advice
    https://epubs.siam.org/doi/10.1137/1.9781611975994.112
    """
    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial]) -> None:
        super().__init__(associativity, evictor_type, predictor_type)

        self.stale = []
        self.marked = [0] * associativity
    
    def access(self, pc, address):
        target_index = -1
        hit = False

        self.before_pred(pc, address)
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if all(mark == 1 for mark in self.marked):
                self.stale = copy.deepcopy(self.cache)
                self.marked = [0] * self.associativity
            
            if address in self.stale:
                target_index = random.choice([i for i, mark in enumerate(self.marked) if mark == 0])
            else:
                target_index = self.evictor.evict([(i, self.preds[i]) for i, mark in enumerate(self.marked) if mark == 0])
        
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.marked[target_index] = 1
        self.after_pred(pc, address, target_index)
        return hit

class LNonMarker(PredictAlgorithm):
    """
    LNONMARKER Algorithm

    Designed by Dhruv Rohatgi. 2020. Near-Optimal Bounds for Online Caching with Machine Learned Advice
    https://epubs.siam.org/doi/10.1137/1.9781611975994.112
    """
    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial]) -> None:
        super().__init__(associativity, evictor_type, predictor_type)

        self.phase = set()
        self.stale = []
        self.marked = [0] * associativity
        self.evicts = {}
    
    def access(self, pc, address):
        target_index = -1
        hit = False
        self.before_pred(pc, address)

        if len(self.phase) == self.associativity:
            self.stale = copy.deepcopy(self.cache)
            self.marked = [0] * self.associativity
            self.evicts = {}
            self.phase = set()

        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if address in self.stale:
                if self.evicts[address] not in self.stale:
                    target_index = random.choice(range(self.associativity))
                else:
                    target_index = random.choice([i for i, mark in enumerate(self.marked) if mark == 0])
            else:
                target_index = self.evictor.evict([(i, self.preds[i]) for i, mark in enumerate(self.marked) if mark == 0])
        
        self.evicts[self.cache[target_index]] = address
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.marked[target_index] = 1
        self.phase.add(address)
        self.after_pred(pc, address, target_index)
        return hit

class Mark0(PredictAlgorithm):
    """
    MARK0 Eviction Strategy

    Designed by Antonios Antoniadis, Joan Boyar, Marek Eliáš, Lene M. Favrholdt, Ruben Hoeksma, Kim S. Larsen, Adam Polak, and Bertrand Simon. 2023. Paging with Succinct Prediction.
    https://dl.acm.org/doi/10.5555/3618408.3618447
    """
    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial]):
        super().__init__(associativity, evictor_type, predictor_type)
        if not isinstance(self.predictor, BinaryPredictor):
            raise ValueError('Mark0: predictor must be a BinaryPredictor')
        self.marked = [0] * associativity
        self.S_address = [None] * associativity
        self.S_visited = [0] * associativity
    
    def access(self, pc, address):
        target_index = -1
        hit = False

        self.before_pred(pc, address)
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            if address in self.S_address and 0 in self.S_visited:
                target_index = random.choice([i for i, visited in enumerate(self.S_visited) if visited == 0])
            else:
                target_index = self.cache.index(None)
        else:
            if all(visited == 1 for visited in self.S_visited):
                self.marked = [0] * self.associativity
                self.S_address = copy.deepcopy(self.cache)
                self.S_visited = [0] * self.associativity

            if address in self.S_address and 0 in self.S_visited:
                target_index = random.choice([i for i, visited in enumerate(self.S_visited) if visited == 0])
            else:
                target_index = random.choice([i for i, mark in enumerate(self.marked) if mark == 0])
        
        self.S_address[target_index] = None
        self.S_visited[target_index] = 1
        self.marked[target_index] = 1
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.after_pred(pc, address, target_index)
        if self.preds[target_index] == 1:
            self.cache[target_index], self.pcs[target_index] = None, None
        return hit

class MarkAndPredict(PredictAlgorithm):
    """
    MARK&PREDICT Eviction Strategy

    Designed by Antonios Antoniadis, Joan Boyar, Marek Eliáš, Lene M. Favrholdt, Ruben Hoeksma, Kim S. Larsen, Adam Polak, and Bertrand Simon. 2023. Paging with Succinct Prediction.
    https://dl.acm.org/doi/10.5555/3618408.3618447
    """
    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial]):
        super().__init__(associativity, evictor_type, predictor_type)
        if not isinstance(self.predictor, PhasePredictor):
            raise ValueError('MarkAndPredict: predictor must be a PhasePredictor')
        if not isinstance(self.evictor, BinaryEvictor):
            raise ValueError('MarkAndPredict: evictor must be a BinaryEvictor')
        self.marked = [0] * associativity
    
    def access(self, pc, address):
        target_index = -1
        hit = False

        self.before_pred(pc, address)
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if all(mark == 1 for mark in self.marked):
                self.marked = [0] * self.associativity
            target_index = self.evictor.evict([(i, self.preds[i]) for i, mark in enumerate(self.marked) if mark == 0])
        
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.marked[target_index] = 1
        self.after_pred(pc, address, target_index)
        return hit

class FollowerRobust(PredictAlgorithm):
    """
    F&R Algorithm

    Parameters:

    - a

    - lazy_evictor_type

    Designed by Karim Abdel Sadek and Marek Elias. 2024. Algorithms for Caching and MTS with reduced number of predictions.
    https://arxiv.org/abs/2404.06280
    """
    @staticmethod
    def create_windows(S, W, F, k, a):
        def func(i):    
            return (2**(i+1))-1
        for i in range(0, int(np.log2(k)) + 1):
            S.append((int(k - (k // (2 ** i)) + 1)))
        for i in range(1, int(np.log2(k)) + 1):
            n = []
            for h in range(S[i - 1], S[i]):
                n.append(h)
            W.append(n)
        W.append([S[-1]])
        for g in range(0, len(W)-1):
            gap = int(len(W[g])//(func(g+1)-func(g)))
            if (gap >= a):
                for m in W[g][::gap]:
                    F.append(m)
            else:
                for m in range(S[g], S[-1]+1, a):
                    F.append(m)
                break
        if k == 10:
            F = [1,6,9]
        return S, W, F

    @staticmethod
    def differ(a, b):
        aa = list(a).copy()
        bb = list(b).copy()
        for x in bb:
            if x == None:
                continue
            elif x in aa:
                aa.remove(x)
        if aa == []:
            return bb
        return aa

    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial], **kwargs):
        super().__init__(associativity, evictor_type, predictor_type)
        if not isinstance(self.predictor, StatePredictor):
            raise ValueError('FollowerRobust: predictor must be a StatePredictor')

        if 'a' in kwargs:
            self.a = kwargs['a']
        else:
            self.a = 1
        if 'lazy_evictor_type' in kwargs:
            if kwargs['lazy_evictor_type'] is None:
                self.lazy_evictor = None
            else:
                self.lazy_evictor = kwargs['lazy_evictor_type']()
        else:
            self.lazy_evictor = LRUEvictor()
        self.key_scores = {} if self.lazy_evictor is not None else None
        self.timestamp = 0

        self.remaining_robust_step = 0
        self.follower_cost = 0
        self.belady_cost = 0
        self.pred_gap = 0

        self.sim_cache = [None] * associativity
        self.sim_pcs = [None] * associativity
        self.traces = []
        
        self.old = []
        self.unmarked = []
        self.unmarked_for_reload = []
        self.marked = []
        self.clean = []
        self.S = []
        self.W = []
        self.F = []
        if (self.a == 1):
            self.S, self.W, self.F = FollowerRobust.create_windows(self.S, self.W, self.F, self.associativity, self.a)

    def online_belady(self):
        cache = []
        for i, current in enumerate(self.traces):
            if current in cache:
                continue
            if len(cache) < self.associativity:
                cache.append(current)
            else:
                future_uses = {item: self.traces[i + 1:].index(item) if item in self.traces[i + 1:] else float('inf') for item in cache}
                to_remove = max(future_uses, key=future_uses.get)
                cache.remove(to_remove)
                cache.append(current)
        
        return cache
    
    def follow_robust(self, pc, address):
        target_index = -1
        # get next state
        preds = self.predictor.refresh_scores(self.timestamp, pc, address, self.snapshot()[0])
        assert(preds is not None)
        if address in self.sim_cache:
            target_index = self.sim_cache.index(address)
        elif None in self.sim_cache:
            target_index = self.sim_cache.index(None)
            self.preds = preds
        else:
            if self.remaining_robust_step == 0:
                # follower
                self.follower_cost += 1
                if address not in self.online_belady():
                    self.belady_cost += 1

                if address not in self.preds and (self.follower_cost <= self.belady_cost):
                    if self.pred_gap <= 0:
                        self.preds = preds
                        self.pred_gap = self.a
                        dd = FollowerRobust.differ(self.cache, self.preds)
                        target_index = self.sim_cache.index(random.choice(dd))
                    else:
                        target_index = random.choice(range(self.associativity))
                elif address in self.preds:
                    dd = FollowerRobust.differ(self.cache, self.preds)
                    target_index = self.sim_cache.index(random.choice(dd))
                else:
                    self.follower_cost = 0
                    self.belady_cost = 0
                    self.remaining_robust_step = self.associativity

                    self.old = copy.deepcopy(self.sim_cache)
                    self.unmarked = copy.deepcopy(self.sim_cache)
                    self.marked = []
                    self.unmarked_for_reload = []
                    self.clean = []
                                    
            if self.remaining_robust_step != 0:
                if address not in self.marked:
                    self.remaining_robust_step -= 1
                    this_arrival_index = self.associativity - self.remaining_robust_step
                    if address in self.unmarked:
                        self.unmarked.remove(address)
                    if address not in self.marked:
                        self.marked.append(address)
                    if address not in self.old:
                        self.clean.append(address)
                    if ((self.a==1) and (this_arrival_index in self.F)) or ((self.a > 1) and (self.pred_gap <= 0)):
                        self.pred_gap = self.a
                        self.preds = preds
                    if this_arrival_index in self.S:
                        self.unmarked_for_reload = []
                        for p in self.unmarked:
                            if (p in self.preds) and (p not in self.sim_cache):
                                self.unmarked_for_reload.append(p)
                    if address in self.unmarked_for_reload:
                        dd = FollowerRobust.differ(self.cache, self.preds)
                        target_index = self.sim_cache.index(random.choice(dd))
                    if address in self.clean:
                        dd = FollowerRobust.differ(self.cache, self.preds)
                        target_index = self.sim_cache.index(random.choice(dd))
                if target_index == -1:
                    # random select unmark
                    unmarked_slots = []
                    for page in self.sim_cache:
                        if page in self.unmarked:
                            unmarked_slots.append(self.sim_cache.index(page))
                    target_index = random.choice(unmarked_slots)
        
        self.sim_cache[target_index], self.sim_pcs[target_index] = address, pc
        self.after_pred(pc, address, target_index)
        self.pred_gap -= 1
        self.traces.append(address)
    
    def access(self, pc, address):
        if self.key_scores is not None:
            self.key_scores[address] = self.timestamp
        self.timestamp += 1

        self.follow_robust(pc, address)

        ## Lazy
        target_index = -1
        hit = False
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            center_cache = self.sim_cache
            if self.lazy_evictor is None:
                self.cache = copy.deepcopy(center_cache)
                self.pcs = copy.deepcopy(self.sim_pcs)
                target_index = self.cache.index(address)
            else:
                diff_keys = set(center_cache) - set(self.cache)
                target_index = self.lazy_evictor.evict([(center_cache.index(k), self.key_scores[k] if self.key_scores is not None else 0) for k in diff_keys])
        
        self.cache[target_index], self.pcs[target_index] = address, pc
        return hit

class Guard(PredictAlgorithm):
    """
    Guard algorithm

    Parameters:
    
    - follow_if_guarded

    - relax_times

    - relax_prob

    Our work
    """
    def __init__(self, associativity, evictor_type: Union[Type[Evictor], partial], predictor_type: Union[Predictor, partial], **kwargs) -> None:
        super().__init__(associativity, evictor_type, predictor_type)
        self.old_unvisited_set = []
        self.unguarded_set = []
        self.phase_evicted_set = set()
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
    
    def access(self, pc, address):
        to_guard = False
        target_index = -1
        hit = False

        self.before_pred(pc, address)
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if not self.old_unvisited_set:
                self.old_unvisited_set = list(range(self.associativity))
                self.unguarded_set = list(range(self.associativity))
                self.phase_evicted_set = set()
                self.error_times = 0
            
            if address in self.phase_evicted_set:
                if self.relax_times != 0:
                    self.error_times += 1
                    if self.error_times >= self.relax_times:
                        to_guard = True
                else:
                    if random.random() > self.relax_prob:
                        to_guard = True

            if to_guard and not self.follow_if_guarded:
                target_index = random.choice(self.old_unvisited_set)
            else:
                target_index = self.evictor.evict([(i, self.preds[i]) for i in self.unguarded_set])
            
            self.phase_evicted_set.add(self.cache[target_index])

        if target_index in self.old_unvisited_set:
            self.old_unvisited_set.remove(target_index)

        if to_guard:
            self.unguarded_set.remove(target_index)
        
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.after_pred(pc, address, target_index)
        return hit

#######################################################################

class CombineAlgorithm(EvictAlgorithm):
    def __init__(self, associativity, candidate_algorithms: List[Union[EvictAlgorithm, partial]], lazy_evictor_type: Union[LRUEvictor, RandEvictor, None] = LRUEvictor):
        if lazy_evictor_type is not None and not issubclass(lazy_evictor_type, Evictor):
            raise ValueError('CombineAlgorithm: Invalid Evictor')
        
        super().__init__(associativity)
        self.oracle_algs = []
        self.candidate_algs = []
        self.center = 0
        self.timestamp = 0
        self.lazy_evictor = lazy_evictor_type() if lazy_evictor_type is not None else None
        self.key_scores = {} if lazy_evictor_type == LRUEvictor else None

        for alg_type in candidate_algorithms:
            alg_instance = alg_type(associativity)
            self.candidate_algs.append([alg_instance, 0])
            if hasattr(alg_instance, 'oracle_access'):
                self.oracle_algs.append(alg_instance)

        if len(self.oracle_algs) != 0:
            def oracle_access(self, pc, address, next_access_time):
                for oracle_alg in self.oracle_algs:
                    oracle_alg.oracle_access(pc, address, next_access_time)
            self.oracle_access = types.MethodType(oracle_access, self)
        
        if len(self.candidate_algs) < 2:
            raise ValueError('CombineAlgorithm: Algorithm Count < 2')

    def __push_candidates__(self, pc, address):
        for i, (alg, _) in enumerate(self.candidate_algs):
            if not alg.access(pc, address):
                self.candidate_algs[i][1] += 1
                self.__trigger_miss__(i, address)
        
        if self.key_scores is not None:
            self.key_scores[address] = self.timestamp
        self.timestamp += 1
    
    def __trigger_miss__(self, i, address):
        pass

    @abstractmethod
    def __trigger_elect_center__(self):
        pass

    def access(self, pc, address):
        self.__push_candidates__(pc, address)

        target_index = -1
        hit = False
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            self.__trigger_elect_center__()
            center_cache = self.candidate_algs[self.center][0].cache
            if self.lazy_evictor is None:
                self.cache = copy.deepcopy(center_cache)
                self.pcs = copy.deepcopy(self.candidate_algs[self.center][0].pcs)
                target_index = self.cache.index(address)
            else:
                diff_keys = set(center_cache) - set(self.cache)
                target_index = self.lazy_evictor.evict([(center_cache.index(k), self.key_scores[k] if self.key_scores is not None else 0) for k in diff_keys])
        
        self.cache[target_index] = address
        self.pcs[target_index] = pc
        return hit

class CombineDeterministicAlgorithm(CombineAlgorithm):
    """
    black-box algorithm

    Designed by Thodoris Lykouris and Sergei Vassilvitskii. 2018. Competitive Caching with Machine Learned Advice.
    https://dl.acm.org/doi/10.1145/3447579
    """
    def __init__(self, associativity, candidate_algorithms: List[Union[EvictAlgorithm, partial]], switch_bound=2, lazy_evictor_type: Union[LRUEvictor, RandEvictor, None] = LRUEvictor):
        super().__init__(associativity, candidate_algorithms, lazy_evictor_type)
        self.switch_bound = switch_bound

    def __trigger_elect_center__(self):
        this_cost = self.candidate_algs[self.center][1]
        min_center, (_, min_cost) = min(enumerate(self.candidate_algs), key=lambda x: x[1][1])
        if this_cost >= self.switch_bound * min_cost:
            self.center = min_center

class CombineRandomAlgorithm(CombineAlgorithm):
    """
    Algorithm THRESH

    Designed by Avrim Blum and Carl Burch. 1997. On-line learning and the metrical task system problem.
    https://dl.acm.org/doi/10.1145/267460.267475
    """
    def __init__(self, associativity, candidate_algorithms: List[Union[EvictAlgorithm, partial]], alpha=0.0, beta=0.99, lazy_evictor_type: Union[LRUEvictor, RandEvictor, None] = LRUEvictor):
        super().__init__(associativity, candidate_algorithms, lazy_evictor_type)
        self.alpha = alpha
        self.beta = beta
        self.n = len(self.candidate_algs)
        self.weights = [1] * self.n
    
    def __trigger_miss__(self, i, key):
        self.weights[i] *= self.beta
    
    def __trigger_elect_center__(self):
        W = sum(self.weights)
        threshold = self.alpha * W / self.n
        valid_index, valid_weights = zip(*[(i, weight) for i, weight in enumerate(self.weights) if weight > threshold])
        if valid_weights:
            self.center = random.choices(valid_index, weights=valid_weights)[0]

class CombineWeightsAlgorithm(CombineAlgorithm):
    """
    Imitation learing for Parrot
    """
    def __init__(self, associativity, candidate_algorithms: List[Union[EvictAlgorithm, partial]], weights: Union[List[float], None], lazy_evictor_type: Union[LRUEvictor, RandEvictor, None] = LRUEvictor):
        super().__init__(associativity, candidate_algorithms, lazy_evictor_type)
        self.n = len(self.candidate_algs)
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [1] * self.n
    
    def snapshot(self):
        return (list(zip(self.cache, self.pcs)), self.candidate_algs[self.center][0].preds)

    def reset(self, weights):
        self.weights = weights

    def __trigger_elect_center__(self):
        self.center = random.choices(list(range(self.n)), weights=self.weights)[0]

#######################################################################

class RandAlgorithm(EvictAlgorithm):
    def __init__(self, associativity):
        super().__init__(associativity)
        self.evictor = RandEvictor()
    
    def access(self, pc, address):
        target_index = -1
        hit = False
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            target_index = self.evictor.evict(list(enumerate(self.cache)))
        
        self.cache[target_index] = address
        self.pcs[target_index] = pc
        return hit

class LRUAlgorithm(EvictAlgorithm):
    def __init__(self, associativity):
        super().__init__(associativity)
        self.evictor = LRUEvictor()
        self.scores = [0] * associativity
        self.timestamp = 0
    
    def access(self, pc, address):
        target_index = -1
        hit = False
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            target_index = self.evictor.evict(list(enumerate(self.scores)))
        
        self.cache[target_index] = address
        self.pcs[target_index] = pc
        self.scores[target_index] = self.timestamp
        self.timestamp += 1
        return hit

class MarkerAlgorithm(EvictAlgorithm):
    def __init__(self, associativity):
        super().__init__(associativity)
        self.evictor = MarkerEvictor()
        self.scores = [0] * associativity
    
    def access(self, pc, address):
        if all(x == 1 for x in self.scores):
            self.scores = [0] * self.associativity

        target_index = -1
        hit = False
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            target_index = self.evictor.evict(list(enumerate(self.scores)))
        
        self.cache[target_index] = address
        self.pcs[target_index] = pc
        self.scores[target_index] = 1
        return hit

####################################################################

class PredictAlgorithmFactory:
    predictor_evict_dict = {
        "PLECO": (MaxEvictor, PLECOPredictor),
        "PLECO-State": (DummyEvictor, PLECOStatePredictor),
        "POPU": (MaxEvictor, POPUPredictor),
        "POPU-State": (DummyEvictor, POPUStatePredictor),
        "Parrot": (MaxEvictor, ParrotPredictor),
        "OracleDis": (ReuseDistanceEvictor, OracleReuseDistancePredictor),
        "OracleBin": (BinaryEvictor, OracleBinaryPredictor),
        "OraclePhase": (BinaryEvictor, OraclePhasePredictor),
        "OracleState": (DummyEvictor, OracleStatePredictor),
    }

    @staticmethod
    def generate_predictive_algorithm(alg_type: Union[Type[PredictAlgorithm], partial], pred_type_str: str, **kwargs) -> partial:
        evictor_type, predictor_type = PredictAlgorithmFactory.predictor_evict_dict[pred_type_str]
        
        evictor_partial = evictor_type
        predictor_partial = predictor_type
        if pred_type_str == 'Parrot':
            # shared_model
            if 'shared_model' not in kwargs:
                raise ValueError('PredictAlgorithmFactory: Parrot need [shared_model]')
            predictor_partial = partial(predictor_type, shared_model=kwargs['shared_model'])
        elif pred_type_str.startswith('Oracle'):
            reuse_dis_noise_sigma = 0
            lognormal = True
            if 'reuse_dis_noise_sigma' in kwargs:
                reuse_dis_noise_sigma = kwargs['reuse_dis_noise_sigma']
            if 'lognormal' in kwargs:
                lognormal = kwargs['lognormal']

            if pred_type_str == 'OracleDis':
                predictor_partial = partial(predictor_type, reuse_dis_noise_sigma=reuse_dis_noise_sigma, lognormal=lognormal)
            else:
                if 'associativity' not in kwargs:
                    raise ValueError(f'PredictAlgorithmFactory: {pred_type_str} need [associativity]')
                associativity = kwargs['associativity']
                if pred_type_str == 'OracleState':
                    predictor_partial = partial(predictor_type, associativity=associativity, reuse_dis_noise_sigma=reuse_dis_noise_sigma, lognormal=lognormal)
                else:
                    bin_noise_prob = 0
                    if 'bin_noise_prob' in kwargs:
                        bin_noise_prob = kwargs['bin_noise_prob']
                    predictor_partial = partial(predictor_type, associativity=associativity, bin_noise_prob=bin_noise_prob, reuse_dis_noise_sigma=reuse_dis_noise_sigma, lognormal=lognormal)
        elif pred_type_str.endswith('State'):
            if 'associativity' not in kwargs:
                raise ValueError(f'PredictAlgorithmFactory: {pred_type_str} need [associativity]')
            associativity = kwargs['associativity']
            predictor_partial = partial(predictor_type, associativity=associativity)

        if isinstance(alg_type, partial):
            this_partial = copy.deepcopy(alg_type)
            this_partial.keywords['evictor_type'] = evictor_partial
            this_partial.keywords['predictor_type'] = predictor_partial
            return this_partial
        else:
            return partial(alg_type, evictor_type=evictor_partial, predictor_type=predictor_partial)

def format_guard(relax_times, relax_prob):
    if relax_times == 0 and relax_prob == 0:
        return "-no-relax"
    elif relax_times == 0 and relax_prob != 0:
        return f"-relax-prob-{relax_prob}"
    elif relax_times != 0 and relax_prob == 0:
        return f"-relax-times-{relax_times}"
    else:
        raise ValueError('relax_times and relax_prob invaild')

def format_oracle(reuse_dis_noise_sigma, bin_noise_prob):
    if reuse_dis_noise_sigma == 0 and bin_noise_prob == 0:
        return "-oracle"
    elif reuse_dis_noise_sigma == 0 and bin_noise_prob != 0:
        return f"-bin-{bin_noise_prob}"
    elif reuse_dis_noise_sigma != 0 and bin_noise_prob == 0:
        return f"-dis-{reuse_dis_noise_sigma}"
    else:
        return f"-dis-{reuse_dis_noise_sigma}-bin-{bin_noise_prob}"

def pretty_print(callable: Union[EvictAlgorithm, partial], verbose=False) -> str:
    this_cls = callable
    if hasattr(callable, 'func'):
        this_cls = callable.func
    this_cls_name = this_cls.__name__.replace("Algorithm", '').replace("CombineDeterministic", 'CombDet').replace('CombineRandomAlgorithm', 'CombRand').replace("MarkAndPredict", "Mark&Predict").replace('PredictiveMarker', 'PredMark')
    metadata = this_cls_name
    if hasattr(callable, 'keywords'):
        kw = callable.keywords
        if issubclass(this_cls, CombineAlgorithm):
            algs = kw['candidate_algorithms']
            alg_names = []
            for alg in algs:
                alg_names.append(pretty_print(alg, verbose))
            metadata += ("[" + (", ".join(alg_names)) + "]")
        
        if 'predictor_type' in kw:
            predictor_type = kw['predictor_type']
            pred_kw = {}
            if hasattr(predictor_type, 'func'):
                pred_kw = predictor_type.keywords
                predictor_type = predictor_type.func
            predictor = predictor_type.__name__.replace("Predictor", '').replace('OracleReuseDistance', 'Belady').replace('OracleBinary', 'FBP')
            metadata += f'[{predictor}]'

            if issubclass(predictor_type, OraclePredictor) and verbose:
                reuse_dis_noise_sigma = bin_noise_prob = 0
                if 'reuse_dis_noise_sigma' in pred_kw:
                    reuse_dis_noise_sigma = pred_kw['reuse_dis_noise_sigma']
                if 'bin_noise_prob' in pred_kw:
                    bin_noise_prob = pred_kw['bin_noise_prob']
                metadata += format_oracle(reuse_dis_noise_sigma, bin_noise_prob) 

        if issubclass(this_cls, Guard):
            follow_if_guarded = False
            relax_times = relax_prob = 0
            if 'follow_if_guarded' in kw:
                follow_if_guarded = kw['follow_if_guarded']
            if follow_if_guarded:
                metadata += '-unv'
            else:
                metadata += '-f-pred'
            if 'relax_times' in kw:
                relax_times = kw['relax_times']
            if 'relax_prob' in kw:
                relax_prob = kw['relax_prob']
            metadata += format_guard(relax_times, relax_prob)
            
    return metadata