import collections
import os

import tqdm
from cache.hash import HashFunction
from data_trace.data_trace import CSVIntListReader
from utils.aligner import Aligner


class TrieDataTrace(object):
    """Represents the ordered load calls for some program with a cursor.

    Should be used in a with block.
    """
    dummy_pc = 0
    def __init__(self, filename, max_look_ahead=int(1e7)):
        """Constructs from a file containing the memory trace.

        Args:
        filename (str): filename of the file containing the memory trace. Must
            conform to one of the expected .csv or .txt formats.
        max_look_ahead (int): number of load calls to look ahead in
            most_future_access(). All addresses not been loaded by the
            max_look_ahead limit are considered tied.
        cache_line_size (int): size of cache line used in Cache reading this
            trace.
        """
        

        self._filename = filename
        self._max_look_ahead = max_look_ahead

        self._num_next_calls = 0

        # Maps address --> list of next access times in the look ahead buffer
        self._look_ahead_buffer = collections.deque()

        # Optimization: only catch the StopIteration in _read_next once.
        # Without this optimization, the StopIteration is caught max_look_ahead
        # times.
        self._reader_exhausted = False

    def _read_next(self):
        """Adds the next row in the CSV memory trace to the look-ahead buffer.

        Does nothing if the cursor points to the end of the trace.
        """
        if self._reader_exhausted:
            return

        try:
            address = self._reader.next()
            self._look_ahead_buffer.append(address)
        except StopIteration:
            self._reader_exhausted = True

    def next(self):
        """The next load call under the cursor. Advances the cursor.

        Returns:
        load_call (tuple)
        """
        self._num_next_calls += 1
        address = self._look_ahead_buffer.popleft()

        self._read_next()
        return TrieDataTrace.dummy_pc, address

    def done(self):
        """True if the cursor points to the end of the trace."""
        return not self._look_ahead_buffer

    def __enter__(self):
        self._file = open(self._filename, "r")
        filename = os.path.basename(self._filename)
        _, extension = os.path.splitext(self._filename)
        if extension == ".csv":
            self._reader = CSVIntListReader(self._file)
        else:
            raise ValueError(
                "Extension {} not a supported extension.".format(extension))

        # Initialize look-ahead buffer
        for _ in tqdm.tqdm(
            range(self._max_look_ahead), desc="Initializing DataTrace"):
            self._read_next()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.close()


class OracleTrieDataTrace(object):
    """Represents the ordered load calls for some program with a cursor.

    Should be used in a with block.
    """
    dummy_pc = 0
    def __init__(self, filename, aligner: Aligner, hasher: HashFunction, scale_times=1, offset=1, max_look_ahead=int(1e7)):
        """Constructs from a file containing the memory trace.

        Args:
        filename (str): filename of the file containing the memory trace. Must
            conform to one of the expected .csv or .txt formats.
        max_look_ahead (int): number of load calls to look ahead in
            most_future_access(). All addresses not been loaded by the
            max_look_ahead limit are considered tied.
        cache_line_size (int): size of cache line used in Cache reading this
            trace.
        """
        

        self._filename = filename
        self._max_look_ahead = max_look_ahead

        self._num_next_calls = 0

        # Maps address --> list of next access times in the look ahead buffer
        self._bucket_access_times = collections.defaultdict(collections.deque)
        self._look_ahead_buffer = collections.deque()
        self._aligner = aligner
        self._hasher = hasher
        self._bucket_counter = [0] * self._hasher._num_buckets
        self._scale = scale_times
        self._offset = offset

        # Optimization: only catch the StopIteration in _read_next once.
        # Without this optimization, the StopIteration is caught max_look_ahead
        # times.
        self._reader_exhausted = False

    
    def get_key(self, pc, address):
        aligned_address = self._aligner.get_aligned_addr(address)
        return [tuple(aligned_address[:i]) for i in range(1, len(aligned_address)+1)]
    
    def get_bucket_idx(self, pc, address):
        return self._hasher.get_bucket_index(self._aligner.get_aligned_addr(address), pc)

    def _read_next(self):
        """Adds the next row in the CSV memory trace to the look-ahead buffer.

        Does nothing if the cursor points to the end of the trace.
        """
        if self._reader_exhausted:
            return

        try:
            address = self._reader.next()
            self._look_ahead_buffer.append(address)

            this_bucket_idx = self.get_bucket_idx(OracleTrieDataTrace.dummy_pc, address)
            aligned_address_list = self.get_key(OracleTrieDataTrace.dummy_pc, address)
            for aligned_addr in aligned_address_list:
                self._bucket_access_times[aligned_addr].append(self._bucket_counter[this_bucket_idx])
            self._bucket_counter[this_bucket_idx] += 1

        except StopIteration:
            self._reader_exhausted = True

    def next(self):
        """The next load call under the cursor. Advances the cursor.

        Returns:
        load_call (tuple)
        """
        self._num_next_calls += 1
        address = self._look_ahead_buffer.popleft()
        # Align to cache line
        aligned_address_list = self.get_key(OracleTrieDataTrace.dummy_pc, address)
        for aligned_addr in aligned_address_list:
            self._bucket_access_times[aligned_addr].popleft()
            if not self._bucket_access_times[aligned_addr]:
                del self._bucket_access_times[aligned_addr]

        self._read_next()
        return OracleTrieDataTrace.dummy_pc, address

    def done(self):
        """True if the cursor points to the end of the trace."""
        return not self._look_ahead_buffer
    
    def next_bucket_access_time_by_address(self, address):
        res = []
        this_bucket_idx = self.get_bucket_idx(OracleTrieDataTrace.dummy_pc, address)
        aligned_address_list = self.get_key(OracleTrieDataTrace.dummy_pc, address)
        for aligned_addr in aligned_address_list:
            accesses = self._bucket_access_times[aligned_addr]
            if not accesses:
                bucket_size = self._bucket_counter[this_bucket_idx]
                res.append(bucket_size * self._scale + self._offset)
            else:
                res.append(accesses[0])
        return res

    def __enter__(self):
        self._file = open(self._filename, "r")
        filename = os.path.basename(self._filename)
        _, extension = os.path.splitext(self._filename)
        if extension == ".csv":
            self._reader = CSVIntListReader(self._file)
        else:
            raise ValueError(
                "Extension {} not a supported extension.".format(extension))

        # Initialize look-ahead buffer
        # for _ in tqdm.tqdm(
        #     range(self._max_look_ahead), desc="Initializing OracleDataTrace"):
        #     self._read_next()
        
        for _ in range(self._max_look_ahead):
            self._read_next()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.close()