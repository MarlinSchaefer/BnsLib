import numpy as np
import threading
import queue
import time
import multiprocessing as mp

from .file_generator import FileGenerator, format_batch
from ....types import DictList
from ....utils import input_to_list


class GroupedIndexFileGenerator(FileGenerator):
    """An extension of the FileGenerator that allows to group indices
    and shuffle them only within these groups.
    
    This behavior can be useful if one wants to load data from many
    files. If a single file should be loaded in one call it might be
    convenient to group the requested indices by their files.
    
    Arguments
    ---------
    *args :
        All positional arguments are passed to FileGenerator. For details
        please refer to its documentation.
    group_by : {list of int or callable or None, None}
        Specify how to group the inidices. If a list of integers is
        provided it is understood that the elements specify the number
        of indices in each group in the order in which they appear in the
        index_list. If the sum of the numbers is smaller than the length
        of the index_list a final group containing the remaining indices is
        added. If the argument is a callable it is assumed to take any index
        from the index_list as argument and return a hashable and sortable
        value specifying the group of the particular index. All indices are
        then grouped by the returned value of the callable.
    shuffle_groups : {bool, True}
        Whether or not to shuffle the order of the groups. Shuffling the group
        members within each group is controlled by the `shuffle` argument.
    **kwargs :
        All other keyword-arguments are passed to FileGenerator. For details
        please refer to its documentation.
    """
    def __init__(self, *args, group_by=None, shuffle_groups=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if group_by is None:
            self.index_groups = [np.arange(len(self.index_list), dtype=int)]
        elif callable(group_by):  # Assume group_by is function
            tmp = DictList()
            for i, idx in enumerate(self.index_list):
                group = group_by(idx)
                tmp.append(group, i)
            self.index_groups = [np.array(tmp[key], dtype=int)
                                 for key in sorted(tmp.as_dict())]
        else:  # Assume group_by is iterable
            group_by = list(group_by)
            if sum(group_by) < len(self.index_list):
                group_by.append(len(self.index_list) - sum(group_by))
            self.index_groups = []
            sidx = 0
            for i in range(len(group_by)):
                eidx = sidx + group_by[i]
                self.index_groups.append(np.arange(sidx, eidx, dtype=int))
                sidx = eidx
        self.shuffle_groups = bool(shuffle_groups)
        self.on_epoch_end()
    
    def on_epoch_end(self):
        # The __init__ method of the FileGenerator calls this method
        # before construction is complete and self.index_groups is
        # available.
        if hasattr(self, 'index_groups'):
            self.apply_shuffle()
            self.set_indices()
            self.cbatch = 0
    
    def apply_shuffle(self):
        if self.shuffle:
            for group in self.index_groups:
                np.random.shuffle(group)
            if self.shuffle_groups:
                np.random.shuffle(self.index_groups)
    
    def set_indices(self):
        self.indices = np.concatenate(self.index_groups)


class PrefetchedFileGenerator(GroupedIndexFileGenerator):
    def __init__(self, *args, prefetch=None, workers=None, timeout=0.01,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.prefetch = prefetch if prefetch is not None else 0
        self.workers = workers
        self.timeout = timeout
        self._init_queues()
    
    def _init_queues(self):
        if self.prefetch > 0 and self.workers is not None:
            if not hasattr(self, 'fetched'):
                self.fetched = queue.Queue(maxsize=2*self.prefetch)
            if not hasattr(self, 'index_queue'):
                self.index_queue = queue.Queue(maxsize=2*self.prefetch)
            self.last_fetched = -1
            self.last_index_put = -1
    
    def __getitem__(self, index):
        if index == 0:
            self.empty_queues()
        if self.workers is None or self.prefetch < 1:
            return super().__getitem__(index)
        else:
            upper = min(index + self.prefetch, len(self))
            if upper > self.last_index_put:
                for i in range(self.last_index_put+1, upper):
                    self.index_queue.put(i)
                    self.last_index_put = i
                if len(self) <= upper:
                    self.last_index_put = len(self)
            while True:
                try:
                    data = self.fetched.get(timeout=self.timeout)
                    break
                except queue.Empty:
                    continue
            return data
    
    def empty_queues(self):
        if hasattr(self, 'fetched'):
            while True:
                try:
                    self.fetched.get(timeout=0.01)
                except queue.Empty:
                    break
            while True:
                try:
                    self.index_queue.get(timeout=0.01)
                except queue.Empty:
                    break
            self._init_queues()
    
    def on_epoch_end(self):
        super().on_epoch_end()
        self.empty_queues()
    
    def fetch_func(self, idx, index_pipe, output_pipe, event):
        data = None
        index = None
        while not event.is_set():
            if data is None:
                try:
                    index = index_pipe.get(timeout=self.timeout)
                    data = super().__getitem__(index)
                except queue.Empty:
                    continue
            try:
                if self.last_fetched + 1 != index:
                    time.sleep(self.timeout)
                else:
                    if not event.is_set():
                        output_pipe.put(data, timeout=self.timeout)
                        with self.lock:
                            self.last_fetched = index
                        data = None
            except queue.Full:
                continue
    
    def __enter__(self):
        if self.workers is not None and self.workers > 0 and self.prefetch > 0:
            self.event = threading.Event()
            self.lock = threading.Lock()
            self.threads = []
            for i in range(self.workers):
                thread = threading.Thread(target=self.fetch_func,
                                          args=(i,
                                                self.index_queue,
                                                self.fetched,
                                                self.event))
                self.threads.append(thread)
                thread.start()
        return
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if hasattr(self, 'event'):
            self.event.set()
            if hasattr(self, 'threads'):
                while len(self.threads) > 0:
                    thread = self.threads.pop(0)
                    thread.join()
            self.event = None


class PrefetchedFileGeneratorMP(PrefetchedFileGenerator):
    """Multiprocessing version of the PrefetchedFileGenerator. It is a
    drop-in replacement, where multiple processes rather than multiple
    threads are required.
    
    The file-handler used to generate the data must implement the two
    methods
        -serialize
        -from_serialized
    The serialize method should convert the object into a serialized
    object, i.e. a construct that can be stored by the json module. The
    from_serialized method has to be a class-method that returns a copy
    of the serialized object.
    This is used to ensure that each process can have its own local copy
    of the handler to guard against problems when accessing files.
    """
    def __init__(self, *args, synced=None, **kwargs):
        workers = kwargs.get('workers', None)
        if workers is not None and workers < 0:
            workers = mp.cpu_count()
        kwargs['workers'] = workers
        super().__init__(*args, **kwargs)
        if synced is None:
            synced = []
        synced = input_to_list(synced)
        self.synced = ['timeout',
                       'batch_size',
                       'indices',
                       'index_list']
        self.synced.extend(synced)
    
    def _init_queues(self):
        if self.prefetch > 0 and self.workers is not None:
            if not hasattr(self, 'fetched'):
                self.fetched = mp.Queue(maxsize=2*self.prefetch)
            if not hasattr(self, 'index_queue'):
                self.index_queue = mp.Queue(maxsize=2*self.prefetch)
            if hasattr(self, 'last_fetched'):
                self.last_fetched.value = -1
            else:
                self.last_fetched = mp.Value('i', -1)
            self.last_index_put = -1
    
    def fetch_func(self, idx, index_pipe, output_pipe, event, msg_pipe):
        data = None
        index = None
        fh = self.file_handler.from_serialized(self.file_handler.serialize())
        with fh:
            while not event.is_set():
                while msg_pipe.poll():
                    name, value = msg_pipe.recv()
                    setattr(self, name, value)
                if data is None:
                    try:
                        index = index_pipe.get(timeout=self.timeout)
                        if (index + 1) * self.batch_size > len(self.indices):
                            batch = self.indices[index*self.batch_size:]
                        else:
                            batch = self.indices[index*self.batch_size:(index+1)*self.batch_size]  # noqa: E501
                        data = [fh[self.index_list[i]] for i in batch]
                        data = format_batch(data,
                                            input_shape=fh.input_shape,
                                            output_shape=fh.output_shape)
                    except queue.Empty:
                        continue
                try:
                    if self.last_fetched.value + 1 != index:
                        time.sleep(self.timeout)
                    else:
                        if not event.is_set():
                            output_pipe.put(data, timeout=self.timeout)
                            self.last_fetched.value = index
                            data = None
                except queue.Full:
                    continue
    
    def __enter__(self):
        if self.workers is not None and self.workers > 0 and self.prefetch > 0:
            self.event = mp.Event()
            self.processes = []
            self.pipes = []
            for i in range(self.workers):
                send_pipe, recv_pipe = mp.Pipe()
                process = mp.Process(target=self.fetch_func,
                                     args=(i,
                                           self.index_queue,
                                           self.fetched,
                                           self.event,
                                           recv_pipe))
                self.processes.append(process)
                process.start()
                self.pipes.append((send_pipe, recv_pipe))
            self.subprocess = False
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if hasattr(self, 'event'):
            self.event.set()
            if hasattr(self, 'processes'):
                for spipe, rpipe in self.pipes:
                    while rpipe.poll():
                        try:
                            rpipe.recv(timeout=0.01)
                        except queue.Empty:
                            break
                    while spipe.poll():
                        try:
                            spipe.recv(timeout=0.01)
                        except queue.Empty:
                            break
                    spipe.close()
                    rpipe.close()
                self.empty_queues()
                while len(self.processes) > 0:
                    process = self.processes.pop(0)
                    process.join()
            self.event = None
    
    def __setattr__(self, name, value):
        if hasattr(self, 'synced'):
            conditions = [name in self.synced,
                          hasattr(self, 'pipes'),
                          hasattr(self, 'subprocess')]
            if all(conditions) and not self.subprocess:
                self._send((name, value))
        super().__setattr__(name, value)
    
    def _send(self, item):
        for spipe, rpipe in self.pipes:
            spipe.send(item)


class ScaledPrefetchedGeneratorMP(PrefetchedFileGeneratorMP):
    """Multiprocessing version of the PrefetchedFileGenerator. It is a
    drop-in replacement, where multiple processes rather than multiple
    threads are required.
    
    The file-handler used to generate the data must implement the two
    methods
        -serialize
        -from_serialized
    The serialize method should convert the object into a serialized
    object, i.e. a construct that can be stored by the json module. The
    from_serialized method has to be a class-method that returns a copy
    of the serialized object.
    This is used to ensure that each process can have its own local copy
    of the handler to guard against problems when accessing files.
    """
    def __init__(self, *args, target=None, **kwargs):
        super().__init__(*args, **kwargs)
        if 'target' not in self.synced:
            self.synced.append('target')
        if 'rescaled' not in self.synced:
            self.synced.append('rescaled')
        self.rescaled = None
        self.rescale(target)
    
    def fetch_func(self, idx, index_pipe, output_pipe, event, msg_pipe):
        data = None
        index = None
        rescaled = self.rescaled
        fh = self.file_handler.from_serialized(self.file_handler.serialize())
        fh.rescale(self.target)
        with fh:
            while not event.is_set():
                while msg_pipe.poll():
                    name, value = msg_pipe.recv()
                    setattr(self, name, value)
                if rescaled != self.rescaled:
                    rescaled = self.rescaled
                    fh.rescale(self.target)
                if data is None:
                    try:
                        index = index_pipe.get(timeout=self.timeout)
                        if (index + 1) * self.batch_size > len(self.indices):
                            batch = self.indices[index*self.batch_size:]
                        else:
                            batch = self.indices[index*self.batch_size:(index+1)*self.batch_size]  # noqa: E501
                        data = [fh[self.index_list[i]] for i in batch]
                        data = format_batch(data,
                                            input_shape=fh.input_shape,
                                            output_shape=fh.output_shape)
                    except queue.Empty:
                        continue
                try:
                    if self.last_fetched.value + 1 != index:
                        time.sleep(self.timeout)
                    else:
                        output_pipe.put(data, timeout=self.timeout)
                        self.last_fetched.value = index
                        data = None
                except queue.Full:
                    continue
    
    def rescale(self, target):
        self.rescaled = time.perf_counter()
        self.target = target
