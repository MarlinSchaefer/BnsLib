import time
import numpy as np
import pandas as pd

from ..types import DictList


class Profiler(object):
    """A very basic code profiler based on the time.perf_counter method.

    Arguments
    ---------
    printn : {None or int, None}
        How many of the top entries to print. If set to None, all
        entries will be printed.
    sorting : {None or `name` or `ncals` or `tottime` or `avg_time`, None}
        What to sort by.
    ascending : {bool, False}
        Whether to sort the entries in ascending or descending order.

    Usage
    -----
    The profiler has to be started and stopped with a given name
    manually. This can be made simple using a `with` statement. All
    calls to the profiler with the same name are counted as the same
    call. So names for function-calls that should be separated have to
    be unique.

    >>> from BnsLib.utils import Profiler
    >>> profiler = Profiler()
    >>> with profiler('entry1'):
    >>>     a = [0 for _ in range(100)]
    >>> profiler.print()
    
    Instead of using `with`-statements one can also directly call the
    start and stop methods.

    >>> profiler.start('entry2')
    >>> b = [1 for _ in range(1000)]
    >>> profiler.end('entry2')
    >>> profiler.sort('tottime')
    >>> profiler.print()
    """
    def __init__(self, printn=None, sorting=None, ascending=False):
        self.stats = {'start': DictList(),
                      'end': DictList()}
        self.printn = printn
        self.sorting = 'none' if sorting is None else sorting
        self.ascending = ascending

    def __call__(self, name):
        return CoreProfiler(name, self.stats)

    def sort(self, sorting, ascending=False):
        if sorting is None:
            self.sorting = 'none'
        else:
            self.sorting = sorting.lower()
        self.ascending = ascending

    @property
    def dict(self):
        data = DictList({'name': [],
                         'ncals': [],
                         'tottime': [],
                         'avg_time': []})
        starts = self.stats['start']
        ends = self.stats['end']
        for key in starts.keys():
            start = np.array(starts[key])
            end = np.array(ends[key])
            data.append('name', key)
            data.append('ncals', len(start))
            data.append('tottime', np.sum(end - start))
            data.append('avg_time', np.mean(end - start))
        return data.as_dict()

    @property
    def dataframe(self):
        return pd.DataFrame(self.dict)

    def print(self):
        print(f"Total time: {self.dataframe['tottime'].sum()}\n")
        print(self)

    def __str__(self):
        if self.sorting == 'none':
            df = self.dataframe
        else:
            df = self.dataframe.sort_values(self.sorting,
                                            ascending=self.ascending)
        if self.printn is not None:
            df = df[:self.printn]
        return str(df)

    def start(self, name):
        self.stats['start'].append(name, time.perf_counter())

    def end(self, name):
        self.stats['end'].append(name, time.perf_counter())

    def reset(self):
        self.stats = {'start': DictList(),
                      'end': DictList()}


class CoreProfiler(object):
    def __init__(self, name, stats):
        self.name = name
        self.stats = stats

    def __enter__(self):
        self.stats['start'].append(self.name, time.perf_counter())

    def __exit__(self, a, b, c):
        self.stats['end'].append(self.name, time.perf_counter())
