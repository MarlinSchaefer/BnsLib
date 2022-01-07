from .file_generator import FileGenerator
from ....types import DictList
import numpy as np


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
        self.apply_shuffle()
        self.set_indices()
    
    def on_epoch_end(self):
        # The __init__ method of the FileGenerator calls this method
        # before construction is complete and self.index_groups is
        # available.
        if hasattr(self, 'index_groups'):
            self.apply_shuffle()
            self.set_indices()
    
    def apply_shuffle(self):
        if self.shuffle:
            for group in self.index_groups:
                np.random.shuffle(group)
            if self.shuffle_groups:
                np.random.shuffle(self.index_groups)
    
    def set_indices(self):
        self.indices = np.concatenate(self.index_groups)
