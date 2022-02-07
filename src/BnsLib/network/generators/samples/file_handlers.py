import h5py
import gc

from .file_generator import FileHandler
from ....types import LimitedSizeDict


class H5pyHandler(FileHandler):
    """A base-class for handling HDF5-files. It has to be sub-classed and
    should not be used directly.
    
    Indices passed to this handler are expected to be integers.
    
    When sub-classing this class two methods need to be overwritten:
        -__len__
        -_getitem_open
    Make sure to refer to help(H5pyHandler.__len__) and
    help(H5pyHandler._getitem_open) for information on how to implement
    them.
    
    In case you want to also use some form of re-scaling that can be
    accessed during iteration, also implement the rescale method.
    
    Arguments
    ---------
    file_path : str
        The path to the file that should be read from.
    base_index : {int or None, None}
        When multiple handlers are used it may be inconvenient to index
        each from 0 to the respective length. Instead it may be easier to
        index them continuously. The `base_index` tells the file handler
        which index it receives corresponds to 0, i.e. the index passed
        to the file will be index - base_index.
    """
    def __init__(self, file_path, base_index=None):
        super().__init__(file_path)
        if base_index is None:
            self.base_index = 0
        else:
            self.base_index = base_index
        self.file = None
    
    def correct_index(self, index):
        return index - self.base_index
    
    def __contains__(self, index):
        index = self.correct_index(index)
        if index < 0:
            return False
        return index < len(self)
    
    def __len__(self):
        """Return the number of samples contained in this Handler.
        
        For implementation it is recommended to use a structure similar
        to the one shown below. Make sure to check if the file is open
        already. If only a `with` statement is used, the file will be
        closed after the context finishes regardless of whether it was
        open before or not.
        
        Replace `...` in the code below with the appropriate code.
        
        ```
        if self.file is None:
            with h5py.File(self.file_path, 'r') as fp:
                length = ...
        else:
            length = len(self.file[...])
        return length
        ```
        """
        raise NotImplementedError
        
    def open(self, mode='r'):
        if self.file is None:
            self.file = h5py.File(self.file_path, mode)
        return self.file
    
    def close(self):
        self.file.close()
        self.file = None
    
    def __enter__(self, mode='r'):
        return self.open(mode=mode)
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
    
    def __getitem__(self, index):
        if index not in self:
            raise IndexError
        
        index = self.correct_index(index)
        
        if self.file is None:
            with self as fp:
                ret = self._getitem_open(index, fp)
        else:
            ret = self._getitem_open(index, self.file)
        return ret
    
    def _getitem_open(self, index, fp):
        """Returns a single sample from the index of the open file.
        
        Arguments
        ---------
        index : int
            The index for which to retrieve the sample.
        fp : open h5py.File object
            The open file-object from which to read.
        
        Returns
        -------
        sample:
            A single sample that can be processed downstream.
        """
        raise NotImplementedError
    
    def serialize(self):
        dic = {}
        dic['file_path'] = self.file_path
        dic['base_index'] = self.base_index
        return dic
    
    @classmethod
    def from_serialized(cls, dic):
        fpath = dic.pop('file_path', None)
        return cls(fpath, **dic)
    
    def rescale(self, target):
        raise NotImplementedError


class CachedH5pyHandler(H5pyHandler):
    """A base-class to handle HDF5-file accesses and caching the values.
    This class has to be sub-classed and the methods
        -__len__
        -_getitem_open
    have to be overwritten. See the documentation for more information.
    
    Arguments
    ---------
    file_path : str
        The path of the file from which data should be read.
    base_index : {int or None, None}
        When multiple handlers are used it may be inconvenient to index
        each from 0 to the respective length. Instead it may be easier to
        index them continuously. The `base_index` tells the file handler
        which index it receives corresponds to 0, i.e. the index passed
        to the file will be index - base_index.
    cachesize : {int or BnsLib.types.DataSize or None, None}
        The size of the cache. If the value is an integer it will be
        interpreted with the given `unit`. If None, no size limit is
        placed on the cache. See the documentation of
        BnsLib.types.LimitedSizedDict for details of the size-determination.
    unit : {str, `B`}
        The (binary) unit of the provided cachesize. See the documentation of
        BnsLib.types.DataSize for details.
    """
    def __init__(self, file_path, base_index=None, cachesize=None, unit='B'):
        super().__init__(file_path, base_index=base_index)
        self.cache = LimitedSizeDict(size_limit=cachesize, unit=unit)
    
    def __getitem__(self, index):
        if index not in self:
            raise IndexError
        
        index = self.correct_index(index)
        
        if index in self.cache:
            ret = self.cache[index]
        else:
            if self.file is None:
                with self as fp:
                    ret = self._getitem_open(index, fp)
            else:
                ret = self._getitem_open(index, self.file)
            self.cache[index] = ret
        
        return ret
    
    def rescale(self, target):
        raise NotImplementedError


class LoadDataHandler(FileHandler):
    """A base-class for handeling files when the entire file should
    be loaded into memory before accessing its contents.
    
    This class has to be sub-classed to be used. The method
        -load_data
        -__len__
    has to be overwritten. Furthermore, the __getitem__-method can be
    overwritten to gain more control over how the returned data is formatted.
    
    The data can be loaded by calling LoadDataHandler.load_data and be
    unloaded by calling LoadDataHandler.unload_data.
    
    Arguments
    ---------
    file_path : str
        The path of the file from which data should be read.
    base_index : {int or None, None}
        When multiple handlers are used it may be inconvenient to index
        each from 0 to the respective length. Instead it may be easier to
        index them continuously. The `base_index` tells the file handler
        which index it receives corresponds to 0, i.e. the index passed
        to the file will be index - base_index.
    load_on_init : {bool, False}
        Whether or not to call LoadDataHandler.load_data on initialization.
    """
    def __init__(self, file_path, base_index=None, load_on_init=False):
        super().__init__(file_path)
        self.base_index = base_index if base_index is not None else 0
        self.data = None
        if load_on_init:
            self.load_data()
    
    def correct_index(self, index):
        return index - self.base_index
    
    def __contains__(self, index):
        index = self.correct_index(index)
        if index < 0:
            return False
        return index < len(self)
    
    def __len__(self):
        raise NotImplementedError
    
    def load_data(self):
        """This function must be overwritten for this class to function.
        
        The data must be loaded into self.data.
        
        By default the __getitem__ method will return self.data[index],
        where the index is corrected by the `base_index`.
        """
        raise NotImplementedError
    
    def unload_data(self):
        if self.data is not None:
            del self.data
            self.data = None
            gc.collect()
    
    def __getitem__(self, index):
        if index not in self:
            raise IndexError
        
        index = self.correct_index(index)
        return self.data[index]
    
    def open(self, mode='r'):
        return NotImplemented
    
    def close(self):
        return NotImplemented
    
    def __enter__(self, mode='r'):
        return NotImplemented
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        return NotImplemented
    
    def serialize(self):
        raise TypeError("This class is not intended for use in multiprocessing.")
    
    @classmethod
    def from_serialized(self, dic):
        raise TypeError("This class is not intended for use in multiprocessing.")
    
    def rescale(self, target):
        raise NotImplementedError
