import multiprocessing as mp

class DictList(object):
    """A table-like object. It is a dictionary where each value is a
    list.
    
    Arguments
    ---------
    dic : {dict or None, None}
        A dictionary from which to start
    
    Attributes
    ----------
    dic : dict
        A dictionary where each entry is a list. This attribute is
        returned by the function `as_dict`.
    """
    def __init__(self, dic=None):
        self.dic = dic
    
    def __getitem__(self, key):
        return self.dic[key]
    
    @property
    def dic(self):
        return self._dic
    
    @dic.setter
    def dic(self, dic):
        self._dic = {}
        if dic is None:
            return
        elif isinstance(dic, dict):
            for key, val in dic.items():
                if isinstance(val, list):
                    self._dic[key] = val
                else:
                    self._dic[key] = [val]
        else:
            msg = 'The input has to be a dict.'
            raise TypeError(msg)
    
    def append(self, key, value=None):
        """Append data to the DictList. If some keys are not known to
        this DictList it will create a new entry.
        
        Arguments
        ---------
        key : hashable object or dict
            If this entry is a dictionary, all values of the keys will
            be appended to the DictList. If this entry is a hashable
            object, it will be understood as a key to the DictList and
            the optional argument 'value' will be appended to the
            DictList.
        value : {None or object, None}
            If 'key' is not a dictionary, this value will be appended to
            the DictList.
        
        Returns
        -------
        None
        """
        if isinstance(key, dict):
            for k, val in key.items():
                if k in self.dic:
                    self.dic[k].append(val)
                elif isinstance(val, list):
                    self.dic[k] = val
                else:
                    self.dic[k] = [val]
        elif key in self.dic:
            self.dic[key].append(value)
        elif isinstance(value, list):
            self.dic[key] = value
        else:
            self.dic[key] = [value]
    
    def as_dict(self):
        return self.dic
    
    def keys(self):
        return self.dic.keys()

class MPCounter(object):
    def __init__(self, val=0):
        assert isinstance(val, int), 'Initial value has to be an integer.'
        self.val = mp.Value('i', val)
    
    def increment(self, n=1):
        with self.val.get_lock():
            self.val.value += n
    
    def __add__(self, other):
        if not isinstance(other, (int, type(self))):
            msg = 'Can only add an integer or MPCounter object to an '
            msg += 'MPCounter object.'
            raise TypeError(msg)
        if isinstance(other, int):
            return MPCounter(val=self.value+other)
        return MPCounter(val=self.value+other.value)
    
    def __iadd__(self, other):
        if not isinstance(other, (int, type(self))):
            msg = 'Can only add an integer or MPCounter object to an '
            msg += 'MPCounter object.'
            raise TypeError(msg)
        if isinstance(other, int):
            self.increment(other)
        else:
            self.increment(other.value)
    
    def __eq__(self, other):
        if not isinstance(other, (int, type(self))):
            msg = 'Can only compare to int or MPCounter.'
            raise TypeError(msg)
        if isinstance(other, int):
            return self.value == other
        return self.value == other.value
    
    @property
    def value(self):
        return self.val.value
