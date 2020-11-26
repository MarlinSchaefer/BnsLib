import numpy as np

def list_length(inp):
    """Returns the length of a list or 1, if the input is not a list.
    
    Arguments
    ---------
    inp : list or other
        The input.
    
    Returns
    -------
    int
        The length of the input, if the input is a list. Otherwise
        returns 1.
    
    Notes
    -----
    -A usecase for this function is to homologize function inputs. If
     the function is meant to operate on lists but can also accept a
     single instance, this function will give the length of the list the
     function needs to create. (Useful in combination with the function
     input_to_list)
    """
    if isinstance(inp, list):
        return len(inp)
    else:
        return 1

def input_to_list(inp, length=None):
    """Convert the input to a list of a given length.
    If the input is not a list, a list of the given length will be
    created. The contents of this list are all the same input value.
    
    Arguments
    ---------
    inp : list or other
        The input that should be turned into a list.
    length : {int or None, None}
        The length of the output list. If set to None this function will
        call list_length to determine the length of the list.
    
    Returns
    -------
    list
        Either returns the input, when the input is a list of matching
        length or a list of the wanted length filled with the input.
    """
    if length is None:
        length = list_length(inp)
    if isinstance(inp, list):
        if len(inp) != length:
            msg = f'Length of list {len(inp)} does not match the length'
            msg += f' requirement {length}.'
            raise ValueError(msg)
        else:
            return inp
    else:
        return [inp] * length

def field_array_to_dict(inp):
    """Convert a pycbc.io.record.FieldArray to a standard Python
    dictionary.
    
    Arguments
    ---------
    inp : pycbc.io.record.FieldArray or numpy.recarry
        The array to convert.
    
    Returns
    -------
    dict:
        A dict where each value is a list containing the values of the
        numpy array.
    """
    return {name: list(inp[name]) for name in inp.dtype.names}

def dict_to_field_array(inp):
    """Convert a Python dictionary to a numpy FieldArray.
    
    Arguments
    ---------
    inp : dict
        A dictionary with structure `<name>: <list of values>`. All
        lists must be of equal length.
    
    Returns
    -------
    numpy field array
    """
    assert isinstance(inp, dict)
    keys = list(inp.keys())
    assert all([len(inp[key]) == len(inp[keys[0]]) for key in keys])
    out = []
    for i in range(len(inp[keys[0]])):
        out.append(tuple([inp[key][i] for key in keys]))
    dtypes = [(key, np.dtype(type(out[0][i]))) for (i, key) in enumerate(keys)]
    return np.array(out, dtype=dtypes)
