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

def input_to_list(inp, length=1):
    """Convert the input to a list of a given length.
    If the input is not a list, a list of the given length will be
    created. The contents of this list are all the same input value.
    
    Arguments
    ---------
    inp : list or other
        The input that should be turned into a list.
    length : {int, 1}
        The length of the output list.
    
    Returns
    -------
    list
        Either returns the input, when the input is a list of matching
        length or a list of the wanted length filled with the input.
    """
    if isinstance(inp, list):
        if len(inp) != length:
            msg = f'Length of list {len(inp)} does not match the length'
            msg += f' requirement {length}.'
            raise ValueError(msg)
        else:
            return inp
    else:
        return [inp] * length
