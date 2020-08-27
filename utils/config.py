import configparser
import numpy as np

special_values={
    'pi': np.pi,
    'e': np.e,
    'null': None,
    'none': None
    }

known_functions={
    'sin': (np.sin, 1),
    'cos': (np.cos, 1),
    'tan': (np.tan, 1),
    'exp': (np.exp, 1),
    '+': (lambda x, y: x+y, 2),
    '-': (lambda x, y: x-y, 2),
    '*': (lambda x, y: x*y, 2),
    '/': (lambda x, y: x/y, 2)
    }

class BinaryTree(object):
    """A class that implements a basic binary tree that can be sorted by
    a few functions.
    
    Arguments
    ---------
    content : object
        The content this node of a tree should hold.
    parent : {BinaryTree or None, None}
        The BinaryTree-node of which this node is a child.
    
    Attributes
    ----------
    content : object
        The content held by this node.
    parent : BinaryTree or None
        The parent BinaryTree of which this node is a child.
    left : BinaryTree or None
        The left child of this node.
    right : BinaryTree or None
        The right child of this node.
    root : bool
        Whether or not this node is the root of the tree.
    leaf : bool
        True if this node has no children.
    """
    def __init__(self, content, parent=None):
        self.content = content
        self.parent = parent
        self.left = None
        self.right = None
    
    def copy(self):
        """Returns a new instance of itself.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        copy : BinaryTree
            A new instance of the tree from which it was called.
        """
        ret = BinaryTree(self.content, parent=self.parent)
        ret.set_left_child(self.left)
        ret.set_right_child(self.right)
        return ret
    
    def add_left_leaf(self, content):
        """Set the left child of this node to be a BinaryTree without
        any children.
        
        Arguments
        ---------
        content : object
            The content of the child node.
        
        Returns
        -------
        None
        """
        self.left = BinaryTree(content, parent=self)
    
    def add_right_leaf(self, content):
        """Set the right child of this node to be a BinaryTree without
        any children.
        
        Arguments
        ---------
        content : object
            The content of the child node.
        
        Returns
        -------
        None
        """
        self.right = BinaryTree(content, parent=self)
    
    def set_left_child(self, left):
        """Set the left child of this node.
        
        Arguments
        ---------
        left : BinaryTree
            The tree to be the left child of this node.
        
        Returns
        -------
        None
        """
        self.left = left
    
    def set_right_child(self, right):
        """Set the right child of this node.
        
        Arguments
        ---------
        left : BinaryTree
            The tree to be the right child of this node.
        
        Returns
        -------
        None
        """
        self.right = right
    
    @property
    def root(self):
        """Return True if this node has no parent node and thus is a
        root of a tree.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        bool:
            True if this node has no parent node and thus is a root of a
            tree.
        """
        return self.parent is None
    
    @property
    def leaf(self):
        return self.left is None and self.right is None
    
    def search_left_first(self, self_as_root=False):
        if self_as_root:
            search_tree = self.copy()
            search_tree.parent = None
        ret = [self.content]
        if self.left is not None:
            ret.extend(self.left.search_left_first())
        if self.right is not None:
            ret.extend(self.right.search_left_first())
        return ret
    
    def left_right_mid(self, self_as_root=False):
        if self_as_root:
            search_tree = self.copy()
            search_tree.parent = None
        else:
            search_tree = self
        
        ret = []
        if search_tree.left is not None:
            ret.extend(search_tree.left.left_right_mid())
        if search_tree.right is not None:
            ret.extend(search_tree.right.left_right_mid())
        ret.append(search_tree.content)
        return ret
    
    def right_left_mid(self, self_as_root=False):
        if self_as_root:
            search_tree = self.copy()
            search_tree.parent = None
        else:
            search_tree = self
        
        ret = []
        if search_tree.right is not None:
            ret.extend(search_tree.right.right_left_mid())
        if search_tree.left is not None:
            ret.extend(search_tree.left.right_left_mid())
        ret.append(search_tree.content)
        return ret

class ExpressionString(object):
    def __init__(self, string=None):
        self.set_string(string)
        self.known_functions = known_functions
        self.special_values = special_values
    
    def set_string(self, inp):
        if inp == None:
            inp = ''
        else:
            inp = str(inp)
        self.orig_string = inp
        self.string = inp
        self.levels = np.zeros(len(inp))
    
    def __len__(self):
        return len(self.string)
    
    @property
    def min(self):
        return self.level.min()
    
    @property
    def max(self):
        return self.level.max()
    
    def add_named_value(self, name, value):
        self.special_values[name] = float(value)
    
    def add_named_values(self, value_dict):
        for name, value in value_dict.items():
            self.add_named_value(name, value)
    
    def add_function(self, name, function, num_arguments=1):
        self.known_functions[name] = (function, num_arguments)
    
    def add_functions(self, function_dict):
        for name, function_part in function_dict.items():
            try:
                func, num = function_part
            except TypeError:
                func = function_part
                num = 1
            self.add_function(name, func, num_arguments=num)
    
    def parse_brackets(self):
        string = []
        level = []
        current_level = 0
        for i in range(len(self.orig_string)):
            if self.orig_string[i] in ['(', '[']:
                current_level += 1
            elif self.orig_string[i] in [')', ']']:
                current_level-= 1
            else:
                level.append(current_level)
                string.append(self.orig_string[i])
        self.string = ''.join(string)
        self.level = np.array(level, dtype=int)
    
    def parse_summation(self):
        insert_zero = []
        for i_sub in range(len(self)):
            i = len(self)-1-i_sub
            if self.string[i] in ['+', '-']:
                left = i
                run = True
                while run:
                    if left <= 0:
                        run = False
                    elif self.level[left] < self.level[i]:
                        left += 1
                        run = False
                    else:
                        left -= 1
                right = i
                run = True
                while run:
                    if right >= len(self):
                        run = False
                    elif self.level[right] < self.level[i]:
                        run = False
                    else:
                        right += 1
                if self.string[i] == '-' and left == i:
                    insert_zero.append([i, self.level[i]+1])
                self.level[left:i] += 1
                self.level[i+1:right] += 1
        
        tmp_string = list(self.string)
        tmp_level = list(self.level)
        while len(insert_zero) > 0:
            curr_idx, curr_level = insert_zero.pop(0)
            for i in range(len(insert_zero)):
                insert_zero[i][0] += 1
            tmp_string.insert(curr_idx, '0')
            curr_level = self.level[curr_idx] + 1
            tmp_level.insert(curr_idx, curr_level)
        self.string = ''.join(tmp_string)
        self.level = np.array(tmp_level, dtype=int)
    
    def parse_product(self):
        for i in range(len(self)):
            if self.string[i] in ['*', '/']:
                left = i
                run = True
                while run:
                    if left <= 0:
                        run = False
                    elif self.level[left] < self.level[i]:
                        left += 1
                        run = False
                    else:
                        left -= 1
                right = i
                run = True
                while run:
                    if right >= len(self):
                        run = False
                    elif self.level[right] < self.level[i]:
                        run = False
                    else:
                        right += 1
                self.level[left:i] += 1
                self.level[i+1:right] += 1
    
    def parse_atomic(self, atomic):
        if atomic in self.known_functions:
            return self.known_functions[atomic]
        elif atomic in self.special_values:
            return self.special_values[atomic]
        else:
            #At this point we assume it is a number
            if '.' in atomic:
                return float(atomic)
            else:
                return int(atomic)
    
    def parse_to_atomic(self):
        atomics = [self.string[0]]
        atomic_levels = [self.level[0]]
        for char, level in zip(self.string[1:], self.level[1:]):
            if atomic_levels[-1] == level:
                atomics[-1] += char
            else:
                atomics.append(char)
                atomic_levels.append(level)
        atomics = [self.parse_atomic(atomic) for atomic in atomics]
        return atomics, atomic_levels
    
    def get_child_indices(self, atomics, atomic_levels, idx):
        left = (None, self.max+1)
        left_part = atomic_levels[:idx]
        if len(left_part) > 0:
            for i in range(1, len(left_part)+1):
                level = atomic_levels[idx - i]
                if level <= atomic_levels[idx]:
                    break
                elif level < left[1]:
                    left = (idx-i, level)
        
        right = (None, self.max+1)
        right_part = atomic_levels[idx+1:]
        if len(right_part) > 0:
            for i in range(1, len(right_part)+1):
                level = atomic_levels[idx+i]
                if level <= atomic_levels[idx]:
                    break
                elif level < right[1]:
                    right = (idx+i, level)
        
        return left[0], right[0]
    
    def tree_from_index(self, atomics, atomic_levels, idx, parent=None):
        if idx is None:
            return None
        ret = BinaryTree(atomics[idx], parent=parent)
        left_idx, right_idx = self.get_child_indices(atomics, atomic_levels, idx)
        ret.set_left_child(self.tree_from_index(atomics, atomic_levels, left_idx, parent=ret))
        ret.set_right_child(self.tree_from_index(atomics, atomic_levels, right_idx, parent=ret))
        return ret
    
    def atomics_to_tree(self, atomics, atomic_levels):
        if len(list(filter(lambda level: level == 0, atomic_levels))) > 1:
            raise RuntimeError('Binary tree needs to have a unique root.')
        
        root = np.where(np.array(atomic_levels, dtype=int) == 0)[0][0]
        tree = self.tree_from_index(atomics, atomic_levels, root)
        
        return tree
    
    def carry_out_operations(self):
        atomics, atomic_levels = self.parse_to_atomic()
        tree = self.atomics_to_tree(atomics, atomic_levels)
        order_operations = tree.right_left_mid()
        
        stack = []
        for i, operation in enumerate(order_operations):
            if isinstance(operation, tuple):
                args = []
                for i in range(operation[1]):
                    args.append(stack.pop())
                stack.append(operation[0](*args))
            else:
                stack.append(operation)
        
        return stack.pop()
    
    def parse(self):
        if self.orig_string == '':
            return None
        self.parse_brackets()
        self.parse_summation()
        self.parse_product()
        return self.carry_out_operations()
    
    def parse_string(self, string):
        backup_orig_string = self.orig_string
        backup_string = self.string
        backup_level = self.level
        self.set_string(string)
        res = self.parse()
        self.orig_string = backup_orig_string
        self.string = backup_string
        self.level = backup_level
        return res
    
    def print(self):
        print_string = ''
        for row in range(self.max+1):
            print_string += str(row) + ' '
            for col in range(len(self)):
                if row == self.level[col]:
                    print_string += self.string[col]
                else:
                    print_string += ' '
            print_string += '\n'
        print(print_string)

#def get_config_value(inp):
    #"""Convert the string returned by ConfigParser to an Python
    #expression.
    
    #This function uses some special formatting rules. The string may
    #contain some special known value like Pi or None. These will be
    #converted accordingly. Furthermore, certain operations (+, -, ...)
    #and named functions (sin, cos, ...) are supported.
    
    #Arguments
    #---------
    #inp : str
        #The string returned by ConfigParser.get.
    
    #Returns
    #-------
    #expression:
        #Tries to interpret the string as a Python expression. This is
        #done by parsing the string.
    #"""
    ##Test for string
    #if inp[0] in ["'", '"'] and inp[-1] in ["'", '"']:
        #return inp[1:-1]

#def get_config_type(inp):
    #"""Interpret the string as returned by the ConfigParser and try to
    #guess the data-type.
    
    #Arguments
    #---------
    #inp : str
        #The string as it is returned by ConfigParser.get.
    
    #Returns
    #-------
    #type:
        #The data-type as inferred from the string.
    #"""
    ##Test for None
    #if inp == '' or inp.lower() == 'null' or inp.lower() == 'none':
        #return type(None)
    
    ##Test for obvious string
    #if inp[0] in ['"', "'"] and inp[-1] in ['"', "'"]:
        #return str
    
    ##Test for number
    #tmp = inp.replace('.', '') if ('.' in inp and inp.count('.') == 1) else inp.copy()
    
    #if tmp.isdecimal():
        #if '.' in inp:
            #return float
        #else:
            #return int
        

#def config_to_dict(file_paths):
    #ret = {}
    #return ret
