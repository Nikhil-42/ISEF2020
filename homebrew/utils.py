from numba import jitclass
from numba import deferred_type, optional
from numba import int64, float64
import numba

linked_node_type = deferred_type()
data_type = deferred_type()
linked_node_spec = [
    ('data', numba.typeof((1,1.0,1))),
    ('next', optional(linked_node_type))
]

@jitclass(linked_node_spec)
class LinkedNode(object):
    def __init__(self, data):
        self.data = data
        self.next = None
        
linked_node_type.define(LinkedNode.class_type.instance_type)

stack_spec = [
    ('head', optional(linked_node_type))
]

@jitclass(stack_spec)
class Stack():
    def __init__(self):    
        self.head = None

    def push(self, data):
        new = LinkedNode(data)
        new.next = self.head
        self.head = new

    def pop(self):
        old = self.head
        if old is None:
            raise ValueError("empty stack")
        else:
            self.head = old.next
            return old.data

data_type.define(numba.typeof((int64, float64, int64)))