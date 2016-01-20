from collections import defaultdict
import Queue, sys 

class LinkedList(object):
    """An instance maintains a representation of a LinkedList"""

    def __init__(self):
        self.head= None

    def insert(self, node):
        """Inserts node data at beginning of list"""
        node.child= self.head
        self.head= node
        return
    
    def append(self, node, old_node):
        """Appends node data after old node"""
        node_before= self.head
        while node_before != old_node:
            node_before= node_before.child
        node.child= node_before.child
        node_before.child= node
        return

    def delete(self, node):
        """Deletes node"""
        node_before= self.head
        if node_before == node:
            self.head= self.head.child
            node.child= None
            return node
        while node_before:
            if node_before.child == node:
                node_before.child= node_before.child.child
                node.child= None
            node_before= node_before.child
        return node

    def pop(self):
        """Removes and returns head"""
        node= self.head
        node.child= None
        self.head= self.head.child
        return node

    def peek(self):
        """Returns head"""
        return self.head

    def search(self, node):
        """Return true if node in self, else False"""
        for other_node in iter_nodes:
            if node.data == other_node.data:
                return True
        return False

    def reverse(self):
        """Reverse the sequence of elements, in place in O(n)"""
        curr= self.head
        prev= None
        while curr.child:
            child= curr.child
            curr.child= prev
            prev= curr
            curr= child
        self.head= curr
        self.head.child= prev

    def find_middle(self):
        """Return middle node"""
        node= self.head
        middle= node
        while node.child:
            node= node.child
            middle= middle.child
            if node.child:
                node= node.child
        return middle

    def __iter__(self):
        """Iterates through nodes in LinkedList"""
        node= self.head
        while node.child:
            yield node
            node= node.child
        yield node

    def __repr__(self):
        if self.head:
            node_str= '\n'.join([str(node) for node in self.iter_nodes()])
        else: node_str= 'None'
        return "<LinkedList: head={self.head}\n{node_str}>".format(self=self,
                node_str=node_str)



class LinkedListNode(object): 
    """An instance maintains references to child for a linked list""" 

    def __init__(self, data):
        self.data= data
        self.child= None
    
    def __eq__(self, other):
        return self.data == other

    def __ne__(self, other):
        return self.data != other

    def __repr__(self):
        return "<Node: node.data={0}>".format(self.data)



class DoublyLinkedList(object):
    """An instance maintains a representation of a doubly linked list"""
    
    def __init__(self, node):
        self.first= node 
        self.last= node

    def __repr__(self):
        if self.first: 
            node_str= '\n'.join([str(node) for node in self.iter_nodes()])
        else: node_str= 'None'
        return "<DoublyLinkedList:\t{0}>".format(node_str)

    def prepend(self, node):
        """Prepends node to beginning"""
        if not self.first:
            self.__init__(node)
        else:
            node.child= self.first
            self.first.parent= node
            self.first= node

    def append(self, node):
        """Appends node to end"""
        if not self.first:
            self.__init__(node)
        else:
            node.parent= self.last
            self.last.child= node
            self.last= node

    def __iter__(self):
        """Iterates through nodes"""
        node= self.first
        while node.child:
            yield node
            node= node.child
        yield node
    
    def insert_after(self, node, old_node):
        """Appends node data after old node"""
        if old_node == self.last:
            self.append(node)
            return
        node_before= self.first
        while node_before != old_node:
            node_before= node_before.child
        node_after= node_before.child
        #New sets
        node.child= node_after
        node.parent= node_before
        node_before.child= node
        node_after.parent= node
        return

    def insert_before(self, node, old_node):
        """Inserts node before old node"""
        if old_node == self.first:
            self.prepend(node)
            return
        node_after= self.first
        while node_after != old_node:
            node_after= node_after.child
        node_before= node_after.parent
        #New sets
        node.child= node_after
        node.parent= node_before
        node_before.child= node
        node_after.parent= node
        return
        
    def delete(self, node):
        """
        Removes node from list
        """
        if node == self.first and node == self.last:
            self.first= None
            self.last= None
        elif node == self.first:
            self.first= self.first.child
            self.first.parent= None
        elif node == self.last:
            self.last= self.last.parent
            self.last.child= None
        else:
            find_node= self.first.child
            while find_node != node:
                find_node= find_node.child
            if find_node != node:
                return
            find_node.parent.child= find_node.child
            temp_p= find_node.parent
            find_node.parent= None
            find_node.child.parent= temp_p
            find_node.child= None



class DoublyLinkedListNode(object):
    """
    An instances maintains references to child and parent for a
    doubly-linked-list
    """
    def __init__(self, data):
        self.data= data
        self.child= None
        self.parent= None

    def __eq__(self, other):
        return self.data == other

    def __ne__(self, other):
        return self.data != other

    def __repr__(self):
        return "<Node: {self.data}>".format(self=self)



class FIFO_queue(DoublyLinkedList):
    """
    Implementation of a FIFO queue using a DoublyLinkedList
    """

    def pop(self):
        f= self.first
        self.delete(f)
        return f

    def add(self, node):
        DoublyLinkedList.append(self, node)

    def append(self, node):
        self.add(node)
        raise UserWarning('Use method add(self, node) for FIFO queues')

    def prepend(self, node):
        raise NotImplementedError('Append not implemented for FIFO queues')



class FIFO_dict(dict):
    """
    Implementation of a FIFO queue using a dict
    """

    def __init__(self):
        """
        Initializes a FIFO queue as a dict
        """
        super(self.__class__, self).__init__()
        self.first= None
        self.last= None

    def add(self, node):
        """
        Adds node to the FIFO queue
        """
        if not self.first:
            self.first= node
        else:
            self[self.last]['next']= node
        self.last= node
        self[node]= {}
        self[node]['next']= None

    def pop(self):
        """
        Pops the item from the front of the FIFO queue
        Returns None if queue is empty
        """
        if not self.first:
            return None
        v= self.first
        if self.first == self.last:
            self.first= None
            self.last= None
        else:
            self.first= self[self.first]['next']
        del self[v]
        return v

    def isempty(self):
        """
        Returns True if queue is empty, else returns False
        """
        if self.first == None:
            return True
        return False



class CircularBuffer(object):
    """
    Implementation of a Circular Buffer
    """

    def __init__(self, size=10):
        """
        Initializes a Circular Buffer of size=size
        """
        self.buffer= [[] for _ in range(size)]
        self.first= None
        self.last= None
        self.size= size

    def add(self, item):
        """
        Adds item to Circular Buffer
        """
        if isinstance(self.last, int):
            self.last= (self.last+1)%self.size
        else:
            self.last= 0
            self.first= 0
            self.buffer[self.last]= item
            return
        self.buffer[self.last]= item
        if self.first == self.last:
            self.first= (self.first+1)%self.size

    def pop(self):
        """
        Pops first item added to the Cicular Buffer
        """
        if not isinstance(self.last, int): return
        item= self.buffer[self.first]
        self.buffer[self.first]= []
        if self.last == self.first:
            self.first= None
            self.last= None
            return item
        else:
            self.first= (self.first+1)%self.size
            return item

    def __repr__(self):
        return '<CircularBuffer: size:{s.size} buffer:{s.buffer}>'.format(\
                s=self)



class FIFO_circ(CircularBuffer):
    """
    Implementation of a FIFO queue as a Circular Buffer
    """
    pass



class BST(object):
    """Implementation of a binary search tree using pointers"""

    def __init__(self, data):
        self.data= data
        self.left= None
        self.right= None
        self.parent= None

    def insert(self, bst):
        """Insert subtree bst at appropriate location"""
        if self > bst:
            if self.left:
                self.left.insert(bst)
            else:
                self.left= bst
                bst.parent= self
        else:
            if self.right:
                self.right.insert(bst)
            else:
                self.right= bst
                bst.parent= self
            
    def __iter__(self):
        """Iterate through subtrees in increasing order"""
        if self.left:
            for node in self.left:
                yield node
        yield self
        if self.right:
            for node in self.right:
                yield node

    def convert_to_ll(self):
        """Returns the BST as a linked list"""
        ll= LinkedList()
        for subtree in self:
            node= LinkedListNode(subtree.data)
            ll.insert(node)
        ll.reverse()
        return ll


    def find_minimum(self):
        """Get minimum node in tree"""
        if self.left:
            return self.left.find_minimum()
        return self

    def fix_parent(self, bst):
        """Deletes self, inserting bst in its place"""
        if self.parent:
            if self.parent > self:
                self.parent.left= bst
            else:
                self.parent.right= bst
        if bst:
            bst.parent= self.parent
        self.parent= None

    def delete(self, bst):
        """Deletes subtree bst"""
        if self < bst:
            self.right.delete(bst)
        elif self > bst:
            self.left.delete(bst)
        else:
            if self.left and self.right:
                min_right_side= self.right.find_minimum()
                min_right_side.fix_parent(None)
                min_right_side.left= self.left
                min_right_side.left.parent= min_right_side
                min_right_side.right= self.right
                min_right_side.right.parent= min_right_side
                self.fix_parent(min_right_side)
            elif self.left:
                self.fix_parent(self.left)
            elif self.right:
                self.fix_parent(self.right)
            else:
                self.fix_parent(None)

    def __cmp__(self, bst):
        return self.data.__cmp__(bst.data)

    def __lt__(self, bst):
        return self.data < bst.data

    def __le__(self, bst):
        return self.data <= bst.data

    def __eq__(self, bst):
        return self.data == bst.data

    def __ne__(self, bst):
        return self.data != bst.data

    def __ge__(self, bst):
        return self.data >= bst.data

    def __gt__(self, bst):
        return self.data > bst.data

    def __repr__(self):
        return "<BST: {0}>".format(self.data.__repr__())



class BSTPriorityQueue(BST):
    """Implementation of a priority queue as a binary search tree"""

    def __init__(self, priority=0, contents=None):
        BST.__init__(self, data=priority)
        self.contents= contents

    def add_tuple(self, pq_tuple):
        """Adds contents in tuple: [priority, contents] to PQ"""
        self.add(priority=pq_tuple[0], contents=pq_tuple[1])

    def add(self, priority, contents):
        if self.contents:
            node= BSTPriorityQueue(priority, contents)
            node.contents= contents
            self.insert(node)
        else:
            self.data= priority
            self.contents= contents

    def pop(self):
        node= self.find_minimum()
        self.delete(node)
        return node.contents


    
class PriorityQueue(defaultdict):
    """
    Implementation of a priority queue as a defaultdict.

    This is a wrapper class which much have it's pop command rewritten in
    MinPriorityQueue and MaxPriorityQueue.
    """

    def __init__(self):
        """
        Initializes self as a list defaultdict
        """
        super(PriorityQueue, self).__init__(list)

    def add(self, value, priority):
        """
        Adds a value to the queue with priority
        """
        self[priority].append(value)

    def pop(self):
        """
        Must be re-written in subclasses
        """
        pass

    def isempty(self):
        """
        Returns True if queue is empty, else False
        """
        try:
            min_pri= min([key for key in self.keys() if self[key]])
            return False
        except ValueError:
            return True

    def remove(self, value, priority):
        """
        Removes the given value with set priority in the queue
        """
        self[priority].remove(value)

    def reset(self, value, pold, pnew):
        """
        Resets the priority of value (pold), with pnew
        """
        self.remove(value, pold)
        self.add(value, pnew)



class MinPriorityQueue(PriorityQueue):
    """
    Implementation of a minimum priority queue from super PriorityQueue
    """

    def pop(self):
        """ 
        Returns value from the queue with the minimum priority
        """
        try:
            min_pri= min([key for key in self.keys() if self[key]])
        except ValueError:
            return None
        return self[min_pri].pop()



class MaxPriorityQueue(PriorityQueue):
    """
    Implementation of a maximum priority queue from super PriorityQueue
    """
    
    def pop(self):
        """ 
        Returns value from the queue with the minimum priority
        """
        try:
            max_pri= max([key for key in self.keys() if self[key]])
        except ValueError:
            return None
        return self[max_pri].pop()



class BinaryTree(object):
    """
    Implementation of a Binary Tree using an array
    """

    def __init__(self, l= []):
        """
        Initializes an instance of a Binary Tree using input array l. Doesn't
        heapify result
        """
        self.array= l

    @staticmethod
    def dist_to_root(index):
        """
        Returns distance from index to root
        """
        if index == 0: return 0
        return 1 + BinaryTree.dist_to_root(BinaryTree.parent(index))

    def __len__(self):
        """
        Returns length of binary array
        """
        return len(self.array)

    def __getitem__(self, index):
        """
        Returns value at index in binary array
        """
        return self.array[index]
     
    def __setitem__(self, key, value):
        """
        Sets the array key to value
        """
        self.array[key]= value

    def peek(self):
        """
        Returns value at index 0 in array
        """
        return self[0]

    def add(self, value, comp_fun):
        """
        Appends value to Binary Tree, then sifts up into appropriate location
        """
        self.array.append(value)
        self.sift_up(comp_fun)

    @staticmethod
    def l_child(index):
        """
        Returns location of index's left child)
        """
        return 2*index + 1
    
    @staticmethod
    def r_child(index):
        """
        Returns location of index's right child
        """
        return 2*index + 2

    @staticmethod
    def children(index):
        """
        Returns tuple containing location of index's left and right children
        """
        return BinaryTree.l_child(index), BinaryTree.r_child(index)

    @staticmethod
    def parent(index):
        """
        Returns location of index's parent
        """
        return (index-1)/2
    
    def is_heaped(self, comp_fun):
        """
        Returns True if array is properly heaped as a Binary Array else False
        Must be given a comparison function (generally a lambda function for
        greater than or less than)
        """
        for i in range(len(self)/2):
            if comp_fun(self[i], min(self[BinaryTree.l_child(i)],
                    (self[BinaryTree.r_child(i)] if BinaryTree.r_child(i) <
                        len(self) else self[BinaryTree.l_child(i)]))):
                return False
        return True

    def sift_down(self, comp_fun, source=0):
        """
        Moves value from source down to its proper place in the binary tree
        """
        if BinaryTree.l_child(source) >= len(self):
            return
        swap= source
        l_child, r_child= BinaryTree.children(source)
        if comp_fun(self[swap], self[l_child]):
            swap= l_child
        if r_child < len(self) and comp_fun(self[swap], self[r_child]):
            swap= r_child
        if swap == source:
            return
        self[source], self[swap]= self[swap], self[source]
        self.sift_down(comp_fun, swap)

    def sift_up(self, comp_fun, source= -1):
        """
        Moves value from source up to its proper place in the binary tree
        """
        if source == 0:
            return
        elif source == -1:
            source= len(self) - 1
        parent= BinaryTree.parent(source)
        if comp_fun(self[parent], self[source]):
            self[parent], self[source]= self[source], self[parent]
            self.sift_up(comp_fun, parent)

    def heapify(self, comp_fun):
        """
        Heapify self.array according to comp_fun
        """
        for i in range(BinaryTree.parent(len(self))+1)[::-1]:
            self.sift_down(comp_fun, source=i)



class MaxHeap(BinaryTree):
    """
    Implementation of a Max Heap using the BinaryTree specification
    """
    
    less_than= staticmethod(lambda x, y: x < y)

    def __init__(self, l=[]):
        """
        Initializes an instance of a MaxHeap using input array l. Also
        heapifies the array by default
        """
        self.array= l
        self.heapify(MaxHeap.less_than)

    def is_heaped(self, comp_fun=less_than):
        return super(MaxHeap, self).is_heaped(MaxHeap.less_than)

    def sift_down(self, comp_fun=less_than, source=0):
        return super(MaxHeap, self).sift_down(MaxHeap.less_than, source)

    def sift_up(self, comp_fun=less_than, source=-1):
        return super(MaxHeap, self).sift_up(MaxHeap.less_than, source)

    def add(self, value, comp_fun=less_than):
        return super(MaxHeap, self).add(value, MaxHeap.less_than)



class MinHeap(BinaryTree):
    """
    Implementation of a Min Heap using the BinaryTree specification
    """

    greater_than= staticmethod(lambda x, y: x > y)

    def __init__(self, l=[]):
        """
        Initializes an instance of a MinHeap using input array l. Also
        heapifies the array by default
        """
        self.array= l
        self.heapify(MinHeap.greater_than)

    def is_heaped(self, comp_fun=greater_than):
        return super(MinHeap, self).is_heaped(MinHeap.greater_than)

    def sift_down(self, comp_fun=greater_than, source=0):
        return super(MinHeap, self).sift_down(MinHeap.greater_than, source)

    def sift_up(self, comp_fun=greater_than, source=-1):
        return super(MinHeap, self).sift_up(MinHeap.greater_than, source)

    def add(self, value, comp_fun=greater_than):
        return super(MinHeap, self).add(value, MinHeap.greater_than)



class UnionFindArray(dict):
    """Implementation of a Union-Find data structure with the array
    specification"""

    def __init__(self, S):
        """Initializes all items in S to their own set"""
        self.set= {}
        for s in S:
            self[s]= s
            self.set[s]= [s]

    def find(self, v):
        """Retrieves set of element v"""
        return self[v]

    def union(self, a, b):
        """Joins sets a and b to larger of the two"""
        if len(self.set[a]) > len(self.set[b]):
            self.set[a].extend(self.set[b])
            for key in self.set[b]:
                self[key]= a
            del self.set[b]
        else:
            self.set[b].extend(self.set[a])
            for key in self.set[a]:
                self[key]= b
            del self.set[a]



class UnionFindPointer(object):
    """Implementation of a Union-Find data structure with the use of Pointers
    to speed up the union operation"""

    def __init__(self, S):
        """Initializes all items in S to their own set: O(n)"""
        self.parent= {s:s for s in S}
        self.size= {s:1 for s in S}

    def union(self, a, b):
        """Joins sets a and b to larger of the two: O(1)"""
        if self.size[a] > self.size[b]:
            self.parent[b]= a
            self.size[a]+= self.size[b]
        else:
            self.parent[a]= b
            self.size[b]+= self.size[a]

    def find(self, v):
        """Retrieves set of element v: O(logn)"""
        child= v
        parent= self.parent[child]
        while child != parent:
            child= parent
            parent= self.parent[parent]
        return parent



