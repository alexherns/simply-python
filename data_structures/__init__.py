from collections import defaultdict
from simply_python import sorting
import sys, math 

class LinkedList(object):
    """An instance maintains a representation of a LinkedList"""

    def __init__(self):
        self.head= None

    def __len__(self):
        """Returns length"""
        count= 0
        for _ in self:
            count+= 1
        return count

    def build_from_list(self, l):
        """Adds items from l into beginning of LL"""
        for item in l[::-1]:
            self.insert(LinkedListNode(item))

    def to_list(self):
        """Returns a list from items in linkedlist"""
        l= []
        for node in self:
            l.append(node.data)
        return l

    def insert(self, node):
        """Inserts node data at beginning of list"""
        node.child= self.head
        self.head= node

    def partition(self, x):
        """Partitions LL around pivot x"""
        parent= self.head
        explore_node= parent.child
        while explore_node:
            if explore_node < x:
                parent.child= explore_node.child
                explore_node.child= self.head
                self.head= explore_node
                explore_node= parent.child
            else:
                parent= explore_node
                explore_node= explore_node.child
    
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
        if node == None:
            return None
        self.head= node.child
        node.child= None
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
        """Return middle node.
        Time: O(n)
        Space: O(1)"""
        node= self.head
        middle= node
        while node.child:
            node= node.child
            middle= middle.child
            if node.child:
                node= node.child
        return middle

    def uniquify(self):
        """Deletes any duplicate items in the LL (in-place).
        Time: O(n^2)
        Space: O(1)"""
        node1= self.head
        while node1.child:
            node2= node1.child
            while node2:
                if node1 == node2:
                    self.delete(node2)
                    break
                node2= node2.child
            node1= node1.child

    def uniquify_faster(self):
        """Deletes any duplicate items in the LL (using external buffer).
        Time: O(n)
        Space: O(n)"""
        node= self.head
        seen= set([node.data])
        while node.child:
            child= node.child
            if child.data in seen:
                node.child= child.child
            else:
                seen.add(child.data)
            node= node.child

    def kth_from_end(self, k):
        """Returns kth item from end of list"""
        node1= self.head
        node2= self.head
        for _ in range(k):
            if not node2.child:
                return node1
            node2= node2.child
        while node2:
            node1= node1.child
            node2= node2.child
        return node1

    def is_palindrome(ll):
        """Checks if linked list is palindromic"""
        def palindrome_subfunc(node, k):
            """A recursive subfunction for is_palindrome"""
            if k <= 0:
                return node.child
            elif k== 1:
                if node == node.child:
                    return node.child.child
                return False
            else:
                child= palindrome_subfunc(node.child, k-2)
                if child and node == child:
                    return child.child
                return False
        node= ll.head
        k= len(ll)-1
        if k == 0: return True
        child= palindrome_subfunc(node.child, k-2)
        if node == child and isinstance(child, LinkedListNode):
            return True
        return False

    def __iter__(self):
        """Iterates through nodes in LinkedList"""
        node= self.head
        if node == None:
            return
        while node.child:
            yield node
            node= node.child
        yield node

    def __repr__(self):
        if self.head:
            node_str= '\n'.join([str(node) for node in self])
        else: node_str= 'None'
        return "<LinkedList: head={self.head}\n{node_str}>".format(self=self,
                node_str=node_str)



class LinkedListNode(object): 
    """An instance maintains references to child for a linked list""" 

    def __init__(self, data):
        self.data= data
        self.child= None
    
    def __lt__(self, other):
        return self.data < other

    def __le__(self, other):
        return self.data <= other

    def __ge__(self, other):
        return self.data >= other

    def __gt__(self, other):
        return self.data > other

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



class Queue(object):
    """Implementation of a FIFO queue using a linked list"""


    def __init__(self):
        self.head= None
        self.tail= None

    def enqueue(self, node):
        """Adds node to bottom of queue"""
        if not isinstance(node, LinkedListNode):
            node= LinkedListNode(node)
        if self.tail == None:
            self.tail= node
            self.head= node
        else:
            self.tail.child= node
            self.tail= node

    def peek(self):
        """Peeks at top of queue"""
        return self.head

    def dequeue(self):
        """Removes nodes from top of queue"""
        if not self.head:
            return None
        tmp= self.head
        if tmp == self.tail:
            self.head= None
            self.tail= None
        else:
            self.head= tmp.child
        return tmp.data



class Stack(LinkedList):
    """
    Implementation of a LIFO queue (stack), using a linked list
    """

    def pop(self):
        popped= LinkedList.pop(self)
        if popped != None: return popped

    def add(self, node):
        if not isinstance(node, LinkedListNode):
            node= LinkedListNode(node)
        LinkedList.insert(self, node)



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



class BTree(object):
    """Implementation of a binary tree using pointers"""

    def __init__(self, data):
        self.data= data
        self.left= None
        self.right= None
        self.parent= None

    def __iter__(self):
        """Iterate through subtrees in order"""
        if self.left:
            for node in self.left:
                yield node
        yield self
        if self.right:
            for node in self.right:
                yield node

    def convert_to_ll(self):
        """Returns the tree as a linked list (in order)"""
        ll= LinkedList()
        for subtree in self:
            node= LinkedListNode(subtree.data)
            ll.insert(node)
        ll.reverse()
        return ll

    def isBalanced(self):
        """Returns if tree is balanced (i.e. has no left or right subtrees with
        length difference more than 1)"""
        if self._isbalanced()[0] == False:
            return False
        else:
            return True

    def _isbalanced(self):
        """Worker function for the isBalanced method"""
        if self.left == None:
            lmin, lmax= 0, 0
        else:
            lmin, lmax= self.left._isbalanced()
            if lmin == False:
                return False, False
        if self.right == None:
            rmin, rmax= 0, 0
        else:
            rmin, rmax= self.right._isbalanced()
            if rmin == False:
                return False, False
        if abs(rmax-lmin)>1 or abs(lmax-rmin)>1:
            return False, False
        else:
            return 1+min(lmin,rmin), 1+max(lmax,rmax)

    def find_node(self, data):
        """Finds node containing "data" within Tree in O(n) time"""
        for tree in self:
            if tree == data:
                return tree

    def isBST1(self):
        """Determines if Binary Tree is a BST using min/max properties"""
        if self._isBST1()[0] == False:
            return False
        return True

    def _isBST1(self):
        """Worker for isBST1"""
        if self.left == None and self.right == None:
            return self, self
        if self.right != None:
            rmin, rmax= self.right._isBST1()
        else:
            rmin, rmax= self, self
        if self.left != None:
            lmin, lmax= self.left._isBST1()
        else:
            lmin, lmax= self, self
        if lmin == False or rmin == False:
            return False, False
        elif lmax > self or self >= rmin:
            return False, False
        else:
            return lmin, rmax

    def isBST2(self):
        """Determines if Binary Tree is a BST using in order traversal"""
        BTree.lastSeen= -sys.maxint
        try:
            return self._isBST2()
        finally:
            BTree.lastSeen= -sys.maxint

    def _isBST2(self):
        """Worker for isBST2"""
        if self.left and not self.left._isBST2():
            return False
        if self < BTree.lastSeen:
            return False
        BTree.lastSeen= self
        if self.right and not self.right._isBST2():
            return False
        return True

    def listByLevel(self):
        """Create a linked list of nodes at each level"""
        loadq= Queue()
        unloadq= Queue()
        lists= []
        loadq.enqueue(self)
        while loadq.peek() != None:
            lists.append(LinkedList())
            while loadq.peek() != None:
                unloadq.enqueue(loadq.dequeue())
            while unloadq.peek() != None:
                tree= unloadq.dequeue()
                if tree.left != None:
                    loadq.enqueue(tree.left)
                if tree.right != None:
                    loadq.enqueue(tree.right)
                lists[-1].insert(LinkedListNode(tree))
        return lists

    @staticmethod
    def mrca(tree1, tree2):
        """Returns the most recent common ancestor of two trees"""
        if tree1.find_node(tree2):
            return tree1
        if tree2.find_node(tree1):
            return tree2
        parent= tree1.parent
        child= tree1
        while parent != None:
            if parent.left == child and parent.right != None \
                    and parent.right.find_node(tree2) != None:
                return parent
            if parent.right == child and parent.left != None \
                    and parent.left.find_node(tree2) != None:
                return parent
            child= parent
            parent= child.parent
        return None

    def __cmp__(self, bst):
        return self.data.__cmp__(bst)

    def __lt__(self, bst):
        return self.data < bst

    def __le__(self, bst):
        return self.data <= bst

    def __eq__(self, bst):
        return self.data == bst

    def __ne__(self, bst):
        return self.data != bst

    def __ge__(self, bst):
        return self.data >= bst

    def __gt__(self, bst):
        return self.data > bst

    def __repr__(self):
        return "<BST: {0}>".format(self.data.__repr__())



class BST(BTree):
    """Implementation of a binary search tree using a BTree"""

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

    def find_minimum(self):
        """Get minimum node in tree"""
        if self.left:
            return self.left.find_minimum()
        return self

    def find_next(self):
        """Get next node in order from tree"""
        if self.right:
            return self.right.find_minimum()
        parent= self.parent
        while parent != None and parent < self:
            parent= parent.parent
        if parent > self:
            return parent
        else:
            return None

    def find_node(self, data):
        """Finds node containing "data" within BST in O(logn) time"""
        if self == data:
            return self
        elif self > data and self.left:
            return self.left.find_node(data)
        elif self < data and self.right:
            return self.right.find_node(data)
        else:
            return None

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

    @staticmethod
    def build_from_sorted(sorted_list, minloc=None, maxloc=None):
        """Returns a BST object from a sorted list"""
        assert sorting.is_sorted(sorted_list)
        if minloc == None: minloc= 0
        if maxloc == None: maxloc= len(sorted_list)-1
        if minloc > maxloc: return None
        elif minloc == maxloc: return BST(sorted_list[minloc])
        center= 2**int(math.log(maxloc-minloc+1,2))-1+minloc
        bst= BST(sorted_list[center])
        bst.left= BST.build_from_sorted(sorted_list, minloc, center-1)
        if bst.left != None:
            bst.left.parent= bst
        bst.right= BST.build_from_sorted(sorted_list, center+1, maxloc)
        if bst.right != None:
            bst.right.parent= bst
        return bst



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

    #    def to_sorted_list(self, comp_fun):
    #        """Returns binary tree as list sorted by heap structure. Currently
    #        broken"""
    #        if not self.is_heaped(comp_fun):
    #            self.heapify(comp_fun)
    #        output= type(self)(self.array[:])
    #        end= len(output)-1
    #        while end > 0:
    #            output[end], output[0]= output[0], output[end]
    #            end-= 1
    #            output.sift_down(comp_fun, source=0, end=end)
    #        return output.array

    def sift_down(self, comp_fun, source=0, end=None):
        """
        Moves value from source down to its proper place in the binary tree
        """
        if not end: end= len(self)
        if BinaryTree.l_child(source) >= end:
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
        self.sift_down(comp_fun, swap, end=end)

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
            self.sift_down(comp_fun, source=i, end=len(self))



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

    def sift_down(self, comp_fun, source=0, end=None):
        return super(MaxHeap, self).sift_down(MaxHeap.less_than, source=source,
                end=end)

    def sift_up(self, comp_fun=less_than, source=-1):
        return super(MaxHeap, self).sift_up(MaxHeap.less_than, source)

    def add(self, value, comp_fun=less_than):
        return super(MaxHeap, self).add(value, MaxHeap.less_than)

    def to_sorted_list(self):
        return super(MaxHeap, self).to_sorted_list(MaxHeap.less_than)


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

    def sift_down(self, comp_fun=greater_than, source=0, end=None):
        return super(MinHeap, self).sift_down(MinHeap.greater_than, source,
                end=end)

    def sift_up(self, comp_fun=greater_than, source=-1):
        return super(MinHeap, self).sift_up(MinHeap.greater_than, source)

    def add(self, value, comp_fun=greater_than):
        return super(MinHeap, self).add(value, MinHeap.greater_than)

    def to_sorted_list(self):
        return super(MinHeap, self).to_sorted_list(MinHeap.greater_than)



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



class LinearMap(object):
    """Implementations of a list-based hash-table -- inspired by Allen Downey's
    *Think Complexity*"""


    def __init__(self):
        self.map= []

    def add(self, key, value):
        """Add value indexed by key"""
        self.map.append((key, value))

    def get(self, key):
        """Get value indexed by key"""
        for iterkey, value in self.map:
            if iterkey == key:
                return value
        raise KeyError



class BetterMap(object):
    """Implementation of a "better" list-of-lists hash-table -- inspired by
    Allen Downey's *Think Complexity*"""


    def __init__(self, size=2):
        self.size= size
        self.maps= []
        for _ in range(size):
            self.maps.append(LinearMap())

    def add(self, key, value):
        """Adds value indexed by key"""
        m= self.maps[self.find_map(key)]
        m.add(key, value)

    def get(self, key):
        """Return value indexed by query"""
        m= self.maps[self.find_map(key)]
        return m.get(key)

    def find_map(self, key):
        """Returns index of map hashed by key"""
        return hash(key) % self.size



class HashTable(object):
    """Implementation of a true linear-time hash-table -- inspired by Allen
    Downey's *Think Complexity*"""


    def __init__(self):
        """Initialize an instance of a HashTable"""
        self.size= 0
        self.maps= BetterMap(size=2)

    def resize(self):
        """Expand hashmap size to twice current"""
        new_maps= BetterMap(size=self.size*2)
        for m in self.maps.maps:
            for key, value in m.map:
                new_maps.add(key, value)
        self.maps= new_maps

    def add(self, key, value):
        """Add value indexed by key"""
        self.maps.add(key, value)
        self.size+= 1
        if self.size == self.maps.size:
            self.resize()

    def get(self, key):
        """Return value indexed by key"""
        return self.maps.get(key)

    def __setitem__(self, key, value):
        self.add(key, value)

    def __getitem__(self, key):
        return self.get(key)



class StringBuffer(object):
    """Implementation of a Java-like StringBuffer"""


    def __init__(self, s='', data_structure='list'):
        if data_structure == 'list':
            self.buffer= list(s)
        elif data_structure == 'deque':
            from collections import deque
            self.buffer= deque(s)
        else: 
            raise ValueError, 'Data structure must be list or deque'

    def __len__(self):
        return len(self.buffer)

    def append(self, s):
        """Append string s"""
        assert isinstance(s, str) or isinstance(s, StringBuffer)
        self.buffer.extend(list(s))

    def __iadd__(self, s):
        assert isinstance(s, str) or isinstance(s, StringBuffer)
        self.append(s)

    def __getitem__(self, index):
        assert isinstance(index, int)
        return self.buffer[index]

    def __getslice__(self, start, end):
        assert isinstance(start, int)
        assert isinstance(end, int)
        return StringBuffer(self.buffer[start:end])

    def __setitem__(self, index, char):
        assert isinstance(index, int)
        assert isinstance(char, str)
        assert len(char) == 1
        self.buffer[index]= char

    def __setslice__(self, start, end, substr):
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert isinstance(substr, str)
        assert len(substr) == end-start
        self.buffer[start:end]= list(substr)

    def __add__(self, s):
        assert isinstance(s, StringBuffer) or isinstance(s, str)
        if type(s) == str:
            s= StringBuffer(s)        
        copy= self[:]
        copy.append(s)
        return copy

    def __iter__(self):
        return self.buffer.__iter__()

    def __str__(self):
        return ''.join(self.buffer)

    def __eq__(self, other):
        return self.buffer == other.buffer
