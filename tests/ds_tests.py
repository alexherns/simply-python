import unittest
import random
from simply_python import data_structures
from simply_python import sorting

class LinkedListNodeTests(unittest.TestCase):

    def test_init(self):
        node= data_structures.LinkedListNode(1)
        self.assertEqual(1, node.data)
        self.assertIsNone(node.child)

    def test_equality(self):
        node1= data_structures.LinkedListNode(1)
        node2= data_structures.LinkedListNode(1)
        self.assertEqual(node1, node2)
        node2= data_structures.LinkedListNode(2)
        self.assertNotEqual(node1, node2)

class LinkedListTests(unittest.TestCase):
    
    def test_init(self):
        ll= data_structures.LinkedList()
        for i in range(10):
            i= data_structures.LinkedListNode(i)
            ll.insert(i)
        self.assertEqual(ll.head, i)

    def test_append(self):
        ll= data_structures.LinkedList()
        node1= data_structures.LinkedListNode(1)
        node2= data_structures.LinkedListNode(2)
        ll.insert(node1)
        ll.append(node2, node1)
        self.assertEqual(ll.head, node1)
        self.assertEqual(ll.head.child, node2)
        self.assertIsNone(ll.head.child.child)

    def test_reverse(self):
        ll= data_structures.LinkedList()
        nodes= [data_structures.LinkedListNode(i) for i in range(10)]
        for node in nodes:
            ll.insert(node)
        ll.reverse()
        self.assertTrue(sorting.is_sorted([node.data for node in ll]))

    def test_delete(self):
        ll= data_structures.LinkedList()
        node1= data_structures.LinkedListNode(1)
        node2= data_structures.LinkedListNode(2)
        node3= data_structures.LinkedListNode(3)
        ll.insert(node1)
        ll.append(node2, node1)
        ll.append(node3, node2)
        ll.delete(node1) #test delete from beginning
        self.assertEqual(ll.head, node2)
        ll.insert(node1)
        ll.delete(node2) #test delete in middle
        self.assertEqual(ll.head.child, node3)
        ll.append(node2, node1)
        ll.delete(node3) #test delete from end
        self.assertIsNone(ll.head.child.child)

    def test_middle(self):
        ll= data_structures.LinkedList()
        nodes= [data_structures.LinkedListNode(i) for i in range(10)]
        for node in nodes:
            ll.insert(node)
        ll.reverse()
        self.assertEqual(nodes[5], ll.find_middle())

class DoublyLinkedListNodeTests(unittest.TestCase):

    def test_init(self):
        node= data_structures.DoublyLinkedListNode(1)
        self.assertEqual(1, node.data)
        self.assertIsNone(node.child)

    def test_equality(self):
        node1= data_structures.DoublyLinkedListNode(1)
        node2= data_structures.DoublyLinkedListNode(1)
        self.assertEqual(node1, node2)
        node2= data_structures.DoublyLinkedListNode(2)
        self.assertNotEqual(node1, node2)

class DoublyLinkedListTests(unittest.TestCase):

    dbl_node= data_structures.DoublyLinkedListNode
    dll= data_structures.DoublyLinkedList

    def test_init(self):
        node1= self.dbl_node(1) 
        dll= self.dll(node1)
        self.assertEqual(dll.first, node1)
        self.assertEqual(dll.last, node1)
        self.assertIsNone(dll.first.parent)
        self.assertIsNone(dll.first.child)
        
    def test_prepend(self):
        node1= self.dbl_node(1)
        node2= self.dbl_node(2)
        dll= self.dll(node1)
        dll.prepend(node2)
        self.assertEqual(dll.first, node2)
        self.assertEqual(dll.last, node1)
        self.assertEqual(dll.first.child, node1)
        self.assertEqual(dll.last.parent, node2)

    def test_append(self):
        node1= self.dbl_node(1)
        node2= self.dbl_node(2)
        dll= self.dll(node1)
        dll.append(node2)
        self.assertEqual(dll.first, node1)
        self.assertEqual(dll.last, node2)
        self.assertEqual(dll.first.child, node2)
        self.assertEqual(dll.last.parent, node1)

    def test_insert_after(self):
        node1= self.dbl_node(1)
        node2= self.dbl_node(2)
        node3= self.dbl_node(3)
        dll= self.dll(node1)
        dll.insert_after(node2, node1)
        self.assertEqual(dll.last, node2)
        dll.insert_after(node3, node1)
        self.assertEqual(dll.last, node2)
        self.assertEqual(dll.first.child, node3)
        self.assertEqual(dll.last.parent, node3)
        self.assertEqual(dll.first.child.parent, node1)
        self.assertEqual(dll.last.parent.child, node2)

    def test_insert_before(self):
        node1= self.dbl_node(1)
        node2= self.dbl_node(2)
        node3= self.dbl_node(3)
        dll= self.dll(node2)
        dll.insert_before(node1, node2)
        self.assertEqual(dll.last, node2)
        dll.insert_before(node3, node2)
        self.assertEqual(dll.last, node2)
        self.assertEqual(dll.first.child, node3)
        self.assertEqual(dll.last.parent, node3)
        self.assertEqual(dll.first.child.parent, node1)
        self.assertEqual(dll.last.parent.child, node2)

    def test_delete(self):
        node1= self.dbl_node(1)
        node2= self.dbl_node(2)
        node3= self.dbl_node(3)
        dll= self.dll(node2)
        dll.delete(node2)
        self.assertEqual(dll.first, dll.last)
        self.assertEqual(dll.first, None)
        dll.append(node1)
        dll.insert_after(node2, node1)
        dll.insert_after(node3, node2)
        dll.delete(node3)
        self.assertEqual(dll.last, node2)
        dll.insert_after(node3, node2)
        dll.delete(node1)
        self.assertEqual(dll.first, node2)
        dll.insert_before(node1, node2)
        dll.delete(node2)
        self.assertEqual(dll.last.parent, node1)
        self.assertEqual(dll.first.child, node3)

class FIFO_QueueTests(unittest.TestCase):

    def test_pop(self):
        nodes= [data_structures.DoublyLinkedListNode(i) for i in range(1,11)]
        fifo= data_structures.FIFO_queue(nodes[0])
        for node in nodes[1:]:
            fifo.add(node)
        for node in nodes:
            self.assertEqual(fifo.pop(), node)
        
class FIFO_DictTests(unittest.TestCase):

    def test_methods(self):
        fifo= data_structures.FIFO_dict()
        nodes= [data_structures.DoublyLinkedListNode(i) for i in range(1,11)]
        for node in nodes:
            fifo.add(node)
        for node in nodes:
            self.assertEqual(fifo.pop(), node)
        self.assertTrue(fifo.isempty())

class CircularBufferTests(unittest.TestCase):

    def test_init(self):
        cb= data_structures.CircularBuffer(size=10)
        self.assertEqual(cb.size, 10)
        self.assertEqual(tuple(cb.buffer), tuple([[]]*10))

    def test_add(self):
        cb= data_structures.CircularBuffer(size=10)
        for i in range(100):
            cb.add(i)
        self.assertEqual(cb.size, 10)
        self.assertEqual(tuple(cb.buffer), tuple(range(100)[-10:]))

    def test_pop(self):
        cb= data_structures.CircularBuffer(size=10)
        for i in range(100):
            cb.add(i)
        for i in range(100)[-10:]:
            self.assertEqual(cb.pop(), i)

class FIFO_CircTests(unittest.TestCase):

    def test_init(self):
        fifo= data_structures.FIFO_circ(size=10)
        self.assertEqual(fifo.size, 10)

class BST_Tests(unittest.TestCase):

    def test_init(self):
        bst= data_structures.BST(1)
        self.assertEqual(bst.data, 1)
        self.assertIsNone(bst.left)
        self.assertIsNone(bst.right)
        self.assertIsNone(bst.parent)

    def test_cmp(self):
        bst1= data_structures.BST(1)
        bst2= data_structures.BST(2)
        self.assertTrue(bst1 < bst2)

    def test_insert(self):
        bst10= data_structures.BST(10)
        bst20= data_structures.BST(20)
        bst10.insert(bst20)
        self.assertEqual(bst10.right, bst20)
        self.assertEqual(bst20.parent, bst10)
        bst0= data_structures.BST(0)
        bst10.insert(bst0)
        self.assertEqual(bst10.left, bst0)
        self.assertEqual(bst0.parent, bst10)
        bst5= data_structures.BST(5)
        bst10.insert(bst5)
        self.assertEqual(bst10.left.right, bst5)
        self.assertEqual(bst5.parent.parent.right, bst20)

    def test_iter(self):
        bst= data_structures.BST(10)
        sub_trees= [data_structures.BST(i) for i in \
                [5, 12, 4, 31, 2, 20]]
        for sub_tree in sub_trees:
            bst.insert(sub_tree)
        data_vals= []
        for sub_tree in bst:
            data_vals.append(sub_tree.data)
        self.assertTrue(sorting.is_sorted(data_vals))

        bst= data_structures.BST(random.random())
        sub_trees= [data_structures.BST(random.random()) for i in xrange(1000)]
        for sub_tree in sub_trees:
            bst.insert(sub_tree)
        data_vals= []
        for sub_tree in bst:
            data_vals.append(sub_tree.data)
        self.assertTrue(sorting.is_sorted(data_vals))
    
    def test_delete(self):
        bst= data_structures.BST(10)
        sub_trees= [data_structures.BST(i) for i in \
                [5, 12, 4, 31, 2, 20, 11]]
        for sub_tree in sub_trees:
            bst.insert(sub_tree)
        data_vals= []
        bst.delete(sub_trees[0])
        self.assertEqual(bst.left, sub_trees[2]) # delete left (one child)
        bst.delete(sub_trees[-1])
        self.assertIsNone(bst.right.left) # delete right-left (no children)
        bst.insert(sub_trees[-1])
        self.assertEqual(bst.right.left, sub_trees[-1])
        bst.delete(sub_trees[1])
        self.assertEqual(bst.right, sub_trees[-2])
        self.assertEqual(bst.right.left, sub_trees[-1])
        self.assertEqual(bst.right.right, sub_trees[3])
        self.assertIsNone(bst.right.right.right)
        self.assertEqual(bst.right.left.parent, sub_trees[-2])
        
    def test_convert_to_ll(self):
        bst= data_structures.BST(10)
        sub_trees= [data_structures.BST(i) for i in \
                [5, 12, 4, 31, 2, 20, 11]]
        for sub_tree in sub_trees:
            bst.insert(sub_tree)
        ll= bst.convert_to_ll()
        self.assertTrue(sorting.is_sorted([node.data for node in ll]))

class BST_PQTests(unittest.TestCase):

    def test_init(self):
        pq= data_structures.BSTPriorityQueue()
        self.assertIsNone(pq.parent)
        self.assertIsNone(pq.left)
        self.assertIsNone(pq.right)
        self.assertIsNone(pq.contents)
        pq= data_structures.BSTPriorityQueue(12, 'hello')
        self.assertEqual(pq.contents, 'hello')

    def test_add(self):
        pq= data_structures.BSTPriorityQueue()
        pq.add(12, 'hello')
        self.assertEqual(pq.contents, 'hello')
        pq.add(11, 'minimum')
        self.assertEqual(pq.left.contents, 'minimum')
        self.assertEqual(pq.left.parent, pq)

    def test_pop(self):
        pq= data_structures.BSTPriorityQueue()
        for i in [8, 2, 16, 4, 12, 34, 1]:
            pq.add(i, str(i))
        contents= []
        for _ in range(7):
            contents.append(pq.pop())
        self.assertTrue(sorting.is_sorted(contents))



#class GraphTests(unittest.TestCase):
#
#    def test_dijkstra2(self):
#        vertices= [Graph.Vertex(i) for i in range(10)]
#        G= Graph.Graph(vertices)
#        E1= Graph.Edge(vertices[0], vertices[1])
#        E2= Graph.Edge(vertices[1], vertices[3])
#        E3= Graph.Edge(vertices[0], vertices[3])
#        E1.distance= 1
#        E2.distance= 4
#        E3.distance= 2
#        G.add_edge(E1)
#        G.add_edge(E2)
#        G.add_edge(E3)
#        self.assertEqual(G.disktra2(vertices[0], vertices[3]), 2)
#
#    def sw_init(self):
#        SW= Graph.SmallWorldGraph()
#        self.assertEqual(len(SW), 10)
#
#class BinaryTreeTests(unittest.TestCase):
#
#    def test_dist(self):
#        """
#        Tests dist_to_root()
#        """
#        B= miscellany.BinaryTree()
#        answers= [0, 1, 1, 2, 2, 2, 2, 3, 3, 3]
#        for i in range(10):
#            self.assertEqual(B.dist_to_root(i), answers[i])
#
#    def test_len(self):
#        """
#        Tests __len__()
#        """
#        B= miscellany.BinaryTree([1, 2, 3])
#        self.assertEqual(len(B), 3)
#        B= miscellany.BinaryTree([])
#        self.assertEqual(len(B), 0)
#
#    def test_add(self):
#        """
#        Tests add(value)
#        """
#        greater_than= lambda x, y: x > y
#        B= miscellany.BinaryTree()
#        B.add(1, greater_than)
#        self.assertEqual(B.array, [1])
#
#    def test_get(self):
#        """
#        Tests __get__(index)
#        """
#        B= miscellany.BinaryTree([1, 2, 3])
#        self.assertEqual(B[2], 3)
#        B= miscellany.BinaryTree()
#        self.assertRaises(IndexError, B.__getitem__, 2)
#
#    def test_peek(self):
#        """
#        Tests peek()
#        """
#        B= miscellany.BinaryTree([1, 2, 3])
#        self.assertEqual(B.peek(), 1)
#
#    def test_child_functions(self):
#        """
#        Tests the children functions... breaks some code convention, but these
#        are extremeley simple
#        """
#        self.assertEqual(miscellany.BinaryTree.l_child(2), 5)
#        self.assertEqual(miscellany.BinaryTree.r_child(2), 6)
#        self.assertEqual(miscellany.BinaryTree.children(2), (5, 6))
#        self.assertEqual(miscellany.BinaryTree.parent(6), 2)
#        self.assertEqual(miscellany.BinaryTree.parent(5), 2)
#
#    def test_is_heaped(self):
#        """
#        Tests is_heaped()
#        """
#        greater_than= lambda x, y: x > y
#        B= miscellany.BinaryTree([1, 2, 3, 6, 12])
#        self.assertEqual(B.is_heaped(greater_than), True)
#        B= miscellany.BinaryTree([1, 2, 5, 3, 12, 3, 10])
#        self.assertEqual(B.is_heaped(greater_than), False)
#    
#    def test_set(self):
#        """
#        Tests __setitem__(key, value)
#        """
#        B= miscellany.BinaryTree([1, 2, 3])
#        B[2]= 12
#        self.assertEqual(B.array, [1, 2, 12])
#
#    def test_sift_down(self):
#        """
#        Tests sift_down(comp_fun, source)
#        """
#        less_than= lambda x, y: x < y
#        greater_than= lambda x, y: x > y
#        B= miscellany.BinaryTree([2, 4, 3])
#        B.sift_down(less_than)
#        self.assertEqual(B.array, [4, 2, 3])
#        B= miscellany.BinaryTree([2, 8, 7, 5, 1, 6, 6, 3, 1, 1, 0, 4, 2])
#        B.sift_down(less_than)
#        self.assertEqual(B.array, [8, 5, 7, 3, 1, 6, 6, 2, 1, 1, 0, 4, 2])
#        B= miscellany.BinaryTree([0.24911571079063477, 0.5037165999293618,
#            0.24728785679912424, 0.6761579999613856, 0.31004071491783713])
#        B.sift_down(greater_than, 2)
#        B.sift_down(greater_than, 1)
#        B.sift_down(greater_than, 0)
#        self.assertEqual(B.array, [0.24728785679912424, 0.31004071491783713,
#            0.24911571079063477, 0.6761579999613856, 0.5037165999293618])
#
#    def test_sift_up(self):
#        """
#        Tests sift_up(comp_fun, source)
#        """
#        less_than= lambda x, y: x < y
#        B= miscellany.BinaryTree([2, 1, 6])
#        B.sift_up(less_than)
#        self.assertEqual(B.array, [6, 1, 2])
#        B= miscellany.BinaryTree([12, 8, 6, 4, 2, 5, 3, 1, 3, 0, 2, 1, 0, 13])
#        B.sift_up(less_than)
#        self.assertEqual(B.array, [13, 8, 12, 4, 2, 5, 6, 1, 3, 0, 2, 1 ,0, 3])
#
#    def test_heapify(self):
#        """
#        Tests heapify(comp_fun)
#        """
#        greater_than= lambda x, y: x > y
#        import random
#        for i in range(50):
#            B= miscellany.BinaryTree([random.random() for _ in range(i+2)])
#            B.heapify(greater_than)
#            self.assertTrue(B.is_heaped(greater_than))
#
#class MaxHeapTests(unittest.TestCase):
#
#    def test_init(self):
#        import random
#        for i in range(10, 100):
#            B= miscellany.MaxHeap([random.random() for _ in range(i)])
#            self.assertTrue(B.is_heaped())
#
#class MinHeapTests(unittest.TestCase):
#    
#    def test_init(self):
#        import random
#        for i in range(10, 100):
#            B= miscellany.MinHeap([random.random() for _ in range(i)])
#            self.assertTrue(B.is_heaped())


if __name__ == '__main__':
    unittest.main()
