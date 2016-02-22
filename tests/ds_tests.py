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

    def test_uniquify(self):
        nodes= [data_structures.LinkedListNode(i) for i in range(10)]
        nodes= nodes + [data_structures.LinkedListNode(i) for i in range(10)]
        nodes= nodes + [data_structures.LinkedListNode(i) for i in range(10)]
        self.assertEqual(len(nodes), 30)
        ll1= data_structures.LinkedList()
        ll2= data_structures.LinkedList()
        for node in nodes:
            ll1.insert(node)
        for node in nodes[20:]:
            ll2.insert(node)
        ll1.uniquify()
        ll2.uniquify()
        self.assertEqual([node for node in ll1],
                [node for node in ll2])
    
    def test_uniquify_faster(self):
        nodes= [data_structures.LinkedListNode(i) for i in range(10)]
        nodes= nodes + [data_structures.LinkedListNode(i) for i in range(10)]
        nodes= nodes + [data_structures.LinkedListNode(i) for i in range(10)]
        self.assertEqual(len(nodes), 30)
        ll1= data_structures.LinkedList()
        ll2= data_structures.LinkedList()
        for node in nodes:
            ll1.insert(node)
        for node in nodes[20:]:
            ll2.insert(node)
        ll1.uniquify_faster()
        ll2.uniquify_faster()
        self.assertEqual([node for node in ll1],
                [node for node in ll2])

    def test_kth_back(self):
        nodes= [data_structures.LinkedListNode(i) for i in range(6)]
        ll= data_structures.LinkedList()
        for node in nodes:
            ll.insert(node)
        for k in range(1, 7):
            self.assertEqual(nodes[k-1], ll.kth_from_end(k))

    def test_list_init(self):
        ll= data_structures.LinkedList()
        l= [0, 1, 2, 3, 4, 5, 6]
        ll.build_from_list(l)
        for i in l:
            item= ll.pop()
            self.assertEqual(item, data_structures.LinkedListNode(i))

    def test_partition(self):
        tests= [
                ([1], [1]),
                ([8], [8]),
                ([1, 8], [1, 8]),
                ([8, 1], [1, 8]),
                ([1, 2, 8, 9], [2, 1, 8, 9]),
                ([9, 8, 1, 2], [2, 1, 9, 8]),
                ([1, 2, 3], [3, 2, 1]),
                ([8, 9, 10], [8, 9, 10])
                ]
        self.assertTrue(data_structures.LinkedListNode(2)<7)
        for test_list, answer_list in tests:
            lltest= data_structures.LinkedList()
            llanswer= data_structures.LinkedList()
            lltest.build_from_list(test_list)
            llanswer.build_from_list(answer_list)
            lltest.partition(7)
            for _ in range(len(test_list)):
                self.assertEqual(lltest.pop(), llanswer.pop())
        
    def test_palindromic(self):
        tests= [
                ([0, 1, 0], True),
                ([0, 1, 2, 1, 0], True),
                ([0], True),
                ([0, 2, 1], False),
                ([0, 2], False),
                ([0, 2, 1, 4], False),
                ([0, 2, 1, 4, 1, 2, 0], True),
                ([1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 2], False)
                ]
        for l, outcome in tests:
            ll= data_structures.LinkedList()
            ll.build_from_list(l)
            self.assertEqual(ll.is_palindrome(), outcome)

    def test_tolist(self):
        ll= data_structures.LinkedList()
        self.assertEqual(ll.to_list(), [])
        ll.insert(data_structures.LinkedListNode(0))
        self.assertEqual(ll.to_list(), [0])
        ll.insert(data_structures.LinkedListNode(1))
        ll.insert(data_structures.LinkedListNode(2))
        self.assertEqual(ll.to_list(), [2, 1, 0])
        


class StackTests(unittest.TestCase):
    
    def test_add(self):
        stack= data_structures.Stack()
        stack.add(0)
        self.assertEqual(stack.pop(), 0)
        stack.add(1)
        self.assertEqual(stack.pop(), 1)
        stack.add(2)
        stack.add(3)
        self.assertEqual(stack.pop(), 3)
        self.assertEqual(stack.pop(), 2)


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


class Queue_Tests(unittest.TestCase):

    def test_init(self):
        queue= data_structures.Queue()
        self.assertIsNone(queue.head)
        self.assertIsNone(queue.tail)

    def test_enqueue(self):
        queue= data_structures.Queue()
        nodes= [data_structures.LinkedListNode(i) for i in range(1,11)]
        queue.enqueue(nodes[0])
        self.assertEqual(queue.head, nodes[0])
        self.assertEqual(queue.tail, nodes[0])
        queue.enqueue(nodes[1])
        self.assertEqual(queue.head, nodes[0])
        self.assertEqual(queue.tail, nodes[1])
        for node in nodes[2:]:
            queue.enqueue(node)
        self.assertEqual(queue.head, nodes[0])
        self.assertEqual(queue.tail, nodes[-1])

    def test_dequeue(self):
        queue= data_structures.Queue()
        nodes= [data_structures.LinkedListNode(i) for i in range(1, 11)]
        queue.enqueue(nodes[0])
        self.assertEqual(queue.dequeue(), nodes[0])
        self.assertIsNone(queue.head)
        self.assertIsNone(queue.tail)
        for node in nodes:
            queue.enqueue(node)
        for node in nodes:
            self.assertEqual(queue.dequeue(), node)
        self.assertIsNone(queue.head)
        self.assertIsNone(queue.tail)


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

    def test_build_from_sorted(self):
        l= [1,2]
        bst= data_structures.BST.build_from_sorted(l)
        self.assertEqual(bst.data, 2)
        self.assertEqual(bst.left.data, 1)
        self.assertEqual(bst.right, None)
        l= [1, 2, 3, 4]
        bst= data_structures.BST.build_from_sorted(l)
        self.assertEqual(bst.data, 4)
        self.assertEqual(bst.left.right, 3)
        self.assertEqual(bst.right, None)
        l= range(8)
        bst= data_structures.BST.build_from_sorted(l)
        ll= bst.convert_to_ll()
        self.assertEqual(ll.head, 0)

    def test__balanced(self):
        bst= data_structures.BST.build_from_sorted([1, 2])
        self.assertEqual(bst._isbalanced(), (1, 2))
        bst= data_structures.BST.build_from_sorted([1, 2, 3])
        self.assertEqual(bst._isbalanced(), (2,2))
        bst= data_structures.BST.build_from_sorted([1, 2, 3, 4])
        self.assertEqual(bst._isbalanced(), (False, False))
        bst= data_structures.BST.build_from_sorted([1, 2, 3, 4, 5])
        self.assertEqual(bst._isbalanced(), (2, 3))

    def test_isBalanced(self):
        tests= [([1, 2], True),
                ([1, 2, 3], True),
                ([1, 2, 3, 4], False),
                (range(1,7), True),
                (range(1,13), False),
                (range(1,16), True),
                (range(1,17), False)]
        for test_list, response in tests:
            bst= data_structures.BST.build_from_sorted(test_list)
            self.assertEqual(bst.isBalanced(), response)

    def test_finder(self):
        bst= data_structures.BST(0)
        self.assertEqual(bst.find_node(0), bst)
        self.assertEqual(bst.find_node(1), None)
        self.assertEqual(bst.find_node(-1), None)
        bst.insert(data_structures.BST(-1))
        bst.insert(data_structures.BST(1))
        self.assertEqual(bst.find_node(-1), bst.left)
        self.assertEqual(bst.find_node(1), bst.right)
        self.assertEqual(bst.find_node(2), None)
        bst= data_structures.BST.build_from_sorted(range(10))
        for i in range(10):
            self.assertEqual(bst.find_node(i), data_structures.BST(i))
        
    def test_minimum(self):
        bst= data_structures.BST(0)
        bst.insert(data_structures.BST(1))
        self.assertEqual(bst.find_minimum(), bst)
        bst.insert(data_structures.BST(-1))
        self.assertEqual(bst.find_minimum(), data_structures.BST(-1))
        bst= data_structures.BST.build_from_sorted(range(10))
        self.assertEqual(bst.right.find_minimum(), data_structures.BST(8))

    def test_next(self):
        bst= data_structures.BST(0)
        self.assertEqual(bst.find_next(), None)
        bst.insert(data_structures.BST(-1))
        self.assertEqual(bst.find_next(), None)
        bst.insert(data_structures.BST(1))
        self.assertEqual(bst.find_next(), data_structures.BST(1))
        self.assertEqual(bst.left.find_next(), bst)
        bst= data_structures.BST.build_from_sorted(range(101))
        for i in range(100):
            self.assertEqual(bst.find_node(i).find_next(), bst.find_node(i+1))

    def test_listbylevel(self):
        bst= data_structures.BST(5)
        self.assertIsInstance(bst.listByLevel(), list)
        self.assertIsInstance(bst.listByLevel()[0], data_structures.LinkedList)
        self.assertIsInstance(bst.listByLevel()[0].head,
                data_structures.LinkedListNode)
        self.assertEqual(bst.listByLevel()[0].head.data, bst)
        bst= data_structures.BST.build_from_sorted([4, 5, 6])
        self.assertEqual(bst.listByLevel()[0].head.data, bst)
        self.assertEqual(bst.listByLevel()[1].head.data, bst.right)
        self.assertEqual(bst.listByLevel()[1].head.child.data, bst.left)
        bst= data_structures.BST.build_from_sorted(range(10))
        self.assertEqual(bst.listByLevel()[0].head.data, bst)
        self.assertEqual(bst.listByLevel()[3].head.child.data,
                bst.left.right.left)

    def test_mrca(self):
        bst= data_structures.BST.build_from_sorted(range(1, 16))
        self.assertEqual(data_structures.BST.mrca(bst.left, bst.right), bst)
        self.assertEqual(data_structures.BST.mrca(bst.left.left, bst.right), bst)
        self.assertEqual(data_structures.BST.mrca(bst.left.right.left, bst.right), bst)
        self.assertEqual(data_structures.BST.mrca(bst.left.left,
            bst.left.right), bst.left)
        self.assertEqual(data_structures.BST.mrca(bst.left.left.left,
            bst.right.right.right), bst)
        self.assertEqual(data_structures.BST.mrca(bst.right.left.left,
            bst.left.right.right), bst)

    def test_subtree(self):
        bfs= data_structures.BST.build_from_sorted
        bst= bfs(range(15))
        self.assertEqual(bst.containsSubTree(bfs(range(3))), True)
        self.assertEqual(bst.containsSubTree(bfs(range(4, 7))), True)
        self.assertEqual(bst.containsSubTree(bfs(range(7))), True)
        self.assertEqual(bst.containsSubTree(bfs(range(8, 15))), True)
        self.assertEqual(bst.containsSubTree(bfs(range(4))), False)
        self.assertEqual(bst.containsSubTree(bfs(range(5))), False)


class BTree_Tests(unittest.TestCase):

    def test_isBST1(self):
        bt= data_structures.BTree
        tree= bt(0)
        tree.left= bt(-1)
        tree.right= bt(1)
        self.assertEqual(tree.isBST1(), True)
        tree.left.right= bt(-0.5)
        tree.left.left= bt(-2)
        tree.right.left= bt(0.5)
        tree.right.right= bt(2)
        self.assertEqual(tree.isBST1(), True)
        tree.right.right= bt(-3)
        self.assertEqual(tree.isBST1(), False)

    def test_isBST2(self):
        bt= data_structures.BTree
        tree= bt(0)
        tree.left= bt(-1)
        tree.right= bt(1)
        self.assertEqual(tree.isBST2(), True)
        tree.left.right= bt(-0.5)
        tree.left.left= bt(-2)
        tree.right.left= bt(0.5)
        tree.right.right= bt(2)
        self.assertEqual(tree.isBST2(), True)
        tree.right.right= bt(-3)
        self.assertEqual(tree.isBST2(), False)

    def test_mrca(self):
        bt= data_structures.BTree
        tree= bt(0)
        tree.left= bt(-1)
        tree.right= bt(1)
        tree.left.parent= tree
        tree.right.parent= tree
        self.assertEqual(bt.mrca(tree, tree), tree)
        self.assertEqual(bt.mrca(tree.left, tree), tree)
        self.assertEqual(bt.mrca(tree, tree.right), tree)
        self.assertEqual(bt.mrca(tree.left, tree.right), tree)
        self.assertEqual(bt.mrca(tree.right, tree.left), tree)

    def test_matches(self):
        bt= data_structures.BTree
        tree= bt(0)
        self.assertEqual(tree.fullmatch(bt(0)), True)
        self.assertEqual(tree.fullmatch(bt(1)), False)
        tree2= bt(0)
        tree2.right= bt(1)
        tree2.right.parent= tree2
        self.assertEqual(tree.fullmatch(tree2), False)


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


class GraphTests(unittest.TestCase): 
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
    pass


class MaxHeapTests(unittest.TestCase):

    def test_init(self):
        import random
        for i in range(10, 100):
            B= data_structures.MaxHeap([random.random() for _ in range(i)])
            self.assertTrue(B.is_heaped())


class MinHeapTests(unittest.TestCase):
    
    def test_init(self):
        import random
        for i in range(10, 100):
            B= data_structures.MinHeap([random.random() for _ in range(i)])
            self.assertTrue(B.is_heaped())


class LinearHashTests(unittest.TestCase):

    def test_init(self):
        table= data_structures.LinearMap()
        self.assertTrue(table.map == [])

    def test_add(self):
        table= data_structures.LinearMap()
        table.add('one', 1)
        table.add('two', 2)
        self.assertEqual(table.map[0], ('one', 1))
        self.assertEqual(table.map[1], ('two', 2))

    def test_get(self):
        table= data_structures.LinearMap()
        self.assertRaises(KeyError, table.get, 'one')
        table.add('one', 1)
        self.assertEqual(table.get('one'), 1)
        self.assertRaises(KeyError, table.get, 'two')
        table.add('two', 2)
        self.assertEqual(table.get('two'), 2)


class BetterHashTests(unittest.TestCase):

    def test_init(self):
        table= data_structures.BetterMap()
        self.assertEqual(len(table.maps), 2)
        self.assertEqual(len(table.maps), table.size)
        self.assertTrue(isinstance(table.maps[0], data_structures.LinearMap))

    def test_hashing(self):
        table= data_structures.BetterMap()
        self.assertEqual(table.find_map('hello'), hash('hello') % table.size)

    def test_add(self):
        table= data_structures.BetterMap()
        table.add('one', 1)
        table.add('two', 2)
        self.assertEqual(table.maps[table.find_map('one')].map[0], ('one', 1))

    def test_get(self):
        table= data_structures.BetterMap()
        self.assertRaises(KeyError, table.get, 'one')
        table.add('one', 1)
        self.assertEqual(table.get('one'), 1)
        self.assertRaises(KeyError, table.get, 'two')
        table.add('two', 2)
        self.assertEqual(table.get('two'), 2)


class HashTableTests(unittest.TestCase):

    def test_init(self):
        table= data_structures.HashTable()
        self.assertTrue(table.size == 0)
        self.assertTrue(isinstance(table.maps, data_structures.BetterMap))

    def test_add(self):
        table= data_structures.HashTable()
        table.add('one', 1)
        self.assertEqual(table.maps.maps[table.maps.find_map('one')].map[0],
                ('one', 1))

    def test_get(self):
        table= data_structures.HashTable()
        self.assertRaises(KeyError, table.get, 'one')
        table.add('one', 1)
        self.assertEqual(table.get('one'), 1)
        self.assertRaises(KeyError, table.get, 'two')

    def test_resize(self):
        table= data_structures.HashTable()
        for i in range(100):
            table.add(str(i), i)
        self.assertEqual(table.size, 100)
        self.assertEqual(table.maps.size, 128)


class StringBufferTests(unittest.TestCase):

    def test_init(self):
        sb= data_structures.StringBuffer()
        self.assertEqual([], sb.buffer)
        sb= data_structures.StringBuffer('hello')
        self.assertEqual(['h', 'e', 'l', 'l', 'o'], sb.buffer)

    def test_append(self):
        sb= data_structures.StringBuffer()
        sb.append('hello')
        self.assertEqual(['h', 'e', 'l', 'l', 'o'], sb.buffer)
        sb.append('world')
        self.assertEqual(['h', 'e', 'l', 'l', 'o', 'w', 'o', 'r', 'l', 'd'], sb.buffer)

    def test_getitem(self):
        sb= data_structures.StringBuffer()
        self.assertRaises(IndexError, sb.__getitem__, 0)
        sb= data_structures.StringBuffer('hello')
        self.assertEqual('h', sb[0])
        self.assertRaises(IndexError, sb.__getitem__, 12)

    def test_getslice(self):
        sb= data_structures.StringBuffer()
        self.assertEqual('', str(sb[0:5]))
        sb= data_structures.StringBuffer('hello')
        self.assertEqual('hell', str(sb[0:4]))
        self.assertEqual('hello', str(sb[0:10]))
        self.assertEqual('', str(sb[10:20]))
        self.assertEqual('hello', str(sb[:]))

    def test_setitem(self):
        sb= data_structures.StringBuffer()
        self.assertRaises(IndexError, sb.__setitem__, 0, 'a')
        sb= data_structures.StringBuffer('hello')
        sb[2]= 'e'
        self.assertEqual('heelo', str(sb))
        self.assertRaises(IndexError, sb.__setitem__, 10, 'a')

    def test_setslice(self):
        sb= data_structures.StringBuffer()
        self.assertRaises(AssertionError, sb.__setslice__, 0, 5, 'he')
        sb[0:2]= 'he'
        self.assertEqual(sb.buffer, ['h', 'e'])

    def test_concat(self):
        sb= data_structures.StringBuffer('hello')
        wb= data_structures.StringBuffer('world')
        self.assertEqual(str(sb+'world'), 'helloworld')
        self.assertEqual(str(sb+wb), 'helloworld')


class BitVectorTests(unittest.TestCase):

    def test_init(self):
        bv= data_structures.BitVector(size=100)
        self.assertEqual(bv.size, 100)
        self.assertEqual(len(bv.array), 4)
        self.assertEqual(bv.array[0], 0)
        bv2= data_structures.BitVector(size=100, preset=True)
        self.assertEqual(bv2.array[0], 4294967295)

    def test_checkbit(self):
        bv= data_structures.BitVector(size=100)
        for i in range(100):
            self.assertEqual(bv.checkBit(i), 0)
            self.assertEqual(bv[i], 0)
        bv2= data_structures.BitVector(size=100, preset=True)
        for i in range(100):
            self.assertEqual(bv2.checkBit(i), 1)
            self.assertEqual(bv2[i], 1)

    def test_checkslice(self):
        bv= data_structures.BitVector(size=100)
        self.assertEqual(bv[:20], [0 for _ in range(20)])
        bv2= data_structures.BitVector(size=150, preset=True)
        self.assertEqual(bv2[100:130], [1 for _ in range(30)])

    def test_checkset(self):
        bv= data_structures.BitVector(size=50)
        bv.setBit(30)
        self.assertEqual(bv[29], 0)
        self.assertEqual(bv[30], 1)
        self.assertEqual(bv[31], 0)
        bv[24]= 12
        self.assertEqual(bv[24], 1)
        bv[24]= False
        self.assertEqual(bv[24], 0)

    def test_clear(self):
        bv= data_structures.BitVector(size=40, preset=True)
        bv.clearBit(22)
        self.assertEqual(bv[22], 0)

    def test_switch(self):
        bv= data_structures.BitVector(size= 23, preset=True)
        self.assertEqual(bv[12], 1)
        bv.switchBit(12)
        self.assertEqual(bv[12], 0)
        bv.switchBit(12)
        self.assertEqual(bv[12], 1)

if __name__ == '__main__':
    unittest.main()
