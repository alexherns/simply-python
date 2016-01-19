class Graph(dict):
    """
    A dictionary based representation of a graph
    """

    def __init__(self, vertices=[], edges=[]):
        """
        Initializes a Graph from a list of vertices and list of edges
        """
        for vertex in vertices:
            self.add_vertex(vertex)
        for edge in edges:
            self.add_edge(edge)

    def add_vertex(self, vertex):
        """
        Adds a vertex to the graph, with no edges defined
        """
        self[vertex]= {}

    def add_edge(self, edge):
        """
        Adds an edge to the graph. If edge already exists, creates it again
        """
        self[edge[0]][edge[1]]= edge
        self[edge[1]][edge[0]]= edge

    def get_edge(self, edge):
        """
        Attempts to get edge from the Graph. Returns None if edge not in Graph
        """
        if len(edge) != 2:
            raise TypeError
        try:
            return self[edge[0]][edge[1]]
        except KeyError:
            return None

    def remove_edge(self, edge):
        """
        Attempts to remove edge from Graph. Does nothing if edge not in Graph
        """
        if self.get_edge(edge):
            del self[edge[0]][edge[1]]
            del self[edge[1]][edge[0]]

    def vertices(self):
        """
        Returns a list of all vertices in the Graph
        """
        return self.keys()

    def edges(self):
        """
        Returns a list of all edges in the Graph
        """
        es= []
        for vertex1 in self.vertices():
            for vertex2 in self.out_vertices(vertex1):
                es.append(self[vertex1][vertex2])
        return es

    def out_vertices(self, vertex):
        """
        Returns a list of all vertices adjacent to vertex
        """
        return self[vertex].keys()

    def out_edges(self, vertex):
        """
        Returns a list of all edges incident to vertex
        """
        return self[vertex].values()

    def add_all_edges(self):
        """
        Connects all vertices in graph with edges, without adding loops
        """
        for n1 in self.vertices():
            for n2 in self.vertices():
                if n1 != n2:
                    self.add_edge((n1, n2))

    def minimum_other_vertex(self, vertex):
        """
        Returns other vertex with fewest number of out-edges
        """
        return min([(len(self.out_edges(v)), v) for v in self.vertices() if v != vertex])

    def add_regular_edges(self, n):
        """
        Adds edges to an edgeless graph to create a regular graph
        """
        self.check_empty()
        vertices= self.vertices()
        if len(vertices)*n % 2 != 0:
            raise IndexError
        from numpy.random import choice
        vertices= self.vertices()
        for vertex in vertices:
            print vertex
            while len(self.out_edges(vertex)) < n:
                other= self.minimum_other_vertex(vertex)[1]
                if len(self.out_edges(other)) < n:
                    self.add_edge((vertex, other))

    def check_empty(self):
        """
        If Graph is not empty, raises AttributeError
        """
        if self.size():
            raise AttributeError

    def order(self):
        """
        Returns the order of the Graph (# vertices)
        """
        return len(self.vertices())

    def size(self):
        """
        Returns the size of the Graph (# edges)
        """
        return len(self.edges())

    def clear(self):
        """
        Removes all vertices from Graph
        """
        for vertex in self.vertices():
            del self[vertex]

    def bfs_is_connected(self):
        """
        Uses Bread-First-Search to determine if Graph is strongly connected
        """
        q= Queue.Queue()
        origins= [self.vertices()[0]]
        traveled= set(origins)
        while origins:
            for o in origins:
                for child in self.out_vertices(o):
                    if child not in traveled:
                        q.put(child)
                        traveled.add(child)

            origins= []
            while not q.empty():
                origins.append(q.get())
        if len(traveled) == self.order():
            return True
        return False

    def dijkstra(self, source=None, destination=None):
        """
        Returns the shortest path length between source and destination without
        concern for edge distances
        """
        for vertex in self.vertices():
            vertex.d= sys.maxint
        if not source:
            source= self.vertices()[0]
        q= miscellany.FIFO_dict()
        source.d= 0
        q.append(source)
        while not q.isempty():
            source= q.pop()
            print source
            print source.d
            d= source.d
            for out_vertex in self.out_vertices(source):
                if out_vertex.d == sys.maxint:
                    out_vertex.d= d+1
                    q.append(out_vertex)
                if out_vertex == destination:
                    return out_vertex.d
        return d

    def disktra2(self, source=None, destination=None):
        """
        Returns the shortest path length between source and destination where
        edge distance is evaluated
        """
        if not source: source= self.vertices()[0]
        source.d= 0
        q= miscellany.MinPriorityQueue()
        q.add(source, source.d)
        visited= set()
        seen= set()
        while not q.isempty():
            source= q.pop()
            d= source.d
            if source == destination:
                return d
            visited.add(source)
            for out_vertex in self.out_vertices(source):
                if out_vertex in visited:
                    continue
                if out_vertex in seen:
                    out_vertex.d= min(out_vertex.d, source.d +
                            self[source][out_vertex].distance)
                else:
                    out_vertex.d= source.d + self[source][out_vertex].distance
                seen.add(out_vertex)
                q.add(out_vertex, out_vertex.d)
        return d

    def choose_random(self, exclude):
        """
        Returns a random node from the Graph that is NOT the current one
        """
        other_edges= list(set(self.vertices()) - set(exclude))
        return random.choice(other_edges)



class RandomGraph(Graph):
    """
    A placeholder class to be subclassed for various types of RandomGraphs
    """

    def __init__(self, n=1):
        """
        Initializes a RandomGraph with n vertices, and populates it according to function defined in subclass
        """
        vertices= [Vertex(i) for i in range(n)]
        for vertex in vertices:
            self.add_vertex(vertex)
        self.populate_graph()

    def populate_graph(self):
        """
        Placeholder function. You need to define this in each subclass
        """

    def maybe_add_edge(self, edge, p):
        """
        Will stochastically add edge according to probability (p). Returns True if added, and False otherwise
        """
        if random.random() < p:
            self.add_edge(edge)
            return True
        return False


class Erdos_Renyi(RandomGraph):
    """An implementation of a Erdos-Renyi type RandomGraph"""
    
    def __init__(self, n=1, p=1):
        self.p= float(p)
        self.n= n
        super(self.__class__, self).__init__(n=n)

    def populate_graph(self):
        vertices= self.vertices()
        for i in range(self.order()):
            for j in range(i+1, self.order()):
                self.maybe_add_edge((vertices[i], vertices[j]), self.p)



class SmallWorldGraph(RandomGraph):
    """
    An implementation of a Strogatz-Watts type RandomGraph
    """ 

    def __init__(self, size=10, K=2, beta=0.3):
        super(self.__class__, self).__init__(n=size)
        self.size= size
        self.fill_small_world(K=K, beta=beta)
        
    def fill_small_world(self, K, beta):
        self.fill_radial(K=K)
        self.fill_rewires(beta=beta)

    def fill_radial(self, K):
        vs= self.vertices()
        size= len(self)
        for i in range(size):
            for j in range(1, K/2 + 1):
                k= (i + j + 1)%size
                self.add_edge((vs[i], vs[k]))

    def fill_rewires(self, beta):
        vs= self.vertices()
        size= len(self)
        for i in vs:
            js= [j for j in self.out_vertices(i) if j > i]
            for j in js:
                if random.random() > beta:
                    continue
                other= self.choose_random(exclude=[i]+js)
                self.remove_edge((i, j))
                self.add_edge((i, other))

    def __len__(self):
        return self.size



class BarabasiAlbert(RandomGraph):
    """
    An implementation of a Barabasi-Albert type RandomGraph
    """

    def __init__(self, size=10, No=2):
        if No < 2:
            raise ValueError("No must be >= 2")
        if size < No:
            raise ValueError("size must be >= No")
        try:
            size= int(size)
            No= int(No)
        except ValueError:
            raise ValueError("size and No must be convertible to Type<int>")
        super(self.__class__, self).__init__(n=No)
        vs= self.vertices()
        for i in range(No):
            for j in range(i+1, No):
                self.add_edge((vs[i], vs[j]))
        for i in range(No, size):
            new_v= Vertex(i)
            self.add_vertex(new_v)
            self.connect_barabasi(new_v)

    def connect_barabasi(self, vertex):
        sum_kj= 2*self.size()
        for i in self.vertices():
            if i == vertex:
                continue
            ki= len(self.out_edges(i))
            pi= float(ki)/sum_kj
            print i, pi
            self.maybe_add_edge((vertex, i), pi)


class Vertex(object):
    """
    A Vertex in a Graph
    """

    def __init__(self, label=''):
        self. label= label

    def __repr__(self):
        return "<Vertex: {0}>".format(repr(self.label))

    __str__  = __repr__

    def __lt__(self, other):
        return self.label < other.label

    def __le__(self, other):
        return self.label <= other.label

    def __eq__(self, other):
        return self.label == other.label

    def __ne__(self, other):
        return not self.__eq__(other)

    def __gt__(self, other):
        return self.label > other.label

    def __ge__(self, other):
        return self.label >= other.label


class Edge(tuple):
    """
    An Edge in a Graph
    """

    def __new__(cls, n1, n2):
        return tuple.__new__(cls, (n1, n2))

    def __repr__(self):
        return "<Edge: ({0}, {1})".format(repr(self[0]), repr(self[1]))

    __str__ = __repr__



