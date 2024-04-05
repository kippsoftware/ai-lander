"""ExprNode - expression tree builder with gradient backpropagation.

The root represents the final operation in the expression. The leaf
nodes represent the individual variables or scalars used as operands.

For example, for the expression z = tanh(mx + b), we build this tree:

    z
      tanh
        +
          *
            m
            x
          b

For backpropagation, we distribute a "gradient" across the expression
graph, starting at the root, and set the gradient of each child,
moving recursively through the tree to the leaf nodes, which represent
the individual variable operands or scalar values of the expression.
Thus we can learn how changing each single input variable affects the
value of the entire expression.

Let the gradient at the root of the expression be 1.0. For each child
of a node, let the child gradient be the derivative of the expression
the node represents with respect to the child.

For example:
    a = 1 ; a.grad = 1.0
    b = c + 1 ; b.grad = 1.0, c.grad = 1.0
    e = 2
    d = 3e ; d.grad = 1.0, e.grad = 3
    g = 2
    h = 3
    f = gh ; f.grad = 1.0, g.grad = 3, h.grad = 2
    x = 1
    z = tanh(x) ; z.grad = 1.0, x.grad = 0.4200

For the purpose of course development, this code is derived from the
micrograd code by Andrej Karpathy. See LICENSE.

"""
import math

class ExprNode:
    """A differentiable expression node"""

    def __init__(self, name = '.', data = 0.0, children = ()):
        """
        name - string
        data - scalar either given or computed during tree construction
        children - tuple of ExprNode
        grad - holds local derivative for the backpropagation 
        backprop() - sets grad according to chain rule
        """
        self.name = name
        self.data = data
        self.children = children
        self.grad = 0.0
        self.backprop = lambda: None

    def __str__(self, indent = ''):
        out = [ f'{indent}{self.name} {self.data:0.4f} grad {self.grad:0.4f}']
        out.extend(child.__str__(indent + '  ') for child in self.children)
        return '\n'.join(out)

    def sort(self):
        """Recursively sort this node before its children nodes into a
        flat list with no duplicates"""
        nodes = []
        visited = set()
        def recur(node):
            nodes.append(node)
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    recur(child)
        recur(self)
        return nodes

    def __add__(self, y):
        """z = x + y"""
        x = self
        y = y if isinstance(y, ExprNode) else ExprNode(data=y)
        z = ExprNode('+', x.data + y.data, (x, y))
        def backprop():
            """d(x+y)dx = 1 dx
            d(x+y)dy = 1 dy"""
            x.grad += z.grad
            y.grad += z.grad
        z.backprop = backprop
        return z

    def __mul__(self, y):
        """z = x * y"""
        x = self
        y = y if isinstance(y, ExprNode) else ExprNode(data=y)
        z = ExprNode('*', x.data * y.data, (x, y))
        def backprop():
            """d(xy)dx = y dx
            d(xy)dy = x dy"""
            x.grad += y.data * z.grad
            y.grad += x.data * z.grad
        z.backprop= backprop
        return z

    def __neg__(self):
        """z = -x"""
        return self * -1

    def __sub__(self, y):
        """z = x - y"""
        return self + y * -1

    def tanh(self):
        """z = tanh(x)"""
        x = self
        z = ExprNode('tanh', math.tanh(x.data), (x,))
        def backprop():
            """dx = (1 - tanh**2(x)) dz """
            x.grad += (1 - z.data**2) * z.grad
        z.backprop = backprop
        return z

    def backward(self):
        """Propagate all the gradients through the expression graph rooted in self"""
        nodes = self.sort()
        for node in nodes:
            node.grad = 0.0
        # set base case dz/dz = 1.0
        self.grad = 1.0 
        for node in nodes:
            node.backprop()

if __name__ == '__main__':
    m = ExprNode('m', 1.0)
    x = ExprNode('x', 2.0)
    b = ExprNode('b', -3.0)
    z = (m * x + b).tanh()
    print(z)
    z.backward()
    print('backward')
    print(z)
