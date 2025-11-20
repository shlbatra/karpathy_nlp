import random
import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''): # children set up graph
        self.data = data
        self.grad = 0.0 # No effect on output
        self._backward = lambda: None # Build the chain rule here
        self._prev = set(_children)
        self._op = _op
        self.label = label
        
    def __repr__(self):
        return f"Value(data={self.data})"
    # Important to know how to get local derivatives even if use complex fn.
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) # consider int if not Value
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward(): # gradients just get copied backwards for addition operation
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + (-other)
    

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) # consider int if not Value
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other): # check if other * self so 2*Value -> use here as 2.__mul__ not recognize Value type
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)) # Make sure power is either int or float
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    
    def __truediv__(self, other):
        return self * other ** -1
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self, ), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out
    
    # a/b = a* (1/b) = a * (b**-1)
    
    def backward(self):
        visited = set()
        topo = []
        # start at o and then go through all children from right to left
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v) # current node add only after all children have been processed
        build_topo(self)
        self.grad = 1 # starting point
        for node in reversed(topo):
            node._backward()

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin, nonlin = True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
        self.nonlin = nonlin

    def __call__(self, x):
        # wx + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh() if self.nonlin else act
        return out
    
    def parameters(self):
        return self.w + [self.b]



class Layer(Module):
    
    def __init__(self, nin, nout): # nin number of dimensions per neuron, nout number of neurons per layer
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):

    def __init__(self, nin, nouts): # nin is number of inputs, nouts -> list -> size of all layers in MLP
        sz = [nin] + nouts

        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]