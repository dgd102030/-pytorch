import torch
from torch import nn
import numpy as np
class DNN_layer():
    def __init__(self, in_, out_, next = None, pre = None):
        self.w = torch.autograd.Variable(torch.randn(in_, out_).float(), requires_grad = True)
        self.b = torch.autograd.Variable(torch.ones(out_).float(), requires_grad = True)
        self.next = next
        self.pre = pre
    def forward(self, input):
        self.input = input
        hidden = torch.mm(input, self.w) + self.b #+ self.b1
        if self.next != None:
            self.output = torch.sigmoid(hidden)
        else:
            self.output = hidden
        return self.output

    def cal_grad(self, input, dz_next, w):
        m1 = torch.mm(dz_next, w.T)
        m2 = (1 - self.output)* self.output
        dz = m1 * m2
        self.dz = dz
        self.w.grad = torch.mm(input.T, dz)
        self.b.grad = dz.sum(dim = 0)

    def cal_grad_last_layer(self, input, y_tre):
        self.dz = (self.output - y_tre)
        self.w.grad = torch.mm(input.T, self.dz)
        self.b.grad = (self.output - y_tre).sum(dim=0)

    def backward(self, y_tre = None):
        if self.next != None:
            self.cal_grad(self.input, self.next.dz, self.next.w)
        else: ###最后一层
            self.cal_grad_last_layer(self.input, y_tre)

class DNN():
    def __init__(self, layer_nodes, y_tre):
        self.head_layer = DNN_layer(layer_nodes[0][0], layer_nodes[0][1])
        head = self.head_layer
        for i in range(1, len(layer_nodes)):
            layer = DNN_layer(layer_nodes[i][0], layer_nodes[i][1])
            layer.pre = head
            head.next = layer
            head = layer
        self.last_layer = head
        self.y_tre = y_tre
    def forward(self, input):
        layer = self.head_layer
        output = layer.forward(input)
        while layer.next != None:
            layer = layer.next
            output = layer.forward(output)
        self.last_layer = layer
        return output
    def backward(self):
        layer = self.last_layer
        layer.backward(self.y_tre)
        while layer.pre != None:
            layer = layer.pre
            layer.backward()

def loss_function(y_tre, y_pre):
    return 1/2*torch.sum((y_tre-y_pre)**2)

if __name__ == "__main__":
    input = torch.randn(20, 10)
    y = torch.from_numpy(np.array([0] *10 + [1]*10)).unsqueeze(1)
    nodes_list = [(10, 5), (5, 2), (2, 1)]
    model = DNN(nodes_list, y)
    out = model.forward(input)
    L = loss_function(y, out)
    L.backward()
    print(model.head_layer.next.w.grad[0,:5])
    print('-------------my backward-------------')
    model.backward()
    print(model.head_layer.next.w.grad[0, :5])
