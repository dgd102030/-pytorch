import torch
from torch import nn
import numpy as np
class DNN(nn.Module):
    def __init__(self, in_, hidden_, out_):
        self.w1 = torch.autograd.Variable(torch.randn(in_, hidden_).float(), requires_grad = True)
        self.b1 = torch.autograd.Variable(torch.ones(hidden_).float(), requires_grad = True)
        self.w2 = torch.autograd.Variable(torch.randn(hidden_, out_).float(), requires_grad = True)
        self.b2 = torch.autograd.Variable(torch.ones(out_).float(), requires_grad=True)
    def forward(self, input):
        self.hidden = torch.mm(input, self.w1) + self.b1 #+ self.b1
        self.phi = torch.sigmoid(self.hidden)
        self.output = torch.mm(self.phi, self.w2) + self.b2
        return self.output
    def cal_grad_w1(self, x, y_tre):
        m1 = (self.output - y_tre)
        m2 = torch.mm(m1, self.w2.T)
        m3 = m2 * (1 - self.phi)*self.phi
        m4 = torch.mm(x.T, m3)
        self.w1.grad = m4
        self.b1.grad = m3.sum(dim = 0)
    def cal_grad_w2(self, y_tre):
        m1 = torch.mm(self.phi.T,(self.output - y_tre))
        self.w2.grad = m1
        self.b2.grad = (self.output - y_tre).sum(dim = 0)
    def backward(self, input, y_tre):
        self.cal_grad_w1(input, y_tre)
        self.cal_grad_w2(y_tre)

def loss_function(y_tre, y_pre):
    return 1/2*torch.sum((y_tre-y_pre)**2)
if __name__ == "__main__":
    test1 = torch.randn(20, 10)
    y = torch.from_numpy(np.array([0] *10 + [1]*10)).unsqueeze(1)
    k = DNN(10, 5, 1)
    y_pre = k.forward(test1)
    L = loss_function(y, y_pre)
    L.backward()
    print(k.w1.grad[0,:5])
    print('-------------my backward-------------')
    k.backward(test1, y)
    print(k.w1.grad[0,:5])



