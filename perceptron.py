import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

x1 = torch.from_numpy(np.random.rand(500)+5)
x2 = torch.from_numpy(np.random.rand(500) + 3)
y = 5*x1**2 + 3*x2**2
def torch_(x1, x2, y, lr = 0.0001):
	w1 = Variable(torch.tensor(1).float(), requires_grad = True)
	w2 = Variable(torch.tensor(1).float(), requires_grad = True)
	optim = torch.optim.SGD([w1, w2], lr=lr) #构建优化器
	l = []
	for i in range(x1.shape[0]):
		y_ = w1 * x1[i]**2 + w2*x2[i]**2
		loss = (y[i] - y_)**2 #损失函数
		optim.zero_grad()     #梯度清零
		loss.backward()       #梯度计算
		optim.step()          #参数更新
		s = sum(w1*x1**2 + w2 * x2**2 - y)**2/x1.shape[0]
		l.append(s.item())
	plt.plot(range(len(l)), l)
	plt.show()
	print(w1)
	print(w2)
def loss(w1, w2, x1, x2, y):
	n = x1.shape
	l = (w1*x1**2 + w2*x2**2 - y)**2
	return sum(l)/n[0] if len(n)>0 else l 
def optim(w1, w2, x1, x2, y, lr):
	n = x1.shape
	w1_grad = 2*(w1*x1**2 + w2*x2**2 - y)*x1**2
	w1_grad = sum(w1_grad)/n[0] if len(n)>0 else w1_grad
	w2_grad = 2*(w1*x1**2 + w2*x2**2 - y)*x2**2
	w2_grad = sum(w2_grad)/n[0] if len(n)>0 else w2_grad
	w1 = w1 - lr * w1_grad
	w2 = w2 - lr * w2_grad
	return w1, w2
def update(x1,x2, y, lr):
	l = []
	w1 = torch.tensor(1).float()
	w2 = torch.tensor(1).float()
	for i in range(10):
		w1, w2 = optim(w1, w2, x1[i*10:(i+1)*10], x2[i*10:(i+1)*10], y[i*10:(i+1)*10], lr = lr)
		s = sum(w1*x1**2 + w2 * x2**2 - y)**2/x1.shape[0] #计算每次更新之后总损失
		l.append(s.item())
	plt.plot(range(len(l)), l)
	plt.ylabel('loss')
	plt.show()
	print(w1)   #检验更新之后参数和预期是否一致
	print(w2)
torch_(x1,x2, y, lr=0.0001)