# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data

# activation function
x = torch.linspace(-5, 5, 200)
x = Variable(x)
x_np = x.data.numpy()
y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()


# y_softmax = F.softmax(x)


class Net(torch.nn.Module):  # 继承 torch 的 Module
	def __init__(self, n_feature, n_hidden, n_output):
		super(Net, self).__init__()  # 继承 __init__ 功能
		# 定义每层用什么样的形式
		self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
		self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

	def forward(self, x):  # 这同时也是 Module 中的 forward 功能
		# 正向传播输入值, 神经网络分析出输出值
		x = F.relu(self.hidden(x))  # 激励函数(隐藏层的线性值)
		x = self.predict(x)  # 输出值
		return x

net2 = torch.nn.Sequential(
	torch.nn.Linear(1, 10),
	torch.nn.ReLU(),
	torch.nn.Linear(10, 1)
)
print(net2)
"""
Sequential (
  (0): Linear (1 -> 10)
  (1): ReLU ()
  (2): Linear (10 -> 1)
)
"""
net1 = Net(n_feature=1, n_hidden=10, n_output=1)
print(net1)  # net 的结构
"""
Net (
  (hidden): Linear (1 -> 10)
  (predict): Linear (10 -> 1)
)
"""

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)
# optimizer 是训练的工具
optimizer = torch.optim.SGD(net1.parameters(), lr=0.2)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式 (均方差)

for t in range(100):
	prediction = net1(x)  # 喂给 net 训练数据 x, 输出预测值
	loss = loss_func(prediction, y)  # 计算两者的误差
	optimizer.zero_grad()  # 清空上一步的残余更新参数值
	loss.backward()  # 误差反向传播, 计算参数更新值
	optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

# save model
torch.save(net1, 'net.pkl')  # 保存整个网络
torch.save(net1.state_dict(), 'net_params.pkl')   # 只保存网络中的参数 (速度快, 占内存少)

def restore_net():
	# restore entire net1 to net2
	net2 = torch.load('net.pkl')
	prediction = net2(x)

def restore_params():
	# 新建 net3
	net3 = torch.nn.Sequential(
		torch.nn.Linear(1, 10),
		torch.nn.ReLU(),
		torch.nn.Linear(10, 1)
	)
	# 将保存的参数复制到 net3
	net3.load_state_dict(torch.load('net_params.pkl'))
	prediction = net3(x)

# use batch
BATCH_SIZE = 5      # 批训练的数据个数

x = torch.linspace(1, 10, 10)       # x data (torch tensor)
y = torch.linspace(10, 1, 10)       # y data (torch tensor)

# 先转换成 torch 能识别的 Dataset
torch_dataset = Data.TensorDataset(x, y)

# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
	dataset=torch_dataset,      # torch TensorDataset format
	batch_size=BATCH_SIZE,      # mini batch size
	shuffle=True,               # 要不要打乱数据 (打乱比较好)
	num_workers=2,              # 多线程来读数据
)

for epoch in range(3):   # 训练所有!整套!数据 3 次
	for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
		# 假设这里就是你训练的地方...
		# 打出来一些数据
		print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
			  batch_x.numpy(), '| batch y: ', batch_y.numpy())

# 为每个优化器创建一个 net
net_SGD = Net()
net_Momentum = Net()
net_RMSprop = Net()
net_Adam = Net()
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]
LR = 0.01
# different optimizers
opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]
