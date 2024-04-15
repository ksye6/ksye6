import numpy as np
import matplotlib.pyplot as plt
import torch
print(torch.cuda.is_available())

# 通过有限差分方法近似计算函数的梯度
def finiteDiff(fn, x,  parameters, delta, multiplier=1):
    shape = parameters.shape # parameters 的维度信息
    var = parameters.reshape(-1) # 将多维数组转换为一个一维的、按照原始数组元素顺序排列的数组
    diff = []
    for idx in range(len(var)):
        var[idx] += delta/2
        yplus = fn(x)
        var[idx] -= delta
        yminus = fn(x)
        varDiff = ((yplus-yminus)/delta*multiplier).sum()
        # 计算第 idx 个参数的梯度值。它通过计算正向偏差和负向偏差之间的差异，并除以 delta 进行归一化。.sum() 是对梯度向量进行求和，得到一个标量值。
        var[idx] += delta/2

        diff.append(varDiff)

    return np.array(diff).reshape(*shape) # 将列表 diff 转换为数组，并将其形状重新调整为与参数 parameters 相同的形状。最终返回计算得到的梯度数组。

class Node(object):
    
    def __init__(self, name, parameters=None):
        self.name = name
        self.parameters = parameters # 节点的参数列表，默认为 None。参数是节点在计算中需要使用的变量或权重。
        self.parameters_deltas = [None for _ in range(len(self.parameters))]
        # 节点参数的增量列表，默认为与参数列表相同长度的 None 元素列表。用于存储参数的变化量

class Linear(Node):
    
    def __init__(self, input_shape, output_shape, weight=None, bias=None):
        
        if weight is None: # 线性层的权重，默认为 None。如果未提供权重，则使用随机生成的服从标准正态分布的权重
            weight = np.random.randn(input_shape, output_shape)*0.01
        
        if bias is None: # 线性层的偏置，默认为 None。如果未提供偏置，则使用全零偏置
            bias = np.zeros(output_shape)
        
        super(Linear, self).__init__('linear', [weight, bias]) # 传递节点名称为 'linear'，参数列表为 [weight, bias]
    
    # 线性层节点的前向传播计算，将输入数据与权重矩阵相乘并加上偏置向量，得到线性层的输出。
    def forward(self, x):
        self.x = x
        return np.matmul(x, self.parameters[0]) + self.parameters[1]
    
    def backward(self, delta):
        self.parameters_deltas[0] = self.x.T.dot(delta) # 计算权重的梯度
        self.parameters_deltas[1] = np.sum(delta, 0) # 计算偏置的梯度
        return delta.dot(self.parameters[0].T)
        # 将上游传递下来的梯度与权重矩阵相乘，得到下游传递的梯度。这一步返回计算出的下游传递的梯度值。

class Sigmoid(Node):
    def __init__(self):
        super(Sigmoid, self).__init__('Sigmoid', [])
    
    def forward(self, x):
        self.x = x
        self.y = 1.0/(1.0+np.exp(-x))
        return self.y
    
    def backward(self, delta):
        return delta*((1-self.y)*self.y)

class Mean(Node):
    def __init__(self):
        super(Mean, self).__init__('mean', [])
        
    def forward(self, x):
        self.x = x
        return x.mean()
    
    def backward(self, delta):
        return delta * np.ones(self.x.shape) / np.prod(self.x.shape)

class MSE(Node):
    def __init__(self):
        super(MSE, self).__init__('MSE', [])
    
    def forward(self, x, l):
        self.l = l
        self.x = x
        return (x - l) ** 2
    
    def backward(self, delta):
        return delta * 2 * (self.x - self.l)

def net_forward(net, x, label):
    for node in net:
        if node.name == 'MSE':
            result = x
            x = node.forward(x, label)
        else:
            x = node.forward(x)
    return result, x

def net_backward(net):
    y_delta = 1.0
    for node in net[::-1]:
        y_delta = node.backward(y_delta)
    return y_delta


if __name__ == "__main__":
    
    # delta = 0.01
    # weight = np.random.randn(5, 10)*0.01
    # bias = np.random.randn(10)*0.01
    # 
    # linear = Linear(10, 5, weight, bias)
    # x = np.random.uniform(0.1, 1, [6, 5])
    # y = linear.forward(x)
    # 
    # # weightFD = finiteDiff(linear.forward, x, weight, delta)
    # # biasFD = finiteDiff(linear.forward, x, bias, delta)
    # # xFD = finiteDiff(linear.forward, x, x, delta)
    # 
    # xDiff = linear.backward(np.ones([6, 10]))
    # weightDiff = linear.parameters_deltas[0]
    # biasDiff = linear.parameters_deltas[1]
    # 
    # 
    # sigmoid = Sigmoid()
    # y = sigmoid.forward(x)
    # 
    # # xFD = finiteDiff(sigmoid.forward, x, x, delta)
    # 
    # xDiff = sigmoid.backward(np.ones([6, 5]))
    
    delta = 0.01
    
    net =[Linear(1,2), Sigmoid(), Linear(2,1), MSE(), Mean()]
    
    def target(x):
        return 3*x+1

    x = np.linspace(0,3,100)[..., None]   # 可以将一维数组转换为列向量形式
    
    label = target(x)
    
    pred, loss = net_forward(net, x, label)
    xDiff = net_backward(net)
    
    num_net_forward = lambda x: net_forward(net, x, label)[-1]
    xFD = finiteDiff(num_net_forward, x, x, delta)
    
    linear1_weightFD = finiteDiff(num_net_forward, x, net[0].parameters[0], delta)
    
    
    num_epoch = 300
    learning_rate = 1e-1
    batchSize = 30
    
    loss_list = []
    test_loss_list = []
    
    for epoch in range(num_epoch):
        x = np.random.uniform(-3,3,[batchSize,1])
        label = target(x)
        
        pred, loss = net_forward(net, x, label)
        loss_list.append(loss)
        
        _ = net_backward(net)
        
        for node in net:
            for p, p_delta in zip(node.parameters, node.parameters_deltas):
                p -= learning_rate * p_delta
        
        
        x = np.linspace(-3,3,200)[..., None]
        label = target(x)
        pred, loss = net_forward(net, x, label)
        test_loss_list.append(loss)
        
        plt.figure()
        plt.plot(x, label, 'o', label="Orig.")
        plt.plot(x, pred, '+', label="Pred.")
        plt.legend()
        plt.savefig("C://Users//张铭韬//Desktop//Pred.pdf")
        plt.show()
        
        plt.close()
        
    plt.figure()
    plt.plot(np.array(loss_list), label = "train")
    plt.plot(np.array(test_loss_list), label = "test")
    plt.legend()
    plt.savefig("C://Users//张铭韬//Desktop//loss_curve.pdf")
    plt.show()
    
    plt.close()        
    
        
        
        
        
        
        
        
        
        
        
        
